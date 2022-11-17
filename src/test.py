import argparse
import logging
import os

import torch
import transformers as t
from tqdm import tqdm, trange

from dataset import get_test_dataloaders, task_to_keys
from modeling.palbert_fast import (AlbertPABEEForSequenceClassification,
                                   PAlbertForSequenceClassification)
from trainer import Trainer
from utils.set_deterministic import set_deterministic_mode

task_to_labels = {
    "cola": {0: "0", 1: "1"},
    "sst2": {0: "0", 1: "1"},
    "mrpc": {0: "0", 1: "1"},
    "qqp": {0: "0", 1: "1"},
    "mnli": {0: "entailment", 1: "neutral", 2: "contradiction"},
    "qnli": {0: "entailment", 1: "not_entailment"},
    "rte": {0: "entailment", 1: "not_entailment"},
}


def main(args):

    if args.name == "auto":
        pbar_tasks = tqdm(task_to_keys.keys(), total=8, leave=False)
    else:
        pbar_tasks = tqdm([args.name])
    for task in pbar_tasks:
        pbar_tasks.set_description(desc=f"Currently testing {task}")
        for c_seed in trange(42, 47, leave=False):
            model_path = f"{args.prefix}_{task}_{c_seed}/pytorch_model.bin"
            args.name = task
            run_task(args, model_path=model_path, c_seed=c_seed)


def run_task(args, model_path, c_seed):
    device = torch.device(args.device)
    tokenizer = t.AutoTokenizer.from_pretrained(args.type)
    dataloaders = get_test_dataloaders(args.name, tokenizer)
    num_labels = 1 if args.name == "stsb" else (3 if args.name == "mnli" else 2)
    config = t.AlbertConfig.from_pretrained(
        args.type,
        num_labels=num_labels,
        output_hidden_states=False,
        num_hidden_layers=args.num_hidden_layers,
    )
    if args.pondering:
        model = PAlbertForSequenceClassification.from_pretrained(
            args.type, config=config
        )
        model.albert.set_threshold(args.ponder_threshold)
        if args.init_sinusoidal:
            model.albert.init_layer_pos_embeddings()
        model.classifiers.set_lambda_layer_arch(
            num_layers=args.num_lambda_layers,
            rnn=args.rnn_lambda,
            hidden_dim=model.config.hidden_size * 2
            if "cat" in args.prev_state_influence
            else model.config.hidden_size,
        )
    elif args.pabee:
        model = AlbertPABEEForSequenceClassification.from_pretrained(
            args.type, config=config
        )
        model.set_patience(args.pabee_patience)
    else:
        model = t.AlbertForSequenceClassification.from_pretrained(
            args.type,
            config=config,
        )
    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path))
    else:
        print(
            f"Warning! Can't find model for dataset {args.name}. Using dummy output to pack predictions"
        )

    if args.fp16:
        scaler = torch.cuda.amp.GradScaler()

    trainer = Trainer(
        model=model,
        optimizer=None,
        scheduler=None,
        device=device,
        metric=None,
        scaler=scaler,
        is_regression=args.name == "stsb",
    )

    results = trainer.test(
        loaders=dataloaders,
        pabee=args.pabee,
        pondering=args.pondering,
        use_layer_pos_encoding=args.use_layer_pos_encoding,
        use_prev_hiddens=args.use_prev_hiddens,
        prev_state_influence=args.prev_state_influence,
        exit_criteria=args.exit_criteria,
    )
    for result_key, result in results.items():
        if isinstance(result, float):
            print(result_key + ": " + str(result))
            continue
        with open(
            f"{args.prefix}_{args.name}_{result_key}_{c_seed}.tsv", "w"
        ) as writer:
            for index, item in enumerate(result):
                if args.name == "stsb":
                    item = max(0, min(item, 5))
                    writer.write("%d\t%3.3f\n" % (index, item))
                else:
                    item = task_to_labels[args.name][item]
                    writer.write("%d\t%s\n" % (index, item))


if __name__ == "__main__":
    logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--type",
        type=str,
        default="albert-base-v2",
        help="teacher type",
    )
    parser.add_argument(
        "-n", "--name", type=str, help="glue dataset name", default="auto"
    )

    parser.add_argument("--device", type=str, default=None, help="training device")
    parser.add_argument("--seed", type=int, default=42, help="training seed")

    parser.add_argument("--pondering", action="store_true")
    parser.add_argument("--ponder-threshold", type=float, default=0.5)
    parser.add_argument("--use-prev-hiddens", action="store_true")
    parser.add_argument("--prev-state-influence", type=str, default="cat")
    parser.add_argument("--num-lambda-layers", default=3, type=int)
    parser.add_argument("--pos-encoding-type", default="none", type=str)
    parser.add_argument("--lambda-layer-arch", default="linear_cat", type=str)
    parser.add_argument("--exit-criteria", default="threshold", type=str)

    parser.add_argument("--pabee", action="store_true")
    parser.add_argument("--pabee-patience", type=int, default=6)

    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--num-hidden-layers", type=int, default=12)

    parser.add_argument("--save-model", action="store_true")

    parser.add_argument("--init-sinusoidal", action="store_true")
    parser.add_argument("--prefix", default="")
    args = parser.parse_args()

    if args.pos_encoding_type == "none":
        args.use_layer_pos_encoding = False
    elif args.pos_encoding_type == "random":
        args.use_layer_pos_encoding = True
        args.init_sinusoidal = False
    elif args.pos_encoding_type == "sinusoidal":
        args.use_layer_pos_encoding = True
        args.init_sinusoidal = True

    args.use_prev_hiddens = True
    args.rnn_lambda = args.lambda_layer_arch == "rnn"
    if "linear" in args.lambda_layer_arch:
        if "diff_cat" in args.lambda_layer_arch:
            args.prev_state_influence = "diff_cat"
        elif "diff" in args.lambda_layer_arch:
            args.prev_state_influence = "diff"
        elif "cat" in args.lambda_layer_arch:
            args.prev_state_influence = "cat"
        elif args.lambda_layer_arch == "linear":
            args.use_prev_hiddens = False
        else:
            raise ValueError

    if args.use_layer_pos_encoding != 0:
        args.use_layer_pos_encoding = True
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.prefix == "":
        if args.pabee:
            args.prefix = "pabee"
        elif args.pondering:
            if args.exit_criteria == "sample":
                args.prefix = "pondernet"
            else:
                args.prefix = "ponder"
        else:
            args.prefix = "base"

    set_deterministic_mode(args.seed)
    main(args)
