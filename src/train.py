import argparse
import logging
import os.path
from collections import defaultdict
from itertools import chain

import numpy as np
import torch
import transformers as t
import wandb
from datasets import load_metric
from torch.optim import Adam
from transformers import AutoConfig, AutoModelForSequenceClassification

from dataset import cross_val_split, get_test_dataloaders
from modeling.palbert_fast import (AlbertPABEEForSequenceClassification,
                                   PAlbertForSequenceClassification)
from trainer import Trainer
from utils.set_deterministic import set_deterministic_mode

glue_datasets = [
    "sst2",
    "mnli",
    "cola",
    "stsb",
    "mrpc",
    "qqp",
    "qnli",
    "rte",
    "wnli",
    "hans",
]

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
    wandb.init(project="palbert", config=vars(args))
    if not torch.cuda.is_available():
        print("Warning! Cuda is not available.")
    device = torch.device(args.device)

    print(vars(args))

    tokenizer = t.AutoTokenizer.from_pretrained(args.type)
    if args.validate or not args.cross_val:
        n_splits = -1
    else:
        n_splits = 5
    dataloaders = cross_val_split(
        args.name, tokenizer=tokenizer, batch_size=args.batch_size, n_splits=n_splits
    )
    num_labels = 1 if args.name == "stsb" else (3 if args.name == "mnli" else 2)

    metric_fn = load_metric("glue", args.name, keep_in_memory=True)
    assert metric_fn is not None
    config = AutoConfig.from_pretrained(
        args.type,
        num_labels=num_labels,
        hidden_dropout_prob=args.hidden_dropout_prob,
        attention_probs_dropout_prob=args.attention_probs_dropout_prob,
        classifier_dropout_prob=args.classifier_dropout,
        output_hidden_states=False,
        num_hidden_layers=args.num_hidden_layers,
    )

    print("TRAINING\n\n\n\n")
    split_metrics = defaultdict(lambda: [])
    if (
        args.validate
    ):  # legacy, we do not use this when running experiments in our paper
        dataloaders = [dataloaders[0] for _ in range(5)]
    for split_idx, c_dataloaders in enumerate(dataloaders):  # runs single time
        if args.validate:
            set_deterministic_mode(
                split_idx + args.seed
            )  # setting seed for current run
        if args.pondering:
            if "albert" in args.type:
                model = PAlbertForSequenceClassification.from_pretrained(
                    args.type, config=config
                )
                p_model = model.albert

            p_model.set_threshold(args.ponder_threshold)
            if args.init_sinusoidal:
                p_model.albert.init_layer_pos_embeddings()
            model.classifiers.set_lambda_layer_arch(
                num_layers=args.num_lambda_layers,
                rnn=args.rnn_lambda,
                hidden_dim=model.config.hidden_size * 2
                if "cat" in args.prev_state_influence
                else model.config.hidden_size,
            )
        elif args.pabee:
            if "albert" in args.type:
                model = AlbertPABEEForSequenceClassification.from_pretrained(
                    args.type, config=config
                )
            elif "roberta" in args.type:
                model = RobertaPABEEForSequenceClassification.from_pretrained(
                    args.type, config=config
                )
            else:
                raise TypeError("model not found")
            model.set_patience(args.pabee_patience)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                args.type,
                config=config,
            )

        model.to(device)
        optimizer_config = [
            {
                "params": model.parameters(),
                "lr": args.lr,
                "weight_decay": args.weight_decay,
            }
        ]
        if args.pondering and args.lambda_lr is not None:
            lambda_params = []
            model_parameters = []
            for n, p in model.named_parameters():
                if "exit" in n:
                    lambda_params.append(p)
                else:
                    model_parameters.append(p)
            optimizer_config = [
                {
                    "params": model_parameters,
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                },
                {"params": lambda_params, "lr": args.lambda_lr},
            ]

        optimizer = Adam(optimizer_config, betas=(args.beta1, args.beta2))
        scheduler = None
        scaler = None
        if args.fp16:
            scaler = torch.cuda.amp.GradScaler()

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            metric=metric_fn,
            scaler=scaler,
            is_regression=args.name == "stsb",
        )
        if args.name == "cola":
            val_metric = "matthews_correlation"
        elif args.name == "stsb":
            val_metric = "pearson_spearman"
        elif args.name in ["mrpc", "qqp"]:
            val_metric = "accuracy_f1"
        else:
            val_metric = "accuracy"
        trainer.train(
            dataloaders=c_dataloaders,
            num_epochs=args.epochs,
            beta=args.beta,
            prior_lambda=args.lambda_p,
            debug=args.debug,
            patience=args.patience,
            val_metric=val_metric,
            pabee=args.pabee,
            pondering=args.pondering,
            detach_lambda=args.detach_lambda,
            use_layer_pos_encoding=args.use_layer_pos_encoding,
            use_prev_hiddens=args.use_prev_hiddens,
            prev_state_influence=args.prev_state_influence,
            prior_type=args.prior_type,
            exit_criteria=args.exit_criteria,
        )
        if not os.path.isdir("logs"):
            os.mkdir("logs")
        if args.save_model:
            prefix = "base"
            if args.pondering:
                prefix = "ponder"
                if args.exit_criteria == "sample":
                    prefix = "pondernet"
            if args.pabee:
                prefix = "pabee"
            dataset = args.name
            checkpoint_filename = f"logs/{prefix}_{dataset}_{args.seed}_{args.lambda_p}"
            trainer.save_model(checkpoint_filename)
        for k, v in trainer.best_metric_dict.items():
            split_metrics[k].append(v)
        if args.run_test:
            print("RUNNING TEST SET")
            model.load_state_dict(trainer.best_sd)
            dataloaders = get_test_dataloaders(args.name, tokenizer)
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
                prefix = "base"
                if args.pondering:
                    prefix = "ponder"
                    if args.exit_criteria == "sample":
                        prefix = "pondernet"
                if args.pabee:
                    prefix = "pabee"
                dataset = args.name
                with open(
                    f"logs/{prefix}_{dataset}_{result_key}_{args.seed}_{args.lambda_p}.tsv",
                    "w",
                ) as writer:
                    for index, item in enumerate(result):
                        if args.name == "stsb":
                            item = max(0, min(item, 5))
                            writer.write("%d\t%3.3f\n" % (index, item))
                        else:
                            item = task_to_labels[args.name][item]
                            writer.write("%d\t%s\n" % (index, item))
                print("SAVING_TO_WANDB")
                wandb.save(
                    f"logs/{prefix}_{dataset}_{result_key}_{args.seed}_{args.lambda_p}.tsv"
                )
        if not (args.cross_val or args.validate):
            break

    mean_metrics = {"mean_" + k: np.mean(v) for k, v in split_metrics.items()}
    std_metrics = {"std_" + k: np.std(v) for k, v in split_metrics.items()}
    final_metrics = {k: v for k, v in chain(mean_metrics.items(), std_metrics.items())}
    wandb.log(final_metrics)


if __name__ == "__main__":
    logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--type",
        type=str,
        default="albert-base-v2",
        help="teacher type",
    )
    parser.add_argument("-n", "--name", type=str, help="glue dataset name")

    parser.add_argument("--epochs", type=int, default=25, help="max number of epochs")
    parser.add_argument(
        "--patience", type=int, default=5, help="maximum num steps without improvement"
    )
    parser.add_argument("--valid-metric", type=str, default="accuracy")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--beta2", type=float, default=0.999, help="Adam beta2")
    parser.add_argument("--batch-size", type=int, default=8, help="batch size")
    parser.add_argument("--num-warmup-steps", type=float, default=0)

    parser.add_argument("--hidden-dropout-prob", type=float, default=0.0)
    parser.add_argument("--attention-probs-dropout-prob", type=float, default=0.0)
    parser.add_argument("--classifier_dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42, help="training seed")
    parser.add_argument("--device", type=str, default=None, help="training device")

    parser.add_argument("--log", type=str, help="path to log file", default="./logs")

    parser.add_argument("--pondering", action="store_true")
    parser.add_argument("--beta", type=float, default=0.0, help="KL divergence weight")
    parser.add_argument(
        "--lambda-p", type=float, default=0.1, help="prior distribution parameter"
    )
    parser.add_argument(
        "--ponder-threshold", type=float, default=0.5, help="threshold for Q-Exit"
    )
    parser.add_argument(
        "--lambda-lr", type=float, default=None, help="learning rate for lambda layers"
    )
    parser.add_argument(
        "--prior-type",
        type=str,
        default="geometric",
        help="type of the prior distribution. In our paper we use geometric",
    )
    parser.add_argument(
        "--use-prev-hiddens",
        action="store_true",
        help="flag to use previous hidden state from the previous layer",
    )
    parser.add_argument(
        "--prev-state-influence",
        type=str,
        default="cat",
        help="we can use difference or concatenation of the previous states",
    )
    parser.add_argument(
        "--num-lambda-layers", default=3, type=int, help="number of lambda layers"
    )
    parser.add_argument(
        "--pos-encoding-type",
        default="none",
        type=str,
        help="we can use positional encoding for every layer",
    )
    parser.add_argument(
        "--lambda-layer-arch",
        default="linear_diff",
        type=str,
        help="linear_cat, linear_diff or rnn",
    )
    parser.add_argument(
        "--exit-criteria",
        default="threshold",
        type=str,
        help="exit-criteria=sample leads to PonderNet's exit criteria",
    )

    parser.add_argument("--pabee", action="store_true", help="run PABEE experiment")
    parser.add_argument(
        "--pabee-patience",
        type=int,
        default=6,
        help="patience parameter for the PABEE exit criteria",
    )

    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--num-hidden-layers", type=int, default=12)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--validate", action="store_true")

    parser.add_argument("--save-model", action="store_true")
    parser.add_argument(
        "--cross-val", action="store_true", help="we did not use this flag. Legacy"
    )

    parser.add_argument("--init-sinusoidal", action="store_true")
    parser.add_argument("--detach-lambda", action="store_true")

    parser.add_argument("--run-test", action="store_true")

    args = parser.parse_args()
    if args.run_test:
        print("Will run test set after training")
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
    set_deterministic_mode(args.seed)
    main(args)
