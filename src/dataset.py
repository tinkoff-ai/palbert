import datasets
from torch.utils.data import DataLoader

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
}


def preprocess_dataset(dataset, tokenizer, task, seed: int = 42):
    def get_input_ids(examples, task):
        if task_to_keys[task][1] is None:
            return (examples[task_to_keys[task][0]],)
        return examples[task_to_keys[task][0]], examples[task_to_keys[task][1]]

    encoded_dataset = dataset.map(
        lambda examples: tokenizer(
            *get_input_ids(examples, task),
            max_length=128,
            truncation=True,
            padding="max_length",
        ),
        batched=True,
    )
    encoded_dataset = encoded_dataset.map(lambda x: {"labels": x["label"]})
    try:
        encoded_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels", "token_type_ids"],
        )
    except ValueError:
        encoded_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels"]
        )

    return encoded_dataset


def get_test_dataloaders(dataset_name, tokenizer):
    ds = datasets.load_dataset("glue", dataset_name, cache_dir="app/logs")
    loaders = {}
    for dataset_key, dataset in ds.items():
        if "test" not in dataset_key:
            continue
        dataset = preprocess_dataset(dataset, tokenizer, dataset_name)
        try:
            dataset.set_format(
                type="torch",
                columns=["input_ids", "attention_mask", "labels", "token_type_ids"],
            )
        except ValueError:
            dataset.set_format(
                type="torch", columns=["input_ids", "attention_mask", "labels"]
            )
        loaders[dataset_key] = DataLoader(dataset, batch_size=1, shuffle=False)
    return loaders


def cross_val_split(dataset_name, tokenizer, batch_size, n_splits: int = 5):
    if n_splits == -1:
        if dataset_name == "mnli":
            ds = datasets.load_dataset("glue", dataset_name, cache_dir="app/logs")
            loaders = {}
            for dataset_key, dataset in ds.items():
                if "test" in dataset_key:
                    continue
                dataset = preprocess_dataset(dataset, tokenizer, "mnli")
                try:
                    dataset.set_format(
                        type="torch",
                        columns=[
                            "input_ids",
                            "attention_mask",
                            "labels",
                            "token_type_ids",
                        ],
                    )
                except:
                    dataset.set_format(
                        type="torch", columns=["input_ids", "attention_mask", "labels"]
                    )
                shuffle = "train" in dataset_key
                c_batch_size = batch_size if "train" in dataset_key else 1
                loaders[dataset_key] = DataLoader(
                    dataset, batch_size=c_batch_size, shuffle=shuffle
                )
            return [loaders]

        trains_ds = [
            datasets.load_dataset(
                "glue", dataset_name, split="train", cache_dir="app/logs"
            )
        ]
        vals_ds = [
            datasets.load_dataset(
                "glue", dataset_name, split="validation", cache_dir="app/logs"
            )
        ]
    else:
        num_samples = 100 // n_splits
        vals_ds = datasets.load_dataset(
            "glue",
            dataset_name,
            cache_dir="app/logs",
            split=[
                f"train[{k}%:{k + num_samples}%]" for k in range(0, 100, num_samples)
            ],
        )
        trains_ds = datasets.load_dataset(
            "glue",
            dataset_name,
            cache_dir="app/logs",
            split=[
                f"train[:{k}%]+train[{k + num_samples}%:]"
                for k in range(0, 100, num_samples)
            ],
        )
    dataloaders = []
    for current_train_set, current_val_set in zip(trains_ds, vals_ds):
        train_encoded = preprocess_dataset(
            dataset=current_train_set, tokenizer=tokenizer, task=dataset_name
        )
        try:
            train_encoded.set_format(
                type="torch",
                columns=["input_ids", "attention_mask", "labels", "token_type_ids"],
            )
        except:
            train_encoded.set_format(
                type="torch", columns=["input_ids", "attention_mask", "labels"]
            )
        val_encoded = preprocess_dataset(
            dataset=current_val_set, tokenizer=tokenizer, task=dataset_name
        )
        try:
            val_encoded.set_format(
                type="torch",
                columns=["input_ids", "attention_mask", "labels", "token_type_ids"],
            )
        except:
            val_encoded.set_format(
                type="torch", columns=["input_ids", "attention_mask", "labels"]
            )
        train_dataloader = DataLoader(
            train_encoded, batch_size=batch_size, shuffle=True
        )
        valid_dataloader = DataLoader(val_encoded, batch_size=1)
        dataloaders.append({"train": train_dataloader, "valid": valid_dataloader})
    return dataloaders
