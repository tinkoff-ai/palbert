from datasets import load_dataset

ax_test = load_dataset("glue", "ax", split="test", cache_dir="app/logs")
ax_label = "neutral"
wnli_test = load_dataset("glue", "wnli", split="test", cache_dir="app/logs")
wnli_label = "entailment"

with open("AX.tsv", "w") as writer:
    for idx, _ in enumerate(ax_test):
        writer.write(f"{idx}\t{ax_label}\n")

with open("WNLI.tsv", "w") as writer:
    for idx, _ in enumerate(ax_test):
        writer.write(f"{idx}\t{wnli_label}\n")
