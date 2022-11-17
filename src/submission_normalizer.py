import argparse
import os

label_to_digit = {
    "qqp": {"not_duplicate": 0, "duplicate": 1},
    "sst2": {"negative": 0, "positive": 1},
    # "cola": {"acceptable": 1, "unacceptable": 0},
    # "mrpc": {"not_equivalent": 0, "equivalent": 1},
}


def main(args):
    for task, c_l_t_d in label_to_digit.items():
        for c_s in range(42, 47):
            with open(f"tmp_{args.method}_{task}_test_{c_s}.tsv", "w") as out:
                with open(f"{args.method}_{task}_test_{c_s}.tsv", "r") as inp:
                    for idx, line in enumerate(inp):
                        label = line.split()[-1]
                        out.write(f"{idx}\t{label_to_digit[task][label]}\n")
            os.replace(
                f"tmp_{args.method}_{task}_test_{c_s}.tsv",
                f"{args.method}_{task}_test_{c_s}.tsv",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    args = parser.parse_args()
    main(args)
