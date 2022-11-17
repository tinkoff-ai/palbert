import argparse
import zipfile

tasknames = [
    "cola_test",
    "sst2_test",
    "mrpc_test",
    "qqp_test",
    "mnli_test_matched",
    "mnli_test_mismatched",
    "qnli_test",
    "rte_test",
    "stsb_test",
]
arcnames = [
    "CoLA.tsv",
    "SST-2.tsv",
    "MRPC.tsv",
    "QQP.tsv",
    "MNLI-m.tsv",
    "MNLI-mm.tsv",
    "QNLI.tsv",
    "RTE.tsv",
    "STS-B.tsv",
]
dummy_files = ["AX.tsv", "WNLI.tsv"]


def zipdir(filenames, ziph):
    # ziph is zipfile handle
    for file, arcname in zip(filenames, arcnames):
        ziph.write(file, arcname=arcname)
    for d_f in dummy_files:
        ziph.write(d_f)


def main(args):
    if args.filenames is not None:
        filenames_args = args.filenames.split(" ")
        assert len(filenames_args) == len(arcnames)
        zip_name = f"{args.method}.zip"
        zipf = zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED)
        zipdir(filenames_args, zipf)
        return

    for seed in [42, 43, 44, 45, 46]:
        zip_name = f"{args.method}_{seed}.zip"
        zipf = zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED)
        filenames = [f"{args.method}_{t}_{seed}.tsv" for t in tasknames]
        zipdir(filenames, zipf)
        zipf.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--filenames", type=str, default=None)
    args = parser.parse_args()
    main(args)
