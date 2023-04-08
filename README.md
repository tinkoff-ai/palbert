# PALBERT

Code for the NeurIPS 2022 paper [PALBERT: Teaching ALBERT to Ponder](https://arxiv.org/abs/2204.03276).

<p align="center">
  <img width="512" height="512" src="assets/dalle.png">
</p>

- [PALBERT](#palbert)
  * [How to run](#how-to-run)
    + [Docker](#docker)
    + [Training](#training)
      - [PALBERT](#palbert-1)
      - [PonderNET with ALBERT](#pondernet-with-albert)
      - [PABEE](#pabee)
      - [ALBERT](#albert)
  * [Structure](#structure)
  * [Citation](#citation)

## How to run

### Docker
You can build and run our docker with the following commands:
```
docker build -t ponderbert .
docker run --gpus "device=0" -i -t ponderbert /bin/bash
```

### Training

Examples of the train commands:

#### PALBERT
```commandline
python3 src/train.py \
  --lr 0.00001 \
  --batch-size 32 \
  --lambda-lr 0.00001 \
  --type "albert-base-v2" \
  --name "rte" \
  --fp16 \
  --pondering \
  --beta 0.5 \
  --lambda-layer-arch "linear_cat" \
  --num-lambda-layers 3 \
  --run-test
```

#### PonderNET with ALBERT

```commandline
python3 src/train.py \
  --lr 0.00001 \
  --batch-size 16 \
  --type "albert-base-v2" \
  --name "rte" \
  --fp16 \
  --pondering \
  --beta 0.5 \
  --lambda-layer-arch "linear" \
  --num-lambda-layers 1 \
  --exit-criteria "sample" \
  --run-test
```

#### PABEE

```commandline
python3 src/train.py \
  --lr 0.00001 \
  --batch-size 32 \
  --type "albert-base-v2" \
  --name "rte" \
  --fp16 \
  --pabee
```

#### ALBERT

```commandline
python3 src/train.py \
  --lr 0.00001 \
  --batch-size 128 \
  --type "albert-base-v2" \
  --name "rte" \
  --fp16
```


## Structure

```
src
├── __init__.py
├── create_dummy_test.py  # create dummy test for AX and WNLI
├── dataset.py  # glue dataset loading
├── loss.py  # regularization and kl losses
├── modeling  # model
│   ├── __init__.py
│   └── palbert_fast.py  # albert-based models
├── sub_zipper.py  # zip submission
├── test.py  # create test set predictions
├── train.py  # training script
├── trainer.py  # training loops and main pipeline
└── utils
    ├── __init__.py
    └── set_deterministic.py  # set_seed(42)
```

## Citation

You can cite our paper with the following bibtex:
```
@inproceedings{palbert,
 author = {Balagansky, Nikita and Gavrilov, Daniil},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {S. Koyejo and S. Mohamed and A. Agarwal and D. Belgrave and K. Cho and A. Oh},
 pages = {14002--14012},
 publisher = {Curran Associates, Inc.},
 title = {PALBERT: Teaching ALBERT to Ponder},
 url = {https://proceedings.neurips.cc/paper_files/paper/2022/file/5a9c1af5f76da0bd37903b6f23e96c74-Paper-Conference.pdf},
 volume = {35},
 year = {2022}
}
```
