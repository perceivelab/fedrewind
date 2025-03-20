<div align="center">
  
# FedRewind: Rewinding Continual Model Exchange for Decentralized Federated Learning
  Luca Palazzo, Matteo Pennisi, Federica Proietto Salanitri, Giovanni Bellitto, Simone Palazzo, Concetto Spampinato

[![Paper](https://img.shields.io/badge/arXiv-2307.02984-b31b1b.svg)](https://arxiv.org/abs/2411.09842)
[![Conference](http://img.shields.io/badge/ICPR-2024-blue)](https://link.springer.com/chapter/10.1007/978-3-031-78389-0_6)
</div>


FedRewind is a novel method leveraging continual learning principles in decentralized federated learning, improving robustness against non-i.i.d. data by periodically rewinding models to previously visited nodes. It is based on the [PFLlib](https://github.com/TsingZ0/PFLlib) framework.

## Requirements

- Python 3.8 or higher
- PyTorch 1.10 or higher
- Other dependencies specified in `fedrewind.yaml`

## Installation

### With conda

```bash
conda create -n fedrewind
conda activate fedrewind
```

## Dataset Preparation

### Creating Data Splits

Create non-IID data splits using:

```bash
python system/main.py -data CIFAR-10 -nc 1 --dataset_generate --dataset_niid --dataset_outdir CIFAR-10-nIID-1C-DIRA0.1 --dataset_partition dir --dataset_balance --dataset_dir_alpha 0.1
```


- `-data <dataset>`: Directory containing dataset to be split.
- `--dataset_generate`: Enables dataset's splits generation.
- `--dataset_outdir <output directory>`: Destination directory for dataset's splits.
- `-nc <nodes count>`: Federation nodes count.
- `--dataset_dir_alpha <alpha>`: Lower values increase non-IIDness.
- `--dataset_partition <dir|pat>`: Uses Dirichlet distribution to partition data or .
- `--dataset_balance`: Enables balanced samples count per split.

## Training

### Decentralized Federated Learning

Example of training command:

```bash
python system/main.py \
  -data CIFAR-10-nonIID \
  -nc 10 \
  -nb 10 \
  --batch_size 32 \
  --local_epochs 10 \
  -lr 0.001 \
  --rewind_ratio 0.1 \
  --rewind_strategy atend_pre \
  --global_rounds 50 \
  -go CIFAR-10-nIID \
  -lr 0.001 \
  -lam 0.1 \
  --rewind_rotate \
  --rewind_ratio 0.1 \
  --rewind_strategy atend_pre
```
- `-data <dataset>`: dataset split be used under datast/ directory
- `-nc <nodes>`: number of federeation nodes
- `-nb <classes>`: number of classification labels
- `--batch_size <batch size`: batch size for training step
- `--local_epochs <local epochs`: number of epochs for round
- `--global_rounds <rounds>`: number of federation training rounds
- `-go <run id>`: identifier for experiment
- `-lr <learning rate>`: learning rate
- `--rewind_rotate`: enables model routing
- `--rewind_ratio <rewind ratio>`: local epochs ratio for number of rewind epoch
- `--rewind_strategy <atend_pre|atend|middleway>`: rewind epoch start at end of round minus local epoch * factor, at the end of round, middleway of round 

## Evaluation Metrics

FedRewind uses two metrics:

- **Federation Accuracy (FA)**: Aggregated accuracy across all nodes.
- **Personalized Federation Accuracy (PFA)**: Accuracy on each node's dataset.

## Citation

```bibtex
@inproceedings{palazzo2025fedrewind,
  title={FedRewind: Rewinding Continual Model Exchange for Decentralized Federated Learning},
  author={Palazzo, Luca and Pennisi, Matteo and Proietto Salanitri, Federica and Bellitto, Giovanni and Palazzo, Simone and Spampinato, Concetto},
  booktitle={International Conference on Pattern Recognition},
  pages={79--94},
  year={2025},
  organization={Springer}
}
```

## Original Framework
FedRewind extends the [PFLlib](https://github.com/TsingZ0/PFLlib) framework.

