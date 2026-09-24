from_future_import annotations

import argparse
import csv
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset,Subset

from deft_fl.client import FederatedClient
from deft_fl.server import FedAvgServer
from deft_fl.model import SimpleCNN

@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str ="bloodmnist"
    data_dir: str ="data"
    output_dir: str ="outputs/phase1"
    num_clients: int = 50
    participation_rate:float=0.24
    rounds: int = 5
    local_epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 0.01
    weight_decay: float = 1e-4
    seed: int = 42
    num_workers: int = 0
    device:str="auto"


def parse_args() -> ExperimentConfig:
    default=ExperimentConfig()
    parser=argparse.ArgumentParser(description=_doc_)
    from field,value in asdict(default).items():
        arg="--" + field.replace("_","-")
        parser.add_argument(arg,default=value,type=type(value))
    args=vars(parser.parse_args())
    return ExperimentConfig(**args)

def seed_everything(seed:int):->None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def labels_of(dataset:Dataset)->np.array:
    labels=getattr(dataset,"labels",None)
    if labels is None:
        raise TypeError("Dataset must expose a MedMNIST-Style'labels' attribute")
    return np.array(labels).reshape(-1)


def dirichlet_partition(
    labels:np.ndarray,
    num_clients:int,
    alpha:float,
    seed:int,
    mini_samples:int=1,
    max_attempts:int=100,
)->listp[np.ndarray]:
    if num_clients <= 0 or alpha <= 0:
        raise ValueError("num_clients and alpha must be positive")
    if len(labels) < num_clients * mini_samples:
        raise ValueError("Not enough samples to partition among clients")
    generator=np.random.default_rng(seed)
    for _ in range(max_attempts):]