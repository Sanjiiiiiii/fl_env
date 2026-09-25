"""Phase-1 executable: MedMNIST + Dirichlet Non-IID + FedAvg."""

from __future__ import annotations

import argparse
import csv
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from deft_fl.client import FederatedClient
from deft_fl.model import SimpleCNN
from deft_fl.server import FedAvgServer


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str = "bloodmnist"
    data_dir: str = "data"
    output_dir: str = "outputs/phase1"
    num_clients: int = 50
    participation_rate: float = 0.24
    rounds: int = 5
    local_epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 0.01
    weight_decay: float = 1e-4
    dirichlet_alpha: float = 0.1
    seed: int = 42
    num_workers: int = 0
    device: str = "auto"


def parse_args() -> ExperimentConfig:
    defaults = ExperimentConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    for field, value in asdict(defaults).items():
        arg = "--" + field.replace("_", "-")
        parser.add_argument(arg, default=value, type=type(value))
    args = vars(parser.parse_args())
    return ExperimentConfig(**args)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def labels_of(dataset: Dataset) -> np.ndarray:
    """Extract MedMNIST labels without applying image transformations."""
    labels = getattr(dataset, "labels", None)
    if labels is None:
        raise TypeError("Dataset must expose a MedMNIST-style 'labels' attribute")
    return np.asarray(labels).reshape(-1)


def dirichlet_partition(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int,
    min_samples: int = 1,
    max_attempts: int = 100,
) -> list[np.ndarray]:
    """Partition indices by class using a reproducible Dirichlet allocation.

    Every source index is assigned exactly once.  Re-sampling avoids empty
    client shards, which otherwise make a local DataLoader invalid.
    """
    if num_clients <= 0 or alpha <= 0:
        raise ValueError("num_clients and alpha must both be positive")
    if len(labels) < num_clients * min_samples:
        raise ValueError("Not enough examples to satisfy min_samples for every client")

    generator = np.random.default_rng(seed)
    for _ in range(max_attempts):
        shards: list[list[int]] = [[] for _ in range(num_clients)]
        for label in np.unique(labels):
            class_indices = np.flatnonzero(labels == label).copy()
            generator.shuffle(class_indices)
            proportions = generator.dirichlet(np.full(num_clients, alpha))
            boundaries = (np.cumsum(proportions)[:-1] * len(class_indices)).astype(int)
            for client_id, split in enumerate(np.split(class_indices, boundaries)):
                shards[client_id].extend(split.tolist())
        if min(map(len, shards)) >= min_samples:
            return [np.asarray(sorted(shard), dtype=np.int64) for shard in shards]
    raise RuntimeError(
        f"Could not produce at least {min_samples} examples per client after {max_attempts} attempts; "
        "increase alpha or reduce num_clients."
    )

def load_medmnist(config: ExperimentConfig) -> tuple[Dataset, Dataset, int, int]:
    try:
        from medmnist import INFO
        import medmnist
        from torchvision import transforms
    except ImportError as error:
        raise RuntimeError(
            "Missing dependencies. Install them with: pip install -r requirements.txt"
        ) from error

    if config.dataset not in INFO:
        raise ValueError(f"Unknown MedMNIST dataset: {config.dataset}")

    info = INFO[config.dataset]
    dataset_class = getattr(medmnist, info["python_class"])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5] * info["n_channels"],
            std=[0.5] * info["n_channels"],
        ),
    ])

    root = Path(config.data_dir)
    root.mkdir(parents=True, exist_ok=True)

    train_set = dataset_class(
        split="train", root=root, download=True, transform=transform
    )
    test_set = dataset_class(
        split="test", root=root, download=True, transform=transform
    )
    return train_set, test_set, info["n_channels"], len(info["label"])


def make_clients(
    train_set: Dataset, test_set: Dataset, config: ExperimentConfig, device: torch.device
) -> list[FederatedClient]:
    train_shards = dirichlet_partition(
        labels_of(train_set), config.num_clients, config.dirichlet_alpha, config.seed
    )
    test_shards = dirichlet_partition(
        labels_of(test_set), config.num_clients, config.dirichlet_alpha, config.seed + 1
    )
    clients = []
    for client_id, (train_indices, test_indices) in enumerate(zip(train_shards, test_shards, strict=True)):
        train_loader = DataLoader(Subset(train_set, train_indices.tolist()), batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
        test_loader = DataLoader(Subset(test_set, test_indices.tolist()), batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
        clients.append(FederatedClient(client_id, train_loader, test_loader, device))
    return clients


def select_clients(clients: list[FederatedClient], rate: float, rng: random.Random) -> list[FederatedClient]:
    if not 0 < rate <= 1:
        raise ValueError("participation_rate must be in (0, 1]")
    count = max(1, round(len(clients) * rate))
    return rng.sample(clients, count)


def run(config: ExperimentConfig) -> None:
    seed_everything(config.seed)
    device = torch.device("cuda" if config.device == "auto" and torch.cuda.is_available() else ("cpu" if config.device == "auto" else config.device))
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Config: %s", asdict(config))
    logging.info("Device: %s", device)

    train_set, test_set, in_channels, num_classes = load_medmnist(config)
    clients = make_clients(train_set, test_set, config, device)
    server = FedAvgServer(SimpleCNN(in_channels, num_classes), device)
    central_test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    model_kwargs = {"in_channels": in_channels, "num_classes": num_classes}
    rng = random.Random(config.seed)

    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["round", "participants", "mean_local_loss", "global_loss", "global_accuracy", "mean_local_accuracy"])
        writer.writeheader()
        for round_number in range(1, config.rounds + 1):
            selected = select_clients(clients, config.participation_rate, rng)
            global_state = server.broadcast()
            updates = []
            local_losses = []
            for client in selected:
                state, loss = client.train(global_state, SimpleCNN, model_kwargs, config.local_epochs, config.learning_rate, config.weight_decay)
                updates.append((state, client.num_train_samples))
                local_losses.append(loss)
            server.aggregate(updates)
            new_global_state = server.broadcast()
            local_accuracies = [client.evaluate(new_global_state, SimpleCNN, model_kwargs) for client in selected]
            global_loss, global_accuracy = server.evaluate(central_test_loader)
            row = {"round": round_number, "participants": len(selected), "mean_local_loss": float(np.mean(local_losses)), "global_loss": global_loss, "global_accuracy": global_accuracy, "mean_local_accuracy": float(np.mean(local_accuracies))}
            writer.writerow(row)
            file.flush()
            logging.info("Round %(round)d | participants=%(participants)d | local loss=%(mean_local_loss).4f | global loss=%(global_loss).4f | global accuracy=%(global_accuracy).4f | mean local accuracy=%(mean_local_accuracy).4f", row)
    torch.save(server.model.state_dict(), output_dir / "global_model.pt")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run(parse_args())
