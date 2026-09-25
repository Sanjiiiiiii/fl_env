"""A simulated federated-learning client for the phase-1 FedAvg baseline."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class FederatedClient:
    """Keeps one local data shard and trains a copy of the server model."""

    def __init__(
        self,
        client_id: int,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
    ) -> None:
        self.client_id = client_id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.num_train_samples = len(train_loader.dataset)

    def train(
        self,
        global_state: Mapping[str, torch.Tensor],
        model_factory: type[nn.Module],
        model_kwargs: dict[str, int],
        local_epochs: int,
        learning_rate: float,
        weight_decay: float,
    ) -> tuple[dict[str, torch.Tensor], float]:
        """Train from the received global state and return local parameters."""
        model = model_factory(**model_kwargs).to(self.device)
        model.load_state_dict(global_state)
        model.train()
        optimizer: Optimizer = torch.optim.SGD(
            model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay
        )
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        total_examples = 0

        for _ in range(local_epochs):
            for inputs, targets in self.train_loader:
                inputs = inputs.to(self.device)
                targets = targets.reshape(-1).long().to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(inputs), targets)
                loss.backward()
                optimizer.step()
                batch_size = targets.size(0)
                total_loss += loss.item() * batch_size
                total_examples += batch_size

        mean_loss = total_loss / max(total_examples, 1)
        state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        return state, mean_loss

    @torch.inference_mode()
    def evaluate(
        self,
        global_state: Mapping[str, torch.Tensor],
        model_factory: type[nn.Module],
        model_kwargs: dict[str, int],
    ) -> float:
        """Evaluate the received global model on this client's held-out shard."""
        model = model_factory(**model_kwargs).to(self.device)
        model.load_state_dict(global_state)
        model.eval()
        correct = 0
        total = 0
        for inputs, targets in self.test_loader:
            logits = model(inputs.to(self.device))
            targets = targets.reshape(-1).long().to(self.device)
            correct += (logits.argmax(dim=1) == targets).sum().item()
            total += targets.size(0)
        return correct / max(total, 1)