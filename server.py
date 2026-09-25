"""FedAvg server coordination for phase 1."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch
from torch import nn
from torch.utils.data import DataLoader


class FedAvgServer:
    """Maintains the global model and aggregates client updates by data volume."""

    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.model = model.to(device)
        self.device = device

    def broadcast(self) -> dict[str, torch.Tensor]:
        """Return a CPU snapshot so clients cannot mutate server state."""
        return {name: tensor.detach().cpu().clone() for name, tensor in self.model.state_dict().items()}

    def aggregate(
        self, updates: Iterable[tuple[Mapping[str, torch.Tensor], int]]
    ) -> None:
        """Apply sample-count-weighted FedAvg to client model states."""
        updates = list(updates)
        if not updates:
            raise ValueError("FedAvg requires at least one client update")
        total_samples = sum(sample_count for _, sample_count in updates)
        if total_samples <= 0:
            raise ValueError("Client sample counts must be positive")

        aggregated: dict[str, torch.Tensor] = {}
        for name, reference in self.model.state_dict().items():
            values = [state[name] for state, _ in updates]
            if torch.is_floating_point(reference):
                weighted = sum(
                    value.to(torch.float64) * (count / total_samples)
                    for value, (_, count) in zip(values, updates, strict=True)
                )
                aggregated[name] = weighted.to(dtype=reference.dtype)
            else:
                # CNN buffers such as num_batches_tracked are not learned weights.
                aggregated[name] = values[0].clone()
        self.model.load_state_dict(aggregated)

    @torch.inference_mode()
    def evaluate(self, data_loader: DataLoader) -> tuple[float, float]:
        """Return (cross_entropy_loss, accuracy) on a central test set."""
        self.model.eval()
        criterion = nn.CrossEntropyLoss(reduction="sum")
        total_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in data_loader:
            inputs = inputs.to(self.device)
            targets = targets.reshape(-1).long().to(self.device)
            logits = self.model(inputs)
            total_loss += criterion(logits, targets).item()
            correct += (logits.argmax(dim=1) == targets).sum().item()
            total += targets.size(0)
        return total_loss / max(total, 1), correct / max(total, 1)