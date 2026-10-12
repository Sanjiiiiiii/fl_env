import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class Client:
    def __init__(self, client_id, model, train_dataset, test_dataset,
                 batch_size=32, lr=0.001, local_epochs=1, device='cpu'):
        self.id           = client_id
        self.device       = device
        self.local_epochs = local_epochs

        self.model = copy.deepcopy(model).to(device)
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0, pin_memory=False
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr
        )

    def set_parameters(self, global_params):
        """サーバからグローバルモデルの全パラメータを受け取る"""
        self.model.load_state_dict(
            {k: v.to(self.device) for k, v in global_params.items()}
        )

    def train(self):
        """ローカル学習：純粋な正常学習のみ"""
        self.model.train()
        for _ in range(self.local_epochs):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()
                loss = self.criterion(self.model(x), y)
                loss.backward()
                self.optimizer.step()

    def get_parameters(self):
        """集約のためモデル全体のパラメータを返す"""
        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

    def evaluate(self):
        """テスト精度を返す"""
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                preds = self.model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total   += y.size(0)
        return correct / total if total > 0 else 0.0