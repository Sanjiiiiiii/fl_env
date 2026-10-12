import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class Client:
    def __init__(self, client_id, model, train_dataset, test_dataset,
            batch_size=32, lr=0.001, local_epochs=1, device='cpu',
            is_malicious=False):
        self.id            = client_id
        self.device        = device
        self.local_epochs  = local_epochs
        self.is_malicious  = is_malicious

        self.model = copy.deepcopy(model).to(device)
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,num_workers=0,pin_memory=False
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,num_workers=0,pin_memory=False
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr
        )

    def set_base_params(self, base_params):
        """サーバからグローバルbaseパラメータを受け取る"""
        self.model.set_base_params(
            {k: v.to(self.device) for k, v in base_params.items()}
        )

    def train(self):
        """ローカル学習：base + headを両方更新"""
        self.model.train()
        for _ in range(self.local_epochs):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)

                # Label-Flip攻撃：悪意クライアントはラベルを反転
                if self.is_malicious:
                    y = (y + 1) % self.model.head[-1].out_features

                self.optimizer.zero_grad()
                loss = self.criterion(self.model(x), y)
                loss.backward()
                self.optimizer.step()

    def get_base_params(self):
        """集約のためbaseパラメータを返す"""
        return self.model.get_base_params()

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