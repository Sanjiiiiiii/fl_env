"""クライアント。

Client     : 共有パラメータを受け取り、ローカル学習し、更新を返す (FedPer 相当)
GPFLClient : GPFL の学習手順。PFLlib の clientgpfl.py に合わせてある。

悪意クライアントのデータ汚染 (label_flip) はここで行う。
更新汚染 (sign_flip など) はサーバ側でまとめて行う (attacks.poison_updates)。
"""

import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from attacks import poison_labels


class Client:
    def __init__(
        self,
        client_id,
        model,
        train_dataset,
        test_dataset,
        label_dist,
        batch_size=32,
        lr=1e-3,
        local_epochs=1,
        device="cpu",
        is_malicious=False,
        attack="none",
        **kwargs,
    ):
        self.id = client_id
        self.device = device
        self.local_epochs = local_epochs
        self.is_malicious = is_malicious
        self.attack = attack
        self.label_dist = label_dist  # 真のラベル比率 (解析用)

        self.model = copy.deepcopy(model).to(device)
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )
        self.test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
        self.train_size = len(train_dataset)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._build_optimizer(lr)

    def _build_optimizer(self, lr):
        return torch.optim.Adam(self.model.parameters(), lr=lr)

    def set_shared(self, params):
        self.model.set_shared({k: v.to(self.device) for k, v in params.items()})

    def get_shared(self):
        return self.model.get_shared()

    # --- 学習 ---
    def compute_loss(self, x, y):
        return self.criterion(self.model(x), y)

    def train(self):
        self.model.train()
        total, nb = 0.0, 0
        num_classes = self.model.num_classes
        for _ in range(self.local_epochs):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                if self.is_malicious:
                    y = poison_labels(y, num_classes, self.attack)
                self.optimizer.zero_grad()
                loss = self.compute_loss(x, y)
                loss.backward()
                self.optimizer.step()
                total += loss.item()
                nb += 1
        return total / max(nb, 1)

    # --- 推論 ---
    def logits(self, x):
        return self.model(x)

    @torch.no_grad()
    def predict(self, loader=None):
        self.model.eval()
        loader = loader or self.test_loader
        preds, trues = [], []
        for x, y in loader:
            x = x.to(self.device)
            preds.append(self.logits(x).argmax(dim=1).cpu())
            trues.append(y)
        if not preds:
            return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
        return torch.cat(preds), torch.cat(trues)

    def evaluate(self, loader=None):
        preds, trues = self.predict(loader)
        if len(trues) == 0:
            return 0.0
        return (preds == trues).float().mean().item()


class GPFLClient(Client):
    """GPFL のクライアント。

    PFLlib との違い (意図的な変更):
      - PFLlib は base+head を SGD、CoV と GCE を別々の SGD で回す。
        ここでは 1 つの Adam のパラメータ群を 2 つに分け、
        CoV と GCE の群にだけ weight_decay=mu を掛ける。
        phase3 の他手法が Adam なので、最適化器の違いが比較に混ざらないようにした。
      - lamda の既定値を 1.0 ではなく 0.01 にしてある。
        norm 項はバッチ全体の Frobenius ノルムなのでバッチサイズに比例して大きくなり、
        lamda=1.0 では損失がこの項に支配される (実測で損失 ≈ 99)。
    """

    def __init__(self, *args, lamda=0.01, mu=0.0, **kwargs):
        self.lamda = lamda
        self.mu = mu
        super().__init__(*args, **kwargs)

        d = self.model.feature_dim
        self.generic_cond = torch.zeros(d, device=self.device)
        self.personal_cond = torch.zeros(d, device=self.device)

        # 自分のラベル比率。PFLlib の sample_per_class に相当。
        # 注意: これはクライアントの自己申告にあたる量で、
        #       FedKPer のスコアや BOBA の推定ラベル比率と突き合わせる対象になる。
        self.sample_per_class = torch.tensor(
            self.label_dist, dtype=torch.float32, device=self.device
        )
        self.gce_frozen = copy.deepcopy(self.model.gce).to(self.device)

    def _build_optimizer(self, lr):
        main, sub = [], []
        for name, p in self.model.named_parameters():
            (sub if name.startswith(("cov.", "gce.")) else main).append(p)
        return torch.optim.Adam(
            [{"params": main, "weight_decay": 0.0},
             {"params": sub, "weight_decay": self.mu}],
            lr=lr,
        )

    # --- 条件入力の更新 ---
    @torch.no_grad()
    def _refresh_conditional_inputs(self):
        """PFLlib の set_GCE と同じく、上書き前のローカル GCE から条件入力を作る。"""
        emb = self.model.gce.all_embeddings(self.device)  # (C, D)
        self.generic_cond = emb.mean(dim=0).detach().clone()
        self.personal_cond = (
            emb * self.sample_per_class.unsqueeze(1)
        ).sum(dim=0).detach().clone()

    def set_shared(self, params):
        self._refresh_conditional_inputs()   # 上書き前
        super().set_shared(params)
        self.gce_frozen = copy.deepcopy(self.model.gce).to(self.device)

    # --- 学習・推論 ---
    def compute_loss(self, x, y):
        feat = self.model.base(x)

        feat_P = self.model.cov(feat, self.personal_cond)
        output = self.model.head(feat_P)

        feat_G = self.model.cov(feat, self.generic_cond)

        loss = self.criterion(output, y)
        loss = loss + self.model.gce(feat_G, y)

        with torch.no_grad():
            emb = self.gce_frozen.embedding(y).detach()
        loss = loss + torch.norm(feat_G - emb, 2) * self.lamda
        return loss

    def logits(self, x):
        return self.model(x, self.personal_cond)


def build_client(algo, **kwargs):
    if algo in ("fedper", "splitcnn"):
        kwargs.pop("lamda", None)
        kwargs.pop("mu", None)
        return Client(**kwargs)
    if algo == "gpfl":
        return GPFLClient(**kwargs)
    raise ValueError(f"unknown algo: {algo}")
