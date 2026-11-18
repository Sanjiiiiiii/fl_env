"""モデル定義。

SplitCNN : 共有部 (base+neck) とローカル部 (head) に分けただけの CNN。
           手法としては FedPer 相当。GPFL ではない。
GPFLNet  : GPFL (Zhang et al., ICCV 2023) 本体。
           base / CoV / GCE を共有し、head だけをローカルに残す。
           PFLlib の servergpfl.py・clientgpfl.py の実装に合わせてある。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_trunk(in_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
    )


def _trunk_out_dim(trunk, in_channels):
    with torch.no_grad():
        return trunk(torch.zeros(1, in_channels, 28, 28)).shape[1]


# =====================================================================
# FedPer 相当
# =====================================================================
class SplitCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=8, feature_dim=256, share_neck=True):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.share_neck = share_neck

        self.base = _conv_trunk(in_channels)
        feat = _trunk_out_dim(self.base, in_channels)
        self.neck = nn.Sequential(nn.Linear(feat, feature_dim), nn.ReLU())
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.head(self.neck(self.base(x)))

    def shared_keys(self):
        prefixes = ("base.", "neck.") if self.share_neck else ("base.",)
        return [k for k in self.state_dict() if k.startswith(prefixes)]

    def get_shared(self):
        sd = self.state_dict()
        return {k: sd[k].detach().clone() for k in self.shared_keys()}

    def set_shared(self, params):
        sd = self.state_dict()
        for k, v in params.items():
            sd[k] = v.to(sd[k].device, sd[k].dtype)
        self.load_state_dict(sd)


# =====================================================================
# GPFL の部品 (PFLlib servergpfl.py と同じ構造)
# =====================================================================
class GCE(nn.Module):
    """Global Category Embedding.

    クラスごとの「お手本ベクトル」を持ち、特徴量との cosine 類似度に対する
    softmax 損失を返す。埋め込み自体が学習対象で、サーバで平均される。
    """

    def __init__(self, in_features, num_classes):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.embedding = nn.Embedding(num_classes, in_features)

    def all_embeddings(self, device):
        idx = torch.arange(self.num_classes, device=device)
        return self.embedding(idx)

    def forward(self, x, label):
        emb = self.all_embeddings(x.device)
        cosine = F.linear(F.normalize(x), F.normalize(emb))
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        log_p = F.log_softmax(cosine, dim=1)
        return -torch.mean(torch.sum(one_hot * log_p, dim=1))


class CoV(nn.Module):
    """Conditional Valve.

    条件ベクトル context から (gamma, beta) を作り、特徴量を
    x * (gamma + 1) + beta と変換する (FiLM 的なアフィン変換)。
    同じ特徴量でも、渡す条件を変えることで「全体向け」と「自分向け」の
    2 本の特徴量に分けられる。重み自体は普通のパラメータで、サーバで平均される。
    """

    def __init__(self, in_dim):
        super().__init__()
        self.Conditional_gamma = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(), nn.LayerNorm([in_dim])
        )
        self.Conditional_beta = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(), nn.LayerNorm([in_dim])
        )
        self.act = nn.ReLU()

    def forward(self, x, context):
        gamma = self.Conditional_gamma(context)
        beta = self.Conditional_beta(context)
        return self.act(x * (gamma + 1) + beta)


class GPFLNet(nn.Module):
    """base + CoV + GCE を共有し、head をローカルに残す。"""

    def __init__(self, in_channels=3, num_classes=8, feature_dim=256):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim

        trunk = _conv_trunk(in_channels)
        feat = _trunk_out_dim(trunk, in_channels)
        # PFLlib では model.base の出力が head の入力次元と一致する。
        # ここでも base の末尾に射影を入れて feature_dim にそろえる。
        self.base = nn.Sequential(trunk, nn.Linear(feat, feature_dim), nn.ReLU())
        self.cov = CoV(feature_dim)
        self.gce = GCE(feature_dim, num_classes)
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, x, context):
        return self.head(self.cov(self.base(x), context))

    def shared_keys(self):
        return [
            k for k in self.state_dict()
            if k.startswith(("base.", "cov.", "gce."))
        ]

    def get_shared(self):
        sd = self.state_dict()
        return {k: sd[k].detach().clone() for k in self.shared_keys()}

    def set_shared(self, params):
        sd = self.state_dict()
        for k, v in params.items():
            sd[k] = v.to(sd[k].device, sd[k].dtype)
        self.load_state_dict(sd)


def build_model(algo="fedper", num_classes=8, **kwargs):
    if algo in ("fedper", "splitcnn"):
        return SplitCNN(num_classes=num_classes, **kwargs)
    if algo == "gpfl":
        return GPFLNet(num_classes=num_classes, **kwargs)
    raise ValueError(f"unknown algo: {algo}")
