"""データ準備: BloodMNIST の読み込み、Dirichlet 分割、サーバ用データの切り出し。

Phase2 からの変更点:
  - 各クライアントが「自分の分布に沿ったローカルテストセット」を持つ
    (個別化FLでは、全員共通のテストセットで測ると個別ヘッドが不当に低く出る)
  - BOBA 用に、クラスごとの少量のクリーンデータをサーバ側へ退避する
  - 各クライアントの真のラベル比率を記録する
    (将来 BOBA の推定ラベル比率や FedKPer の自己申告スコアと突き合わせるため)
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset

NUM_CLASSES = 8


class ArrayDataset(Dataset):
    """numpy 配列をそのまま持つ軽量データセット。"""

    def __init__(self, images, labels):
        self.images = images
        self.labels = labels.astype(np.int64)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.images[idx]).permute(2, 0, 1).float() / 255.0
        return img, int(self.labels[idx])


def load_bloodmnist(data_root):
    path = os.path.join(data_root, "bloodmnist.npz")
    data = np.load(path)
    return {
        "train_images": data["train_images"],
        "train_labels": data["train_labels"].squeeze().astype(np.int64),
        "test_images": data["test_images"],
        "test_labels": data["test_labels"].squeeze().astype(np.int64),
    }


def dirichlet_partition(labels, num_clients, alpha, num_classes, rng):
    """クラスごとに Dirichlet 比率でインデックスを配る。"""
    client_idx = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        proportions = rng.dirichlet([alpha] * num_clients)
        cuts = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
        for i, part in enumerate(np.split(idx, cuts)):
            client_idx[i].extend(part.tolist())
    return [np.array(sorted(ci), dtype=np.int64) for ci in client_idx]


def build_federated_data(
    data_root,
    num_clients,
    alpha,
    num_classes=NUM_CLASSES,
    server_per_class=20,
    local_test_ratio=0.2,
    min_local=10,
    seed=42,
):
    """連合学習用のデータ一式を組み立てる。

    返り値:
      clients:    [{train: Dataset, test: Dataset, label_dist: np.ndarray, size: int}, ...]
      server:     Dataset  (BOBA 用のクラス均等なクリーンデータ)
      global_test: Dataset (公式テストセット。参考値として全体精度を見るのに使う)
    """
    rng = np.random.default_rng(seed)
    raw = load_bloodmnist(data_root)
    images, labels = raw["train_images"], raw["train_labels"]

    # --- サーバ用データをクラス均等に抜き取る (BOBA の「純粋ジュース」用) ---
    server_idx = []
    for c in range(num_classes):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        take = min(server_per_class, len(idx))
        server_idx.extend(idx[:take].tolist())
    server_idx = np.array(sorted(server_idx), dtype=np.int64)
    remain = np.setdiff1d(np.arange(len(labels)), server_idx)

    # --- 残りを Dirichlet でクライアントへ ---
    parts = dirichlet_partition(
        labels[remain], num_clients, alpha, num_classes, rng
    )
    parts = [remain[p] for p in parts]

    clients = []
    for p in parts:
        rng.shuffle(p)
        if len(p) < min_local:
            # 極端に小さいクライアントは学習も評価も不安定なので最低限まで補う
            extra = rng.choice(remain, size=min_local - len(p), replace=False)
            p = np.concatenate([p, extra])
        n_test = max(1, int(len(p) * local_test_ratio))
        test_idx, train_idx = p[:n_test], p[n_test:]
        if len(train_idx) == 0:  # 念のため
            train_idx, test_idx = p, p[:1]
        dist = np.bincount(labels[train_idx], minlength=num_classes).astype(np.float64)
        dist = dist / max(dist.sum(), 1.0)
        clients.append(
            {
                "train": ArrayDataset(images[train_idx], labels[train_idx]),
                "test": ArrayDataset(images[test_idx], labels[test_idx]),
                "label_dist": dist,
                "size": int(len(train_idx)),
            }
        )

    server = ArrayDataset(images[server_idx], labels[server_idx])
    global_test = ArrayDataset(raw["test_images"], raw["test_labels"])
    return clients, server, global_test
