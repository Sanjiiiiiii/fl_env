"""評価指標。

  mean_acc   : 正常クライアントのローカル精度の単純平均
  worst_acc  : 正常クライアントの最低精度。ただしテスト件数が
               worst_min_test 未満のクライアントは除く。
               テスト 2 件のクライアントの精度は 0/0.5/1 しか取らず、
               最低値がそのノイズに支配されてしまうため。
  pooled_acc : 全正常クライアントの予測をまとめたサンプル加重精度
  recall     : まとめた予測でのクラス別再現率
  min_recall : その最小値 (どれか 1 クラスが捨てられていないか)
"""

import numpy as np
import torch


def evaluate_clients(clients, benign_ids, num_classes, worst_min_test=5):
    records, preds, trues = [], [], []
    for i in benign_ids:
        p, t = clients[i].predict()
        if len(t) == 0:
            continue
        acc = (p == t).float().mean().item()
        records.append({"id": i, "acc": acc, "n": int(len(t))})
        preds.append(p)
        trues.append(t)

    if not preds:
        return {
            "mean_acc": 0.0, "worst_acc": 0.0, "pooled_acc": 0.0,
            "min_recall": 0.0, "recall": [0.0] * num_classes,
            "worst_id": -1, "n_small": 0,
        }

    p = torch.cat(preds).numpy()
    t = torch.cat(trues).numpy()

    recall = []
    for c in range(num_classes):
        mask = t == c
        recall.append(float((p[mask] == c).mean()) if mask.sum() > 0 else float("nan"))
    valid = [r for r in recall if not np.isnan(r)]

    big = [r for r in records if r["n"] >= worst_min_test] or records
    worst = min(big, key=lambda r: r["acc"])

    return {
        "mean_acc": float(np.mean([r["acc"] for r in records])),
        "worst_acc": float(worst["acc"]),
        "worst_id": int(worst["id"]),
        "n_small": int(len(records) - len(big)),
        "pooled_acc": float((p == t).mean()),
        "min_recall": float(np.min(valid)) if valid else 0.0,
        "recall": recall,
    }
