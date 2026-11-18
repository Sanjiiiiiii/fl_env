"""評価指標。

単純な平均精度だけでは「少数派クラスが壊れている」ことを見逃すので、
BOBA 論文が使っている考え方に合わせて次をそろえる。

  mean_acc   : 正常クライアントのローカル精度の単純平均
  worst_acc  : 正常クライアントの最低精度 (最も被害を受けた人)
  pooled_acc : 全正常クライアントの予測をまとめたサンプル加重精度
  recall     : まとめた予測でのクラス別再現率
  min_recall : その最小値 (どれか 1 クラスが捨てられていないか)
"""

import numpy as np
import torch


def evaluate_clients(clients, benign_ids, num_classes):
    accs, preds, trues = [], [], []
    for i in benign_ids:
        p, t = clients[i].predict()
        if len(t) == 0:
            continue
        accs.append((p == t).float().mean().item())
        preds.append(p)
        trues.append(t)

    if not preds:
        return {
            "mean_acc": 0.0, "worst_acc": 0.0, "pooled_acc": 0.0,
            "min_recall": 0.0, "recall": [0.0] * num_classes,
        }

    p = torch.cat(preds).numpy()
    t = torch.cat(trues).numpy()

    recall = []
    for c in range(num_classes):
        mask = t == c
        recall.append(float((p[mask] == c).mean()) if mask.sum() > 0 else float("nan"))

    valid = [r for r in recall if not np.isnan(r)]
    return {
        "mean_acc": float(np.mean(accs)),
        "worst_acc": float(np.min(accs)),
        "pooled_acc": float((p == t).mean()),
        "min_recall": float(np.min(valid)) if valid else 0.0,
        "recall": recall,
    }
