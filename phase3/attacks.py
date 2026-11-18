"""攻撃。

2種類に分かれる。
  データ汚染型 : ローカル学習のラベルを書き換える  -> poison_labels()
  更新汚染型   : 送信する更新ベクトルを差し替える -> poison_updates()

実装済み:
  none        : 攻撃なし
  label_flip  : ラベルを 1 つずらす (データ汚染)
  sign_flip   : 更新の符号を反転 (更新汚染)
  gaussian    : 大きなランダムベクトルを送る (更新汚染)
  ipm         : 正常更新の平均に -gamma を掛けて送る (更新汚染, Xie et al. 2019)
  alie        : 正常更新の分布の内側に隠れる (更新汚染, Baruch et al. 2019)

ipm と alie は「正常クライアントの更新を攻撃者が知っている」という強い想定
(omniscient attacker) に立つ。防御の上限性能を測るための標準的な設定。
"""

import math

import torch


def poison_labels(y, num_classes, kind):
    if kind == "label_flip":
        return (y + 1) % num_classes
    return y


DATA_ATTACKS = {"label_flip"}
UPDATE_ATTACKS = {"sign_flip", "gaussian", "ipm", "alie"}


def poison_updates(X, mal_idx, kind, params=None):
    """X: (n, d) の更新行列。mal_idx の行を書き換えて返す。"""
    if kind not in UPDATE_ATTACKS or len(mal_idx) == 0:
        return X
    params = params or {}
    X = X.clone()
    mal = torch.as_tensor(mal_idx, dtype=torch.long, device=X.device)
    benign = torch.tensor(
        [i for i in range(X.shape[0]) if i not in set(mal_idx)],
        dtype=torch.long,
        device=X.device,
    )

    if kind == "sign_flip":
        scale = float(params.get("scale", 1.0))
        X[mal] = -scale * X[mal]

    elif kind == "gaussian":
        # 正常更新のノルムを基準にした大きなノイズ
        ref = X[benign].norm(dim=1).mean() if len(benign) else X.norm(dim=1).mean()
        scale = float(params.get("scale", 10.0))
        noise = torch.randn_like(X[mal])
        noise = noise / noise.norm(dim=1, keepdim=True).clamp_min(1e-12)
        X[mal] = noise * ref * scale

    elif kind == "ipm":
        gamma = float(params.get("gamma", 10.0))
        if len(benign) == 0:
            return X
        X[mal] = -gamma * X[benign].mean(dim=0, keepdim=True)

    elif kind == "alie":
        if len(benign) == 0:
            return X
        n, m = X.shape[0], len(mal_idx)
        mu = X[benign].mean(dim=0)
        sigma = X[benign].std(dim=0)
        s = n // 2 + 1 - m
        p = max(min((n - m - s) / max(n - m, 1), 1 - 1e-6), 1e-6)
        z = math.sqrt(2.0) * torch.erfinv(torch.tensor(2 * p - 1.0)).item()
        X[mal] = (mu - z * sigma).unsqueeze(0)

    return X


def describe(kind):
    return {
        "none": "攻撃なし",
        "label_flip": "ラベル反転 (データ汚染)",
        "sign_flip": "更新の符号反転",
        "gaussian": "大きなランダム更新",
        "ipm": "内積操作 (正常平均の逆向き)",
        "alie": "正常分布の内側に潜む",
    }.get(kind, kind)
