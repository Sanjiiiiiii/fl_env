"""集約規則 (AGR)。

すべて「更新ベクトル (delta = ローカル後 - グローバル前)」の行列 X: (n, d) を受け取り、
集約後の delta ベクトル (d,) を返す、という同じ形にそろえてある。
こうしておくと main 側で --aggregator を差し替えるだけで比較できる。

実装済み:
  fedavg      : データ数による加重平均 (防御なしの基準)
  multikrum   : Blanchard et al. 2017
  median      : 座標ごとの中央値 (Yin et al. 2018)
  trimmedmean : 座標ごとの刈り込み平均 (Yin et al. 2018)

未実装 (次の段階):
  boba        : Bao et al. AISTATS 2024。サーバのクラス別データが必要なので
                ctx["server_grads"] を使う想定でフックだけ用意してある。
"""

import torch


# ---------- 平坦化ユーティリティ ----------
def flatten(params, keys):
    return torch.cat([params[k].reshape(-1).float() for k in keys])


def unflatten(vec, template, keys):
    out, i = {}, 0
    for k in keys:
        n = template[k].numel()
        out[k] = vec[i : i + n].view(template[k].shape).to(template[k].dtype)
        i += n
    return out


# ---------- 各 AGR ----------
def fedavg(X, weights=None, ctx=None):
    if weights is None:
        return X.mean(dim=0)
    w = torch.as_tensor(weights, dtype=X.dtype, device=X.device)
    w = w / w.sum()
    return (X * w.unsqueeze(1)).sum(dim=0)


def _krum_scores(X, f):
    n = X.shape[0]
    k = max(n - f - 2, 1)
    d2 = torch.cdist(X, X) ** 2
    scores = []
    for i in range(n):
        row = torch.cat([d2[i, :i], d2[i, i + 1 :]])
        scores.append(torch.sort(row).values[:k].sum())
    return torch.stack(scores)


def multikrum(X, weights=None, ctx=None):
    ctx = ctx or {}
    n = X.shape[0]
    f = int(ctx.get("f", max(1, n // 5)))
    m = int(ctx.get("m", max(1, n - f)))
    scores = _krum_scores(X, f)
    sel = torch.argsort(scores)[:m]
    if ctx is not None:
        ctx["selected"] = sel.tolist()
    return X[sel].mean(dim=0)


def median(X, weights=None, ctx=None):
    return X.median(dim=0).values


def trimmedmean(X, weights=None, ctx=None):
    ctx = ctx or {}
    n = X.shape[0]
    f = int(ctx.get("f", max(1, n // 5)))
    f = min(f, (n - 1) // 2)
    sorted_X, _ = torch.sort(X, dim=0)
    if f > 0:
        sorted_X = sorted_X[f : n - f]
    return sorted_X.mean(dim=0)


def boba(X, weights=None, ctx=None):
    raise NotImplementedError(
        "BOBA は次の段階で実装する。ctx['server_grads'] (クラス別サーバ勾配) が必要。"
    )


AGGREGATORS = {
    "fedavg": fedavg,
    "multikrum": multikrum,
    "median": median,
    "trimmedmean": trimmedmean,
    "boba": boba,
}


def get_aggregator(name):
    if name not in AGGREGATORS:
        raise ValueError(f"unknown aggregator: {name}. 選択肢: {list(AGGREGATORS)}")
    return AGGREGATORS[name]
