"""サーバ。

やること:
  1. 共有パラメータを配る
  2. 集めた更新を行列 X (n, d) にする
  3. 更新汚染型の攻撃を適用する
  4. 差し替え可能な集約規則で 1 本の更新にまとめる
  5. 共有パラメータへ反映する
"""

import torch

from aggregators import flatten, get_aggregator, unflatten
from attacks import poison_updates


class Server:
    def __init__(self, global_model, aggregator="fedavg", device="cpu", agg_ctx=None):
        self.device = device
        self.global_model = global_model.to(device)
        self.aggregator_name = aggregator
        self.aggregator = get_aggregator(aggregator)
        self.keys = self.global_model.shared_keys()
        self.agg_ctx = agg_ctx or {}
        self.last_selected = None

    def get_shared(self):
        return self.global_model.get_shared()

    def step(self, client_params_list, weights, mal_idx, attack="none", attack_params=None):
        """1 ラウンド分の集約。"""
        template = self.get_shared()
        g = flatten(template, self.keys).to(self.device)

        X = torch.stack(
            [flatten(p, self.keys).to(self.device) - g for p in client_params_list]
        )
        X = poison_updates(X, mal_idx, attack, attack_params)

        ctx = dict(self.agg_ctx)
        delta = self.aggregator(X, weights=weights, ctx=ctx)
        self.last_selected = ctx.get("selected")

        new_flat = g + delta
        self.global_model.set_shared(unflatten(new_flat, template, self.keys))
        return X
