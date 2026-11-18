"""DEFT-FL Phase3: 集約規則と攻撃を差し替えて比較する実験ドライバ。

使い方の例:
  python main.py --aggregator fedavg    --attack none       --alpha 0.1
  python main.py --aggregator multikrum --attack label_flip --alpha 0.1
  python main.py --aggregator local     --attack none       --alpha 0.1

結果は results/<タグ>.csv と results/<タグ>.json に出る。
タグには設定が入るので、上書きされずに後から比べられる。
"""

import argparse
import copy
import json
import os
import random
import time

import numpy as np
import torch

from attacks import DATA_ATTACKS, describe
from client import build_client
from data import build_federated_data
from metrics import evaluate_clients
from models import build_model
from server import Server


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name):
    if name == "cuda" and not torch.cuda.is_available():
        print("警告: GPU 未検出。CPU にフォールバックします。")
        return "cpu"
    if name == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return name


def run(args):
    set_seed(args.seed)  # 攻撃あり/なしで必ず同じ種を使う (Phase2 ではここがずれていた)
    device = resolve_device(args.device)

    clients_data, server_data, global_test = build_federated_data(
        data_root=args.data_root,
        num_clients=args.num_clients,
        alpha=args.alpha,
        server_per_class=args.server_per_class,
        seed=args.seed,
    )

    global_model = build_model(args.algo, num_classes=args.num_classes)
    agg_ctx = {"f": args.f, "m": args.num_clients - args.f}
    server = Server(
        copy.deepcopy(global_model),
        aggregator=("fedavg" if args.aggregator == "local" else args.aggregator),
        device=device,
        agg_ctx=agg_ctx,
    )

    mal_ids = list(range(args.num_malicious))
    benign_ids = list(range(args.num_malicious, args.num_clients))

    clients = []
    for i, cd in enumerate(clients_data):
        clients.append(
            build_client(
                args.algo,
                client_id=i,
                model=global_model,
                train_dataset=cd["train"],
                test_dataset=cd["test"],
                label_dist=cd["label_dist"],
                batch_size=args.batch_size,
                lr=args.lr,
                local_epochs=args.local_epochs,
                device=device,
                is_malicious=(i in mal_ids),
                attack=args.attack,
                lamda=args.lamda,
                mu=args.mu,
            )
        )

    print(f"\n{'=' * 60}")
    print(f"algo={args.algo} | aggregator={args.aggregator} "
          f"| attack={args.attack} ({describe(args.attack)})")
    print(f"alpha={args.alpha} | malicious={args.num_malicious}/{args.num_clients} "
          f"| rounds={args.rounds} | seed={args.seed}")
    sizes = [c.train_size for c in clients]
    print(f"ローカル学習データ数: min={min(sizes)} max={max(sizes)} 合計={sum(sizes)}")
    print(f"{'=' * 60}\n")

    history = []
    for r in range(1, args.rounds + 1):
        t0 = time.time()

        if args.aggregator != "local":
            shared = server.get_shared()
            for c in clients:
                c.set_shared(shared)

        losses = [c.train() for c in clients]

        if args.aggregator != "local":
            params = [c.get_shared() for c in clients]
            weights = [c.train_size for c in clients]
            server.step(
                params,
                weights=weights,
                mal_idx=mal_ids,
                attack=args.attack,
                attack_params={"gamma": args.ipm_gamma, "scale": args.attack_scale},
            )

        if r % args.eval_gap == 0 or r == args.rounds:
            if args.aggregator != "local":
                # 評価は「集約後の共有パラメータ + 各自のヘッド」で行う
                shared = server.get_shared()
                for c in clients:
                    c.set_shared(shared)
            m = evaluate_clients(clients, benign_ids, args.num_classes)
            m["round"] = r
            m["train_loss"] = float(np.mean(losses))
            m["sec"] = round(time.time() - t0, 1)
            history.append(m)
            sel = server.last_selected
            picked = ""
            if sel is not None:
                n_mal_kept = len([i for i in sel if i in mal_ids])
                picked = f" | 採用{len(sel)}人(うち悪意{n_mal_kept})"
            print(
                f"Round {r:3d} | mean {m['mean_acc']:.4f} | worst {m['worst_acc']:.4f} "
                f"| pooled {m['pooled_acc']:.4f} | min-recall {m['min_recall']:.4f}"
                f"{picked} | {m['sec']}s"
            )

    tag = (f"{args.algo}_{args.aggregator}_{args.attack}_a{args.alpha}"
           f"_m{args.num_malicious}of{args.num_clients}_s{args.seed}")
    if args.algo == "gpfl":
        tag += f"_lam{args.lamda}_mu{args.mu}"
    os.makedirs(args.out, exist_ok=True)

    import csv

    csv_path = os.path.join(args.out, tag + ".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["round", "train_loss", "mean_acc", "worst_acc", "pooled_acc", "min_recall", "sec"])
        for m in history:
            w.writerow([m["round"], m["train_loss"], m["mean_acc"], m["worst_acc"],
                        m["pooled_acc"], m["min_recall"], m["sec"]])

    summary = {
        "config": vars(args),
        "final": history[-1] if history else {},
        "label_dist": [cd["label_dist"].tolist() for cd in clients_data],
    }
    with open(os.path.join(args.out, tag + ".json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n保存: {csv_path}")
    return history


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="../data")
    p.add_argument("--out", type=str, default="../outputs/phase3")
    p.add_argument("--num_clients", type=int, default=20)
    p.add_argument("--num_malicious", type=int, default=4)
    p.add_argument("--f", type=int, default=4, help="防御が想定する攻撃者数の上限")
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--num_classes", type=int, default=8)
    p.add_argument("--server_per_class", type=int, default=20, help="BOBA 用サーバデータ")
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--local_epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eval_gap", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--algo", type=str, default="fedper", choices=["fedper", "gpfl"],help="fedper: 層分離のみ / gpfl: CoV+GCE つき (ICCV 2023)")
    p.add_argument("--lamda", type=float, default=0.01, help="GPFL の norm 項の係数")
    p.add_argument("--mu", type=float, default=0.0, help="GPFL の CoV/GCE の weight_decay")
    p.add_argument("--aggregator", type=str, default="fedavg",choices=["fedavg", "multikrum", "median", "trimmedmean", "boba", "local"])
    p.add_argument("--attack", type=str, default="none",choices=["none", "label_flip", "sign_flip", "gaussian", "ipm", "alie"])
    p.add_argument("--ipm_gamma", type=float, default=10.0)
    p.add_argument("--attack_scale", type=float, default=10.0)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.attack in DATA_ATTACKS and args.num_malicious == 0:
        print("注意: num_malicious=0 なので攻撃は発生しません。")
    run(args)
