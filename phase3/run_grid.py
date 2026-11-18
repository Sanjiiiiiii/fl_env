"""複数設定をまとめて回すバッチ実行。

例:
  python run_grid.py --rounds 60 --alpha 0.1 --seeds 42 43 44
  python run_grid.py --rounds 60 --aggregators fedavg multikrum --attacks none label_flip

main.run() をそのまま呼ぶので、個別実行と同じ CSV/JSON が出る。
最後に一覧表を results_summary.csv にまとめる。
"""

import argparse
import csv
import itertools
import os

from main import build_parser, run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algos", nargs="+", default=["fedper"])
    ap.add_argument("--lamda", type=float, default=0.01)
    ap.add_argument("--mu", type=float, default=0.0)
    ap.add_argument("--aggregators", nargs="+",
                    default=["local", "fedavg", "multikrum"])
    ap.add_argument("--attacks", nargs="+", default=["none", "label_flip"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--num_clients", type=int, default=20)
    ap.add_argument("--num_malicious", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--data_root", type=str, default="../data")
    ap.add_argument("--out", type=str, default="../outputs/phase3")
    cli = ap.parse_args()

    rows = []
    combos = itertools.product(cli.algos, cli.aggregators, cli.attacks, cli.seeds)
    for algo, agg, atk, seed in combos:
        args = build_parser().parse_args([])
        args.algo = algo
        args.lamda = cli.lamda
        args.mu = cli.mu
        args.aggregator = agg
        args.attack = atk
        args.seed = seed
        args.alpha = cli.alpha
        args.rounds = cli.rounds
        args.num_clients = cli.num_clients
        args.num_malicious = 0 if atk == "none" else cli.num_malicious
        args.device = cli.device
        args.data_root = cli.data_root
        args.out = cli.out

        hist = run(args)
        if hist:
            f = hist[-1]
            rows.append([algo, agg, atk, cli.alpha, seed, f["mean_acc"],
                         f["worst_acc"], f["pooled_acc"], f["min_recall"]])

    os.makedirs(cli.out, exist_ok=True)
    path = os.path.join(cli.out, "results_summary.csv")
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        if write_header:
            w.writerow(["algo", "aggregator", "attack", "alpha", "seed",
                        "mean_acc", "worst_acc", "pooled_acc", "min_recall"])
        w.writerows(rows)
    print(f"\n一覧: {path}")


if __name__ == "__main__":
    main()
