import argparse
import copy
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Subset
from torchvision import datasets, transforms

from model  import GPFLModel
from client import Client
from server import Server

def make_datasets(data_root='../data'):
    """BloodMNIST用のtrain/testデータセット"""
    import numpy as np
    data = np.load(os.path.join(data_root, 'bloodmnist.npz'))

    class BloodMNIST(torch.utils.data.Dataset):
        def __init__(self, images, labels, transform=None):
            self.images    = images
            self.labels    = labels.squeeze().astype(int)
            self.transform = transform
        def __len__(self):
            return len(self.images)
        def __getitem__(self, idx):
            img = torch.tensor(self.images[idx]).permute(2, 0, 1).float() / 255.0
            return img, self.labels[idx]

    train_ds = BloodMNIST(data['train_images'], data['train_labels'])
    test_ds  = BloodMNIST(data['test_images'],  data['test_labels'])
    return train_ds, test_ds

def split_dirichlet(dataset, num_clients, alpha, num_classes=8, seed=42):
    """
    Dirichletによるnon-IID分配
    alpha小→強いnon-IID、alpha大→IIDに近い
    """
    rng = np.random.default_rng(seed)
    labels = np.array([dataset[i][1] for i in range(len(dataset))])

    class_indices = [np.where(labels == c)[0] for c in range(num_classes)]
    client_indices = [[] for _ in range(num_clients)]

    for c_indices in class_indices:
        rng.shuffle(c_indices)
        proportions = rng.dirichlet([alpha] * num_clients)
        proportions = (proportions * len(c_indices)).astype(int)
        # 端数調整
        proportions[-1] = len(c_indices) - proportions[:-1].sum()
        splits = np.split(c_indices, proportions.cumsum()[:-1])
        for i, split in enumerate(splits):
            client_indices[i].extend(split.tolist())

    return client_indices

def run(args):
    torch.manual_seed(42)
    device = args.device

    # RocM用の記述
    if device == 'cuda':
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            torch.cuda.empty_cache()
        else:
            print("警告: GPU未検出、CPUにフォールバック")
            device = 'cpu'
            args.device = 'cpu'
    # ここまで追加

    print(f"\n{'='*50}")
    print(f"alpha={args.alpha} | malicious={args.num_malicious}/{args.num_clients} | rounds={args.rounds}")
    print(f"{'='*50}")

    # データ準備
    train_ds, test_ds = make_datasets(args.data_root)
    client_indices = split_dirichlet(
        train_ds, args.num_clients, alpha=args.alpha
    )

    # グローバルモデルとサーバ初期化
    global_model = GPFLModel(in_channels=3, num_classes=8)
    server = Server(global_model, device=device)

    # クライアント初期化
    clients = []
    for i in range(args.num_clients):
        is_mali = (i < args.num_malicious)
        subset  = Subset(train_ds, client_indices[i]) if client_indices[i] else Subset(train_ds, [0])
        client  = Client(
            client_id     = i,
            model         = copy.deepcopy(global_model),
            train_dataset = subset,
            test_dataset  = test_ds,
            batch_size    = args.batch_size,
            lr            = args.lr,
            local_epochs  = args.local_epochs,
            device        = device,
            is_malicious  = is_mali
        )
        clients.append(client)

    mali_ids   = list(range(args.num_malicious))
    benign_ids = list(range(args.num_malicious, args.num_clients))
    print(f"悪意クライアント: {mali_ids}")
    print(f"正常クライアント: {benign_ids[:5]}...（{len(benign_ids)}人）\n")

    # 学習ループ
    acc_history = []

    for r in range(args.rounds):
        # 1. サーバ→クライアント：baseパラメータ配布
        global_base = server.get_global_base_params()
        for c in clients:
            c.set_base_params(global_base)

        # 2. クライアントローカル学習
        for c in clients:
            c.train()

        # 3. クライアント→サーバ：baseパラメータ収集
        base_params_list = [c.get_base_params() for c in clients]
        weights = [len(client_indices[c.id]) for c in clients]

        # 4. サーバ集約（FedAvg）
        server.aggregate(base_params_list, weights=weights)

        # 5. 評価（正常クライアントの平均精度）
        if (r + 1) % args.eval_gap == 0:
            accs = [clients[i].evaluate() for i in benign_ids]
            mean_acc = np.mean(accs)
            acc_history.append(mean_acc)
            print(f"Round {r+1:3d} | Benign Acc: {mean_acc:.4f} "
            f"| Malicious ratio: {args.num_malicious/args.num_clients:.0%}")

    return acc_history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',     type=str,   default='../data')
    parser.add_argument('--num_clients',   type=int,   default=20)
    parser.add_argument('--num_malicious', type=int,   default=4)   # 20%
    parser.add_argument('--alpha',         type=float, default=0.5) # Dirichlet
    parser.add_argument('--rounds',        type=int,   default=100)
    parser.add_argument('--local_epochs',  type=int,   default=1)
    parser.add_argument('--batch_size',    type=int,   default=32)
    parser.add_argument('--lr',            type=float, default=0.001)
    parser.add_argument('--eval_gap',      type=int,   default=5)
    parser.add_argument('--device',        type=str,   default='cuda')
    args = parser.parse_args()

    # alpha（non-IID度）を変えて比較実験
    results = {}
    for alpha in [10.0, 0.5, 0.1]:
        args.alpha = alpha
        results[alpha] = run(args)

    # 結果プロット
    os.makedirs('../outputs/phase1', exist_ok=True)
    plt.figure(figsize=(10, 5))
    for alpha, accs in results.items():
        x = [(i+1)*args.eval_gap for i in range(len(accs))]
        label = f"alpha={alpha} ({'IID-like' if alpha>=10 else 'strong non-IID' if alpha<=0.1 else 'moderate non-IID'})"
        plt.plot(x, accs, label=label, linewidth=2)
    plt.xlabel("Round")
    plt.ylabel("Benign Client Accuracy")
    plt.title("DEFT-FL Phase1: FedAvg (base only) under Label-Flip Attack")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('../outputs/phase1/phase1_result.png', dpi=150)
    plt.show()
    print("\nOutputs saved to ../outputs/phase1/")