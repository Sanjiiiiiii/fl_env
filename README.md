# DEFT-FL — フェーズ1

MedMNIST を Dirichlet 分布で Non-IID 分割し、単純な CNN を FedAvg で学習する最小ベースラインです。GPFL、ビザンチン攻撃、Multi-Krum、FedKPer は後続フェーズのため含めていません。

## 実行

Python 3.10 以上の環境で依存関係を入れ、プロジェクトの親ディレクトリから実行します。

```powershell
pip install -r requirements.txt
python -m deft_fl.main --rounds 5
```

既定値は設計仕様書のフェーズ1設定に合わせ、BloodMNIST、50クライアント、参加率0.24、Dirichlet $\alpha=0.1$ です。初回実行時に MedMNIST が `data/` へダウンロードされます。

短い動作確認には、次を使えます。

```powershell
python -m deft_fl.main --num-clients 5 --participation-rate 1 --rounds 2 --batch-size 64
```

結果は `outputs/phase1/metrics.csv` と `outputs/phase1/global_model.pt` に出力されます。`metrics.csv` の `global_loss` が低下し、`global_accuracy` が上昇することをフェーズ1の確認指標にします。
