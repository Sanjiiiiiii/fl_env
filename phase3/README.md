# DEFT-FL Phase3

集約規則と攻撃を差し替えて比較するための土台。Phase2 を作り直したもの。

## Phase2 からの変更点

1. **集約規則が差し替え式** — `--aggregator {local,fedavg,multikrum,median,trimmedmean}`。
   すべて「更新ベクトル行列 X (n×d) → 集約後の更新 (d,)」という同じ形なので、
   BOBA も同じ場所に後から足せる。
2. **攻撃が差し替え式** — `--attack {none,label_flip,sign_flip,gaussian,ipm,alie}`。
3. **シードを統一** — Phase2 は攻撃あり/なしで seed が 42/43 と違っていたため、
   差に初期値の違いが混ざっていた。ここでは必ず同じ種を使う。
4. **評価をローカルテストに変更** — 各クライアントは自分の分布に沿ったテストセットを持つ。
   Phase2 は全員共通のテストセットで個別ヘッドを測っていたので、個別化に不利だった。
   (参考: α=0.1 / 攻撃なしで、共通テストだと約 0.39、ローカルテストだと約 0.92)
5. **指標を追加** — 平均精度に加えて、最悪クライアント精度、サンプル加重精度、
   クラス別再現率の最小値。少数派クラスが捨てられていないかを見るため。
6. **サーバ用データを確保** — BOBA が必要とするクラス別クリーンデータ
   (既定 20 件/クラス) を、クライアントへ配る前に取り分ける。

## 注意

`models.py` の `SplitCNN` は **GPFL ではない**。共有部とローカルヘッドに分けただけで、
手法としては FedPer 相当。GPFL の Conditional Valve と Global Category Embedding は未実装。
Phase2 のモデルクラス名が `GPFLModel` だったのは誤解を招くので改めた。

## 実行

```bash
cd phase3

# 単発
python main.py --aggregator fedavg    --attack none       --alpha 0.1 --rounds 100
python main.py --aggregator multikrum --attack label_flip --alpha 0.1 --rounds 100
python main.py --aggregator local     --attack none       --alpha 0.1 --rounds 100

# まとめて
python run_grid.py --rounds 60 --alpha 0.1 --seeds 42 43 44 \
    --aggregators local fedavg multikrum --attacks none label_flip ipm
```

結果は `../outputs/phase3/<設定タグ>.csv` と同名の `.json`、
一覧は `../outputs/phase3/results_summary.csv`。

## 次にやること

- [ ] 1 ラウンドあたりの実測時間を測り、実験の組み合わせ数を決める
- [ ] GPFL (CoV + GCE) を `models.py` に追加
- [ ] FedKPer の損失とスコア重み付けを追加。スコア偽装攻撃も攻撃の 1 種として入れる
- [ ] BOBA を `aggregators.py` の `boba()` に実装 (公式: github.com/baowenxuan/BOBA)
- [ ] 正常クライアントの推定ラベル比率がラウンド間でどれだけ安定するかを測る
