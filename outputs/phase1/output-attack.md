(fl_env) root@Motherpc:~/deft_fl/phase1# python main.py --device cuda --rounds 100 --eval_gap 5
GPU: AMD Radeon RX 7800 XT
VRAM: 15.8 GB

==================================================
alpha=10.0 | malicious=4/20 | rounds=100
==================================================
悪意クライアント: [0, 1, 2, 3]
正常クライアント: [4, 5, 6, 7, 8]...（16人）

MIOpen(HIP): Warning [ParseAndLoadDb] File is unreadable: "/root/anaconda3/envs/fl_env/lib/python3.10/site-packages/_rocm_sdk_libraries/share/miopen/db/gfx1101_30.HIP.fdb.txt"
Round   5 | Benign Acc: 0.4686 | Malicious ratio: 20%
Round  10 | Benign Acc: 0.6617 | Malicious ratio: 20%
Round  15 | Benign Acc: 0.7229 | Malicious ratio: 20%
Round  20 | Benign Acc: 0.7480 | Malicious ratio: 20%
Round  25 | Benign Acc: 0.7613 | Malicious ratio: 20%
Round  30 | Benign Acc: 0.7664 | Malicious ratio: 20%
Round  35 | Benign Acc: 0.7851 | Malicious ratio: 20%
Round  40 | Benign Acc: 0.7881 | Malicious ratio: 20%
Round  45 | Benign Acc: 0.7902 | Malicious ratio: 20%
Round  50 | Benign Acc: 0.8010 | Malicious ratio: 20%
Round  55 | Benign Acc: 0.8015 | Malicious ratio: 20%
Round  60 | Benign Acc: 0.8122 | Malicious ratio: 20%
Round  65 | Benign Acc: 0.8108 | Malicious ratio: 20%
Round  70 | Benign Acc: 0.8064 | Malicious ratio: 20%
Round  75 | Benign Acc: 0.8016 | Malicious ratio: 20%
Round  80 | Benign Acc: 0.8139 | Malicious ratio: 20%
Round  85 | Benign Acc: 0.8150 | Malicious ratio: 20%
Round  90 | Benign Acc: 0.8212 | Malicious ratio: 20%
Round  95 | Benign Acc: 0.8203 | Malicious ratio: 20%
Round 100 | Benign Acc: 0.8153 | Malicious ratio: 20%
GPU: AMD Radeon RX 7800 XT
VRAM: 15.8 GB

==================================================
alpha=0.5 | malicious=4/20 | rounds=100
==================================================
悪意クライアント: [0, 1, 2, 3]
正常クライアント: [4, 5, 6, 7, 8]...（16人）

Round   5 | Benign Acc: 0.3092 | Malicious ratio: 20%
Round  10 | Benign Acc: 0.4639 | Malicious ratio: 20%
Round  15 | Benign Acc: 0.5406 | Malicious ratio: 20%
Round  20 | Benign Acc: 0.5859 | Malicious ratio: 20%
Round  25 | Benign Acc: 0.6025 | Malicious ratio: 20%
Round  30 | Benign Acc: 0.6224 | Malicious ratio: 20%
Round  35 | Benign Acc: 0.6379 | Malicious ratio: 20%
Round  40 | Benign Acc: 0.6506 | Malicious ratio: 20%
Round  45 | Benign Acc: 0.6537 | Malicious ratio: 20%
Round  50 | Benign Acc: 0.6454 | Malicious ratio: 20%
Round  55 | Benign Acc: 0.6579 | Malicious ratio: 20%
Round  60 | Benign Acc: 0.6564 | Malicious ratio: 20%
Round  65 | Benign Acc: 0.6663 | Malicious ratio: 20%
Round  70 | Benign Acc: 0.6667 | Malicious ratio: 20%
Round  75 | Benign Acc: 0.6605 | Malicious ratio: 20%
Round  80 | Benign Acc: 0.6669 | Malicious ratio: 20%
Round  85 | Benign Acc: 0.6699 | Malicious ratio: 20%
Round  90 | Benign Acc: 0.6713 | Malicious ratio: 20%
Round  95 | Benign Acc: 0.6686 | Malicious ratio: 20%
Round 100 | Benign Acc: 0.6682 | Malicious ratio: 20%
GPU: AMD Radeon RX 7800 XT
VRAM: 15.8 GB

==================================================
alpha=0.1 | malicious=4/20 | rounds=100
==================================================
悪意クライアント: [0, 1, 2, 3]
正常クライアント: [4, 5, 6, 7, 8]...（16人）

Round   5 | Benign Acc: 0.2433 | Malicious ratio: 20%
Round  10 | Benign Acc: 0.2989 | Malicious ratio: 20%
Round  15 | Benign Acc: 0.3190 | Malicious ratio: 20%
Round  20 | Benign Acc: 0.3420 | Malicious ratio: 20%
Round  25 | Benign Acc: 0.3643 | Malicious ratio: 20%
Round  30 | Benign Acc: 0.3683 | Malicious ratio: 20%
Round  35 | Benign Acc: 0.3693 | Malicious ratio: 20%
Round  40 | Benign Acc: 0.3686 | Malicious ratio: 20%
Round  45 | Benign Acc: 0.3778 | Malicious ratio: 20%
Round  50 | Benign Acc: 0.3844 | Malicious ratio: 20%
Round  55 | Benign Acc: 0.3807 | Malicious ratio: 20%
Round  60 | Benign Acc: 0.3938 | Malicious ratio: 20%
Round  65 | Benign Acc: 0.3852 | Malicious ratio: 20%
Round  70 | Benign Acc: 0.3884 | Malicious ratio: 20%
Round  75 | Benign Acc: 0.3930 | Malicious ratio: 20%
Round  80 | Benign Acc: 0.3910 | Malicious ratio: 20%
Round  85 | Benign Acc: 0.3870 | Malicious ratio: 20%
Round  90 | Benign Acc: 0.3888 | Malicious ratio: 20%
Round  95 | Benign Acc: 0.3912 | Malicious ratio: 20%
Round 100 | Benign Acc: 0.3905 | Malicious ratio: 20%

Outputs saved to ../outputs/phase1/