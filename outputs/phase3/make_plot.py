import glob
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

METRIC = 'mean_acc'
METRIC_LABEL = 'Benign Client Mean Accuracy'

attacks = ['none', 'label_flip', 'ipm']
attack_titles = {
    'none': 'No Attack (Benign)',
    'label_flip': 'Label-Flip Attack (m=4/20)',
    'ipm': 'IPM Attack (m=4/20)',
}

styles = {
    'fedavg_none':      {'label': 'FedAvg (Base)',      'color': 'black',   'ls': ':',  'marker': 'x'},
    'fedper_local':     {'label': 'FedPer + Local',     'color': '#1f77b4', 'ls': '--', 'marker': 'o'},
    'fedper_fedavg':    {'label': 'FedPer + FedAvg',    'color': '#1f77b4', 'ls': '-',  'marker': 's'},
    'fedper_multikrum': {'label': 'FedPer + MultiKrum', 'color': '#1f77b4', 'ls': '-.', 'marker': '^'},
    'gpfl_local':       {'label': 'GPFL + Local',       'color': '#ff7f0e', 'ls': '--', 'marker': 'o'},
    'gpfl_fedavg':      {'label': 'GPFL + FedAvg',      'color': '#ff7f0e', 'ls': '-',  'marker': 's'},
    'gpfl_multikrum':   {'label': 'GPFL + MultiKrum',   'color': '#ff7f0e', 'ls': '-.', 'marker': '^'},
}

csv_files = sorted(glob.glob('*.csv'))

for attack in attacks:
    plt.figure(figsize=(10, 6))
    plotted = False

    for file_path in csv_files:
        if file_path == 'results_summary.csv':
            continue
        if f'_{attack}_' not in file_path:
            continue
        if attack == 'none' and '_m4of20_' in file_path:
            continue

        if file_path.startswith('fedavg_none'):
            key = 'fedavg_none'
        else:
            match = re.match(r'^(fedper|gpfl)_(local|fedavg|multikrum)_', file_path)
            if not match:
                continue
            key = f'{match.group(1)}_{match.group(2)}'

        df = pd.read_csv(file_path)
        if 'round' not in df.columns or METRIC not in df.columns:
            continue

        st = styles.get(key, {'label': key, 'color': None, 'ls': '-', 'marker': 'o'})
        plt.plot(
            df['round'],
            df[METRIC],
            label=st['label'],
            color=st['color'],
            linestyle=st['ls'],
            marker=st['marker'],
            markersize=5,
            linewidth=2,
            alpha=0.85,
        )
        plotted = True

    if plotted:
        plt.xlabel('Round', fontsize=12)
        plt.ylabel(METRIC_LABEL, fontsize=12)
        plt.title(f'DEFT-FL Phase 3: {attack_titles[attack]} (alpha=0.1, non-IID)', fontsize=13)
        plt.ylim(0, 1.05)
        plt.legend(loc='lower right', fontsize=10, ncol=2)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = f'phase3_compare_{attack}_{METRIC}.png'
        plt.savefig(save_path, dpi=150)
        print(f'生成完了: {save_path}')
    plt.close()
