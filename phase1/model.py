import torch
import torch.nn as nn

class GPFLModel(nn.Module):
    """
    DEFT-FL Phase1: 特徴抽出器(base) + 分類ヘッド(head)の層分離モデル
    baseのみをサーバが集約し、headはクライアントがローカル保持する
    """
    def __init__(self, in_channels=3, num_classes=8):
        super().__init__()
        self.base = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten()
        )
        # ダミー入力でbase出力次元を自動計算
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 28, 28)
            feat_dim = self.base(dummy).shape[1]

        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        feat = self.base(x)
        return self.head(feat)

    def get_base_params(self):
        """集約対象パラメータ（baseのみ）"""
        return {k: v.clone() for k, v in self.state_dict().items()
                if k.startswith('base')}

    def set_base_params(self, base_params):
        """サーバから受け取ったbaseパラメータを適用"""
        state = self.state_dict()
        state.update(base_params)
        self.load_state_dict(state)