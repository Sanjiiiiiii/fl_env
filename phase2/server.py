import copy
import torch

class Server:
    def __init__(self, global_model, device='cpu'):
        self.device       = device
        self.global_model = global_model.to(device)

    def get_global_base_params(self):
        return self.global_model.get_base_params()

    def aggregate(self, client_base_params_list, weights=None):
        """
        FedAvg集約：baseパラメータのみ加重平均
        weights: 各クライアントのデータ数（Noneなら均等）
        """
        n = len(client_base_params_list)
        if weights is None:
            weights = [1.0 / n] * n
        else:
            total = sum(weights)
            weights = [w / total for w in weights]

        aggregated = {}
        for key in client_base_params_list[0]:
            aggregated[key] = sum(
                w * params[key].to(self.device)
                for w, params in zip(weights, client_base_params_list)
            )

        # グローバルモデルのbaseを更新
        self.global_model.set_base_params(aggregated)
        return aggregated