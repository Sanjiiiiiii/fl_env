import copy
import torch

class Server:
    def __init__(self, global_model, device='cpu'):
        self.device       = device
        self.global_model = global_model.to(device)

    def get_global_parameters(self):
        return {k: v.cpu().clone() for k, v in self.global_model.state_dict().items()}

    def aggregate(self, client_params_list, weights=None):
        """標準FedAvg集約"""
        n = len(client_params_list)
        if weights is None:
            weights = [1.0 / n] * n
        else:
            total = sum(weights)
            weights = [w / total for w in weights]

        aggregated = {}
        for key in client_params_list[0]:
            aggregated[key] = sum(
                w * params[key].to(self.device)
                for w, params in zip(weights, client_params_list)
            )

        self.global_model.load_state_dict(aggregated)
        return aggregated