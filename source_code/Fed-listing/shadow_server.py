import os, torch, flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import parameters_to_ndarrays
from pathlib import Path

NUM_CLIENTS  = 10
NUM_ROUNDS   = 50

class RecordingFedAvg(FedAvg):
    def __init__(self, log_dir="shadow_logs/FL12", *args, **kwargs): # Change path after after "/" for each simulated FL training. For instance, in a random distribution setting, if training first simulated FL training, change "shadow_logs/FL12" to ""shadow_logs/FL1"" 
        super().__init__(*args, **kwargs)
        self.log_dir      = Path(log_dir)
        self.prev_global  = None
        self.cid_map   = {}      

    def configure_fit(self, server_round, parameters, client_manager, **kwargs):
        self.prev_global = parameters_to_ndarrays(parameters)
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(self, rnd, results, failures):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if self.prev_global is not None:
            for proxy, fitres in results:
                cur = parameters_to_ndarrays(fitres.parameters)
                deltas = [c - p for c, p in zip(cur, self.prev_global)]
                idx = self.cid_map.setdefault(proxy.cid, len(self.cid_map))
                torch.save(deltas, self.log_dir /
                           f"round{rnd}_client{idx}_deltas.pt")
        return super().aggregate_fit(rnd, results, failures)

if __name__ == "__main__":
    strategy = RecordingFedAvg(
        fraction_fit          = 1.0,
        min_fit_clients       = NUM_CLIENTS,
        min_available_clients = NUM_CLIENTS,
    )
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )


