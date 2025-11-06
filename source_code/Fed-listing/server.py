import os
import torch
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common.parameter import parameters_to_ndarrays

class RecordingStrategy(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Will hold the “old” global weights at the start of each round
        self.prev_weights = None

    def configure_fit(self, server_round, parameters, client_manager, **kwargs):
        # Convert the incoming global Parameters proto → list of np.ndarrays
        self.prev_weights = parameters_to_ndarrays(parameters)
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round, results, failures):
        # Ensure our output folder exists
        os.makedirs("logs", exist_ok=True)

        # If this isn’t the very first round, compute deltas
        if self.prev_weights is not None:
            for client_proxy, fit_res in results:
                # Convert this client’s returned Parameters → list of ndarrays
                curr_weights = parameters_to_ndarrays(fit_res.parameters)
                # Compute per-layer deltas
                deltas = [cw - pw for cw, pw in zip(curr_weights, self.prev_weights)]
                # Save to disk
                client_id = client_proxy.cid  
                torch.save(deltas, f"logs/round{server_round}_client{client_id}_deltas.pt")

        # Finally, perform the normal FedAvg aggregation
        return super().aggregate_fit(server_round, results, failures)


if __name__ == "__main__":
    NUM_CLIENTS = 10
    NUM_ROUNDS = 50

    strategy = RecordingStrategy(
        fraction_fit=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )
