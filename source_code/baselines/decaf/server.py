import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import parameters_to_ndarrays
import pandas as pd
from flwr.common import FitIns
from clients import GCN, GraphSAGE, GIN
# from clients import GCN
import numpy as np

class LoggingFedAvg(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Will hold one dict per (round, client) record
        self.logs = []
        self.global_records = []

    def configure_fit(self, server_round, parameters, client_manager):
    
        instructions = super().configure_fit(server_round, parameters, client_manager)
        
        # Inject server_round into each FitIns.config
        for i, (client, fit_ins) in enumerate(instructions):
            config = dict(fit_ins.config)  # copy existing config
            config["server_round"] = server_round   # add current round
            instructions[i] = (client, FitIns(parameters=fit_ins.parameters, config=config))

        return instructions

    def aggregate_fit(self, server_round, results, failures):
        # For each client update, extract last-layer weights
        for client_proxy, fit_res in results:
            # all_wt = []
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            # Assume the penultimate array is the last layer's weight matrix
            # print(len(ndarrays))
            # for i in ndarrays:
            #     print(type(i))
            #     print(i)
            #     print("\n")
            #     print("=================================================")
            # first_w = ndarrays[1].flatten()
            second_w = ndarrays[-1].flatten()
            # print(first_w)
            # print(second_w)
            # combined = np.concatenate([first_w, second_w], axis = 0)
            flat = second_w.flatten()

            # Build a log record
            record = {
                "round": server_round,
                "client_id": client_proxy.cid,
                **{f"w_{i}": float(val) for i, val in enumerate(flat)}
            }
            self.logs.append(record)

        # Perform the normal FedAvg aggregation
        aggregated = super().aggregate_fit(server_round, results, failures)
        if (aggregated is not None) and (server_round >= 19):
            param_arrays = parameters_to_ndarrays(aggregated[0])
            param_keys = list(self.get_model_state_keys())

            for i, param in enumerate(param_arrays):
                layer_name = param_keys[i]
                param_type = "bias" if "bias" in layer_name else "weight"

                flat = param.flatten()
                record = {
                    "round": server_round,
                    "layer": layer_name, 
                    "type": param_type,
                    **{f"w_{j}": float(v) for j, v in enumerate(flat)}
                }
                self.global_records.append(record)

        return aggregated
    
    def get_model_state_keys(self):
        # model = GCN(in_feats=1433, hidden_feats=16, num_classes=7)
        model = GraphSAGE(in_feats=1433, hidden_feats=16, num_classes=7)
        return model.state_dict().keys()

    def save_global_log(self, filename = "global_all_layer_weights_new"):
        # print("here")
        # print(self.global_records)
        df = pd.DataFrame(self.global_records)
        df.to_csv(filename, index = False)


def main():
    strategy = LoggingFedAvg(
        fraction_fit = 1.0,
        fraction_evaluate = 1.0,
        min_fit_clients = 10,
        min_evaluate_clients = 10,
        min_available_clients = 10,
    )

    fl.server.start_server(
        server_address= "0.0.0.0:8080",
        config = fl.server.ServerConfig(num_rounds=20),
        strategy = strategy,
    )

    df = pd.DataFrame(strategy.logs)
    df.to_excel("last_layer_weights_new.xlsx", index=False)

    strategy.save_global_log()

if __name__ == "__main__":
    main()