from FLcore.client.clientfedavg import clientAVG
from FLcore.server.serverbase import Server
import sys

class FedAvg(Server):
    def __init__(self, args):
        super().__init__(args)
        self.set_clients(clientAVG)
    
    def train(self):
        for i in range(self.global_rounds+1):
            print('global round:',i,'trained client: ',end = '') 
            sys.stdout.flush()
            
            self.selected_clients = self.select_clients()
            self.send_models()

            for client in self.selected_clients:
                client.train()

            if i % self.eval_gap == 0:  # 几轮测试一次全局模型
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            self.receive_models()
            self.aggregate_parameters()

        print('')
        print('Finish')
        self.evaluate()            