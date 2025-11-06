import copy
import torch
import torch.nn as nn
import numpy as np
from FLcore.client.clientbase import Client
from dataset.generate_Cora import generate_Cora
from dataset.generate_CiteSeer import generate_CiteSeer
from dataset.generate_PubMed import generate_PubMed
from dataset.generate_FacebookPagePage import generate_FacebookPagePage
from dataset.generate_CoraFull import generate_CoraFull
from dataset.generate_WikiCs import generate_WikiCS
from dataset.generate_LastFMAsia import generate_LastFM
from dataset.generate_AmazonC import generate_AmazonC
# from dataset.generate_Reddit import generate_Reddit2
import os

class Server():
    def __init__(self,args):
        self.args = args
        self.global_model = copy.deepcopy(args.model)
        self.rand_seed = self.args.random_seed[self.args.cur_experiment_no]
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.split = args.split
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)

        self.clients = []
        self.selected_clients = []
        self.current_acc = 0
        self.best_acc = 0
        self.eval_gap = args.eval_gap

        self.rs_test_acc=[]
        self.rs_train_loss=[]

    def dis_per_client(self,client_data):
        num_classes = self.args.num_classes
        num_clients = self.args.num_clients
        dis_list = [[0 for i in range(num_classes)] for i in range(num_clients)] 
        
        for i in range(num_clients):
            for x in client_data[i].y:
                index =  x -0;
                dis_list[i][index]+=1
        return dis_list



    def get_dataset(self):
        client_data = None
        save_file_path = 'data_save/'+self.dataset+'/'+str(self.rand_seed)+'_'+str(self.args.num_clients)+'.pt'
        if os.path.exists(save_file_path):
            # 如果文件存在，则读取数据
            client_data, self.args.num_classes = torch.load(save_file_path, weights_only=False)
            print("Successfully loaded data from", save_file_path)
        else:   
            print('Generate Dataset ',self.dataset,'...')
            if self.dataset == 'Cora':
                client_data,self.args.num_classes = generate_Cora(self.num_clients,self.split,self.rand_seed)
            elif self.dataset == "CiteSeer":
                client_data,self.args.num_classes = generate_CiteSeer(self.num_clients,self.split,self.rand_seed)
            elif self.dataset == 'PubMed':
                client_data,self.args.num_classes = generate_PubMed(self.num_clients,self.split,self.rand_seed)
            elif self.dataset == 'FacebookPagePage':
                client_data,self.args.num_classes = generate_FacebookPagePage(self.num_clients,self.split,self.rand_seed)
            elif self.dataset == 'CoraFull':
                client_data,self.args.num_classes = generate_CoraFull(self.num_clients,self.split,self.rand_seed)
            elif self.dataset == 'WikiCS':
                client_data,self.args.num_classes = generate_WikiCS(self.num_clients,self.split,self.rand_seed)
            elif self.dataset == 'LastFM':
                client_data,self.args.num_classes = generate_LastFM(self.num_clients,self.split,self.rand_seed)
            elif self.dataset =='AmazonComputers':
                client_data,self.args.num_classes = generate_AmazonC(self.num_clients,self.split,self.rand_seed)
            # elif self.dataset == 'Reddit2':
            #     client_data,self.args.num_classes = generate_Reddit2(self.num_clients,self.split,self.rand_seed)
            else:
                print("No such dataset!")
                exit()
            torch.save((client_data, self.args.num_classes), save_file_path)

        self.num_classes = self.args.num_classes
        self.dis_per_client(client_data)
        return client_data
    
    # 初始化客户端
    def set_clients(self, clientObj):  # clientObj是一个类对象了
        client_data = self.get_dataset()
        for i in range(self.num_clients):
            client = clientObj(self.args,i,client_data[i])
            self.clients.append(client)

    # 按比例选择客户端
    def select_clients(self):
        selected_clients = list(np.random.choice(self.clients, self.num_join_clients, replace=False))
        selected_clients.sort(key=lambda x: x.id)
        return selected_clients

    # 把server模型发送给每个client
    def send_models(self):  # sever->client
        assert (len(self.clients) > 0)

        for client in self.clients:
            client.set_parameters(self.global_model)     
    
    # 从client接收模型
    def receive_models(self):  # client->server
        assert (len(self.selected_clients) > 0)

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_nodes = 0
        
        for client in self.selected_clients:
            tot_nodes += client.num_nodes
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.num_nodes)
            self.uploaded_models.append(client.model)

        for i, w in enumerate(self.uploaded_weights):  # 权重归一化
            # 每个上传client的权重
            self.uploaded_weights[i] = w / tot_nodes    
    
    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for params in self.global_model.parameters():  # 全局模型置为0，方便后面累加了，上一行深拷贝只是为了model shape
            params.data.zero_()
        
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def save_global_model(self):
        model_path = os.path.join("save_models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model, model_path)

    def load_model(self):
        model_path = os.path.join("save_models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    def test_metrics(self):
        total_auc = []
        total_nodes = []
        total_correct = []
        for c in self.clients:    
            acc_num,num_nodes = c.test_metrics()
            total_correct.append(acc_num*1.0)
            total_nodes.append(num_nodes)
        
        ids = [c.id for c in self.clients]

        return ids, total_nodes, total_correct,total_auc
    
    def train_metrics(self):
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl * 1.0)

        ids = [c.id for c in self.clients]
        return ids, num_samples, losses    


    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        test_auc = sum(stats[3]) * 1.0 / sum(stats[1])
        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]

        self.current_acc = test_acc

        if test_acc > self.best_acc:
            self.best_acc = test_acc
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)

        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        return test_acc
