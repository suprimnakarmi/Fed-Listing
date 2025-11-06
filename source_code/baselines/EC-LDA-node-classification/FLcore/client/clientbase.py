import copy
import torch
import torch.nn as nn
import numpy as np
from FLcore.optimizer.dp_optimizer import DPSGD
from privacy_analysis.RDP.get_MaxSigma_or_MaxSteps import get_noise_multiplier



def label_DP_perturb_label(args,dataset):
    eps = args.epsilon
    k=args.num_classes
    ori_label = dataset.y
    per_label = torch.zeros_like(ori_label)
    for i in range(per_label.size()[0]):
        current_label = ori_label[i].item()
        # 计算保持不变的概率 p
        p = torch.exp(torch.tensor(eps)) / (torch.exp(torch.tensor(eps)) + k - 1)
        # 生成随机数
        r = torch.rand(1).item()
        if r <= p:
            per_label[i] = current_label
        else:
            # 扰动成其他标签
            # 生成一个不等于当前值的随机标签
            new_label = torch.randint(0, k, (1,)).item()
            while new_label == current_label:
                new_label = torch.randint(0, k, (1,)).item()
            per_label[i] = new_label

    return per_label




class Client():
    def __init__(self,args,id,dataset):
        
        self.model = copy.deepcopy(args.model)
        self.args=args
        self.id = id
        self.dataset = dataset
        # 按照子图节点的数量来计算聚合时候的权重
        self.num_nodes = self.dataset.num_nodes
        # self.num_edges = self.dataset.num_edges
        self.learning_rate = args.learning_rate
        self.num_classes = args.num_classes
        self.local_epochs = args.local_epochs 
        self.num_clients = args.num_clients
        
        self.num_train_nodes = sum(1 for flag in self.dataset.train_mask if flag)

        if self.args.defense_label_DP:
            self.dataset.per_y = label_DP_perturb_label(self.args,self.dataset)
           
        

        self.loss = nn.CrossEntropyLoss()
        if self.args.defense:
            orders = (list(range(2, 64)) + [128, 256, 512])  # 默认的lamda
            self.batch_size = round(self.num_train_nodes*self.args.sample_rate)
            self.sigma = get_noise_multiplier(self.args.epsilon,1e-5,self.args.sample_rate,self.args.local_epochs*self.args.global_rounds,orders)
            # 调学习率

            # 完全k叉树的节点个数，n层k叉树 n=gnn层数,k为节点的度
            n = self.args.gcn_hops
            k = self.args.DP_degree_limit
            # client的子图的邻接矩阵，以列为单位，看有多少个1，超过k个就随机留下k个1
            # 采样不做，用整个图训练，self.args.sample_rate=1
            self.sigma_mul = (k**n-1)/(k-1)
            # batch_size = self.num_nodes
            
            batch_size = self.batch_size
            sigma = self.sigma*self.sigma_mul
            # sigma = 50
            momentum = 0.0
            delta = 10 ** (-5)
            max_norm = 0.1 #不变
            self.optimizer = DPSGD(
                l2_norm_clip=max_norm,  # 裁剪范数
                noise_multiplier=sigma,
                minibatch_size=batch_size,  # 几个样本梯度进行一次梯度下降
                microbatch_size=1,  # 几个样本梯度进行一次裁剪，这里选择逐样本裁剪
                params=self.model.parameters(),
                lr=self.args.learning_rate_dp,
                momentum=momentum
            )
        else:
            # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,weight_decay=1e-4)
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer,
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay

    def set_parameters(self, model):  # 覆盖model.parameters()的操作；是get/set这种类型的操作
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()  # 深拷贝

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone() 


    def test_metrics(self):
        self.model.eval()  # 设置成“测试模式”,简单理解成不用反向传播了

        test_acc = 0
        test_num = 0   
        # for params in self.model.parameters():
        #     print(params.data)
            
        with torch.no_grad():
            out,emb = self.model(self.dataset.x,self.dataset.edge_index)
            pred = out.argmax(dim=1)
            test_correct = pred[self.dataset.test_mask] == self.dataset.y[self.dataset.test_mask]
            test_acc = int(test_correct.sum()) / int(self.dataset.test_mask.sum())
            return int(test_correct.sum()),int(self.dataset.test_mask.sum())
    
    def train_metrics(self):
        self.model.eval()
        train_num = 0
        losses = 0
        with torch.no_grad():
            out,emb = self.model(self.dataset.x,self.dataset.edge_index)
            loss = self.loss(out[self.dataset.train_mask],self.dataset.y[self.dataset.train_mask])
            train_num += int(self.dataset.train_mask.sum())
            losses += loss.item()*int(self.dataset.train_mask.sum())
        
        return losses,train_num