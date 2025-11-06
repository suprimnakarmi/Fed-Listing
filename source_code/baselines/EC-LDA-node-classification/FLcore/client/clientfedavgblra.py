import torch.nn as nn
import numpy as np
import time
from FLcore.client.clientbase import Client
import sys
import copy
import torch
import random
from privacy_analysis.RDP.compute_rdp import compute_rdp
from privacy_analysis.RDP.rdp_convert_dp import compute_eps
from utils import extract_node_subgraphs,DP_clip_edges



def get_err(emb):
    emb = emb.sum(dim=1)
    mean_emb = torch.mean(emb)
    err=torch.abs(emb-mean_emb).sum()/emb.size()[0]
    return err
    


class clientAVGblra(Client):
    def __init__(self,args,id,dataset):
        super().__init__(args,id,dataset)
        self.pre_model = copy.deepcopy(args.model)
        self.batch_gradient = None
        self.label_distribution = [0 for i in range(self.num_classes)]
        if self.args.defense:
            # tmp = self.dataset.edge_index
            self.dataset.edge_index = DP_clip_edges(self.dataset.edge_index,self.args.DP_degree_limit,self.num_nodes)[0]
            # self.get_subgraphs()
        train_index = []
        index = 0
        for flag in self.dataset.train_mask:
            if flag:
                train_index.append(index)
            index+=1
        self.cal_dis_batch(train_index)
    

        

    def get_subgraphs(self):
        self.subgraphs = extract_node_subgraphs(self.dataset, self.args.gcn_hops,self.args.device)


    # 正常训练
    def train(self):
        # print(self.id,end=' ')
        # sys.stdout.flush()
        print('\nclient id: ',self.id)
        self.model.to(self.args.device)
        self.model.train()
        self.dataset.x = self.dataset.x.to(self.args.device)
        self.dataset.edge_index = self.dataset.edge_index.to(self.args.device)
        self.dataset.y = self.dataset.y.to(self.args.device)
        

        # DP的时候设置为1
        max_local_epochs = self.local_epochs
        
        rdp = 0
        orders = (list(range(2, 64)) + [128, 256, 512])  # 默认的lamda
        epsilon_list = []
        iterations = 1
        delta = 10 ** (-5)
        all_emb = []
        for step in range(max_local_epochs):
            # a = self.dataset.train_mask
            # 无放回
            if self.args.defense:
                self.optimizer.zero_microbatch_grad()
                output, emb = self.model(self.dataset.x,self.dataset.edge_index)
                all_emb.append(torch.var(emb.sum(dim=1)))
                train_index = []
                index = 0
                for flag in self.dataset.train_mask:
                    if flag:
                        train_index.append(index)
                    index+=1
                #
                
                batch_index = random.choices(train_index, k=self.batch_size)
                for index in batch_index:
                    # mapping = self.subgraphs[index][1].item()
                    # output = self.model(self.dataset.x,self.dataset.edge_index)
                    # output = self.model(self.subgraphs[index][0].x,self.subgraphs[index][0].edge_index)
                    loss = self.loss(output[index],self.dataset.y[index])
                    # 此时的y就是源节点的标签
                    # loss = self.loss(output[mapping],self.subgraphs[index][0].y)

                    loss.backward(retain_graph=True)
                    # loss.backward()
                    # loss.backward(create_graph=True)
                    # grad = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)

                    self.optimizer.microbatch_step()
                self.optimizer.step_dp()


            else:
                self.optimizer.zero_grad()
                output, emb = self.model(self.dataset.x,self.dataset.edge_index)
                all_emb.append(torch.var(emb.sum(dim=1)))
                if self.args.defense_label_DP:
                    self.dataset.per_y = self.dataset.per_y.to(self.args.device)
                    loss = self.loss(output[self.dataset.train_mask],self.dataset.per_y[self.dataset.train_mask])
                    # loss = self.loss(output[self.dataset.train_mask],self.dataset.y[self.dataset.train_mask])
                else:
                    loss = self.loss(output[self.dataset.train_mask],self.dataset.y[self.dataset.train_mask])
                loss.backward()
                self.optimizer.step()
                # pred = output.argmax(dim=1)  # Use the class with highest probability.
                # test_correct = pred[self.dataset.test_mask] == self.dataset.y[self.dataset.test_mask]  # Check against ground-truth labels.
                # test_acc = int(test_correct.sum()) / int(self.dataset.test_mask.sum()) 

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()   
        
        return sum(all_emb)/len(all_emb)

        
    

    def get_abs_mean(self,grads):
        total_abs_sum = 0.0
        total_grads = 0
        layer = [0,1,2,3,4,5]
        for i ,grad in enumerate(grads):
            if i in layer:
                total_abs_sum+=grad.abs().sum()
                total_grads+=grad.numel()

        average_abs_sum = total_abs_sum / total_grads
        
        return total_abs_sum
    
    def train_gradient(self):
        print('\nclient id: ',self.id)
        # sys.stdout.flush()
        self.model.to(self.args.device)

        self.pre_model = copy.deepcopy(self.model)
        self.batch_gradient = copy.deepcopy(self.model)
        for params in self.batch_gradient.parameters():
            params.data.zero_()

        # 计算分布情况
        # self.cal_dis()
        self.model.train()
        max_local_epochs = self.local_epochs
        self.dataset.x = self.dataset.x.to(self.args.device)
        self.dataset.edge_index = self.dataset.edge_index.to(self.args.device)
        self.dataset.y = self.dataset.y.to(self.args.device)

        train_index = []
        index = 0
        for flag in self.dataset.train_mask:
            if flag:
                train_index.append(index)
            index+=1
        batch_index = train_index
        self.cal_dis_batch(train_index)
        

        rdp = 0
        orders = (list(range(2, 64)) + [128, 256, 512])  # 默认的lamda
        epsilon_list = []
        iterations = 1
        delta = 10 ** (-5)
        # all_emb实际上是方差
        all_emb = []
        # 误差
        all_err = []
        for step in range(max_local_epochs):
            if self.args.defense:
                # 
                self.optimizer.zero_microbatch_grad()

                output, emb = self.model(self.dataset.x,self.dataset.edge_index)
                all_err.append(get_err(emb))
                all_emb.append(torch.var(emb.sum(dim=1)))
                # train_index = []
                # index = 0
                # for flag in self.dataset.train_mask:
                #     if flag:
                #         train_index.append(index)
                #     index+=1
                # # batch_index = random.choices(train_index, k=self.batch_size)
                # batch_index = train_index
                # self.cal_dis_batch(batch_index)
                for index in batch_index:
                    loss = self.loss(output[index],self.dataset.y[index])
                    loss.backward(retain_graph=True)
                    self.optimizer.microbatch_step()
                
                self.optimizer.step_dp()
                for batch_gradient_params,params in zip(self.batch_gradient.parameters(),self.model.parameters()):

                    batch_gradient_params.data.add_(params.grad.data.clone().detach())
            
                
            else:
                self.optimizer.zero_grad()

                output, emb = self.model(self.dataset.x,self.dataset.edge_index)
                all_emb.append(torch.var(emb.sum(dim=1)))
                all_err.append(get_err(emb))
                if self.args.defense_label_DP:
                    self.dataset.per_y = self.dataset.per_y.to(self.args.device)
                    loss = self.loss(output[self.dataset.train_mask],self.dataset.per_y[self.dataset.train_mask])
                    # loss = self.loss(output[self.dataset.train_mask],self.dataset.y[self.dataset.train_mask])
                else:
                    loss = self.loss(output[self.dataset.train_mask],self.dataset.y[self.dataset.train_mask])
                loss.backward()
                if self.args.model_gradient_clip:
                    self.clip_gradient()

                for batch_gradient_params,params in zip(self.batch_gradient.parameters(),self.model.parameters()):
                   
                    batch_gradient_params.data.add_(params.grad.data.clone().detach())

                self.optimizer.step()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        return sum(all_emb)/len(all_emb),sum(all_err)/len(all_err)

    def cal_dis(self):
        for i in self.dataset.y:
            self.label_distribution[i]+=1
        self.label_distribution = [i/self.num_nodes for i in self.label_distribution]

    def cal_dis_batch(self,batch_index):
        self.label_distribution = [0 for i in range(self.num_classes)]
        for index in batch_index:
            self.label_distribution[self.dataset.y[index]]+=1
        self.label_distribution = [i/len(batch_index) for i in self.label_distribution]
        # print(sum(self.label_distribution))

    def get_gradient(self):
        ret = []
        for params in self.batch_gradient.parameters():
            # ret.append(params.data.clone().detach()*((self.local_epochs-1)/self.local_epochs))
            ret.append(params.data.clone().detach()*(1/self.local_epochs))
            # ret.append(params.data.clone().detach()*(1/(self.local_epochs*(self.args.gcn_hops))))
            # ret.append(params.data.clone().detach()*(4/5))
            # ret.append(params.data.clone().detach())
        return ret

    def get_label_distribution(self):
        return np.array(self.label_distribution)