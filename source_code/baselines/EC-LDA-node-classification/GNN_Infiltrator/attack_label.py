import copy
import torch
import torch.nn.functional as F


class Attack():
    def __init__(self, model, data, device='cpu'):
        self.model = model.to(device)
        self.data = data.to(device)
        self.device = device


class Attack_Label(Attack):
    def __init__(self, model, data, device):
        Attack.__init__(self, model, data, device)

    def label_infer(self, target_idx):
        # 1. add-node test
        est_data = copy.deepcopy(self.data)
        adv_idx = est_data.num_nodes
        
        # 1.1 add nodes
        est_data.x = torch.cat((est_data.x, torch.zeros(1, est_data.x.shape[1]).to(self.device)), dim=0) # cat dim 0: (nodes, feature) 

        # 1.2 add edges
        est_data.edge_index = torch.cat((est_data.edge_index, torch.tensor([[target_idx], [adv_idx]]).to(self.device)), dim=-1) # cat dim 1: (2, edges)
        est_data.edge_index = torch.cat((est_data.edge_index, torch.tensor([[adv_idx], [target_idx]]).to(self.device)), dim=-1)

        # optional: modify y, mask
        
        # 1.3 testing
        self.model.eval()
        output,emb = self.model(est_data.x, est_data.edge_index)
        adv_conf = output[adv_idx] # can only observe the adv side confidence
        # adv_pred = adv_conf.argmax()
        num_classes = adv_conf.shape[-1]

        score_topk,adv_topk = adv_conf.topk(num_classes)

        def rule(A, B):
            return B if torch.abs(A) <= torch.abs(B) else A

        if score_topk[0] >= score_topk[1] * 1.5:
            return adv_topk[0]
        else:
            eVec = torch.ones_like(est_data.x[adv_idx]) ## for general discrete datasets
            hVec = 1
            grad = []
            end = 2
            flag = False
            for i in range(len(score_topk)):
                if score_topk[i]*1.5>= score_topk[0]:
                    continue
                else:
                    end = i
                    flag = True
                    break
            if not flag:
                end = len(score_topk)

            with torch.no_grad():
                est_data.x[adv_idx] = eVec * hVec
                output,emb = self.model(est_data.x, est_data.edge_index)
                pos = output[adv_idx]
                
                for y in adv_topk[0:end]:
                    
                    fpos = F.nll_loss(torch.log(pos), y)

                    f = F.nll_loss(torch.log(adv_conf), y)

                    grad.append((fpos - f) / (hVec))

            return adv_topk[grad.index(rule(grad[0], grad[1]))]
        
    def get_dis(self,num_classes):
        label_dis = [0 for i in range(num_classes)]
        train_idx = torch.where(self.data.train_mask > 0)[0]
        for i in train_idx:
            label_dis[self.label_infer(i)]+=1

        train_num = train_idx.shape[0]
        label_dis = [i/train_num for i in label_dis]
        return label_dis