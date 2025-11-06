import torch
import torch.nn as nn
import torch.nn.functional as F

def cosine_similarity_loss(true_grad,dummy_grad,device):
    scalar_prod = torch.as_tensor([0.0], device=device)
    true_norm = torch.as_tensor([0.0], device=device)
    rec_norm = torch.as_tensor([0.0], device=device)
    
    for rec_g, in_g in zip(dummy_grad, true_grad):
        scalar_prod += (rec_g * in_g).sum()  # elementwise product and then sum --> euclidean norm
        true_norm += in_g.pow(2).sum()
        rec_norm += rec_g.pow(2).sum()
    rec_loss = 1 - scalar_prod / (true_norm.sqrt() * rec_norm.sqrt())
    return rec_loss

class m_CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(m_CrossEntropyLoss, self).__init__()

    def forward(self,predictions,targets):
        # 对预测值进行softmax处理
        predictions_softmax = torch.nn.functional.softmax(predictions, dim=1)
        
        # 计算对数概率
        log_probs = torch.log(predictions_softmax)
        
        # 根据目标值获取对应类别的对数概率
        selected_log_probs = log_probs[range(len(targets)), targets]
        
        # 计算交叉熵损失
        loss = -torch.mean(selected_log_probs)
        
        return loss