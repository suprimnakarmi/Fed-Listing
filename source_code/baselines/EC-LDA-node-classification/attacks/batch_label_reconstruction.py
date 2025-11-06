import torch
import copy
from torch import optim
from torch_geometric.data import Data
from torch_geometric.utils import erdos_renyi_graph
import torch.nn.functional as F
import numpy as np
from attacks.utils.loss_function import cosine_similarity_loss,m_CrossEntropyLoss
from attacks.utils.regularizations import R_clip_dummy_x,R_scale_dummy_x,R_clip_dummy_y,R_scale_dummy_y,R_sum_equal_one,R_centralized
from attacks.utils.attack_model import Dense
# import torch

def regular(x,y):
    clip_x = R_clip_dummy_x(x)
    scale_x = R_scale_dummy_x(x)
    centralized_x = R_centralized(x)

    clip_y = R_clip_dummy_y(y)
    scale_y = R_scale_dummy_y(y)

    sum_equal1 = R_sum_equal_one(y)
    centralized_y = R_centralized(y)
    # 3 3 0 5 5 1 4 目前这样是最好的
    # 3 3 0 5 5 1 4 This is the best at the moment
    return 1e-3*clip_x+1e-3*scale_x +0*centralized_x +1e-5*clip_y+1e-5*scale_y+1e-1*sum_equal1+1e-4*centralized_y

# 备份，这个梯度匹配函数是同时优化x和y的
# Backup, this gradient matching function optimizes x and y simultaneously
def gradients_matching_x_y(x,y,edge_index,gradients,model,args,dummy_node_size,feature_size,count = 0,min_loss=100,res = None,res_y = None):
    optimizer = optim.LBFGS([x, y], lr = 0.5)
    loss_f = torch.nn.CrossEntropyLoss()
    loss_f_cos = cosine_similarity_loss
    
    for iters in range(args.grad_match_iteration+1):

        def closure():
            optimizer.zero_grad()
            output = model(x,edge_index)
            # y_one_hot = F.softmax(y, dim=-1)
            loss = loss_f(output,y)
            dummy_grad = torch.autograd.grad(loss,model.parameters(),create_graph=True)

            grad_diff = 0
            # for gx,gy in zip(dummy_grad,gradients):
            #     grad_diff += ((gx - gy) ** 2).sum()
            x_mean = torch.mean(torch.abs(x))
            grad_diff = loss_f_cos(gradients,dummy_grad,args.device)

            grad_diff = grad_diff + args.grad_match_alpha*x_mean + regular(x,y)
            grad_diff.backward()
           
            return grad_diff
        optimizer.step(closure)
        
        if iters % 1 == 0: 
            current_loss = closure()
            print(iters, "%.4f" % current_loss.item())
            if current_loss<0.0000001 and not np.isnan(current_loss.detach().numpy()):
                return x.clone().detach() ,res_y
            if (current_loss>10000 and iters>2*args.grad_match_iteration/3 )or(np.isnan(current_loss.detach().cpu().numpy()))or current_loss>100:
                dummy_x = torch.normal(mean=0.0,std=0.001,size=(dummy_node_size,feature_size)).to(args.device).requires_grad_(True)
                dummy_y = torch.normal(mean=0.0,std=0.001,size = (dummy_node_size,args.num_classes)).to(args.device).requires_grad_(True)
                # dummy_y = torch.normal(mean=0.0,std=0.001,size = (dummy_node_size,)).requires_grad_(True)
                # dummy_y = torch.randint(0,args.num_classes+1,size = (dummy_node_size,))
                # dummy_y = dummy_y.to(torch.float32)
                # dummy_y = torch.zeros(size=(dummy_node_size,args.num_classes)).requires_grad_(False)
                print('LBFGS损失过大,重新初始化dummy数据')
                if count+1 >3:
                    print('重新初始化dummy数据次数过多,使用之前最优dummy数据')
                    return res,res_y
                return gradients_matching_x_y(dummy_x,dummy_y,edge_index,gradients,model,args,dummy_node_size,feature_size,count+1,min_loss,res,res_y)
            if current_loss.item() < min_loss:
                min_loss = current_loss.item()
                res = x.clone().detach()
                res_y = y.clone().detach()
            
            # history.append(To_image(dummy_data[0].cpu()))   

    return res ,res_y

def regular_x(x,y):
    clip_x = R_clip_dummy_x(x)
    scale_x = R_scale_dummy_x(x)
    centralized_x = R_centralized(x)

    # clip_y = R_clip_dummy_y(y)
    # scale_y = R_scale_dummy_y(y)

    # sum_equal1 = R_sum_equal_one(y)
    # centralized_y = R_centralized(y)
    # # 3 3 0 5 5 1 4 目前这样是最好的
    # return 1e-3*clip_x+1e-3*scale_x +0*centralized_x +1e-5*clip_y+1e-5*scale_y+1e-1*sum_equal1+1e-4*centralized_y
    return 1e-3*clip_x+1e-3*scale_x +0*centralized_x 


# 这个梯度匹配函数只优化x
# The gradient matching function only optimizes x
def gradients_matching_x(x,y,edge_index,gradients,model,args,dummy_node_size,feature_size,count = 0,min_loss=100,res = None,res_y = None):
    optimizer = optim.LBFGS([x], lr = 0.5)
    loss_f = torch.nn.CrossEntropyLoss()
    loss_f_cos = cosine_similarity_loss
    
    for iters in range(args.grad_match_iteration+1):

        def closure():
            optimizer.zero_grad()
            output = model(x,edge_index)
            # y_one_hot = F.softmax(y, dim=-1)
            loss = loss_f(output,y)
            dummy_grad = torch.autograd.grad(loss,model.parameters(),create_graph=True)

            grad_diff = 0
            # for gx,gy in zip(dummy_grad,gradients):
            #     grad_diff += ((gx - gy) ** 2).sum()
            x_mean = torch.mean(torch.abs(x))
            grad_diff = loss_f_cos(gradients,dummy_grad,args.device)
            # grad_diff += args.grad_match_alpha*x_mean
            # clip_x = R_clip_dummy_x(x)
            # scale_x = R_scale_dummy_x(x)
            # # scale_x = 0
            # clip_y = R_clip_dummy_y(y)
            # scale_y = R_scale_dummy_y(y)
            # # scale_y = 0
            # sum_equal1 = R_sum_equal_one(y)
            # centralized_y = R_centralized(y)
            # grad_diff = grad_diff+1e-3*clip_x+1e-3*scale_x + args.grad_match_alpha*x_mean+1e-5*clip_y+1e-5*scale_y+1e-5*sum_equal1+1e-3*centralized_y
            grad_diff = grad_diff + regular_x(x,y)
            grad_diff.backward()
            # current_loss = grad_diff.clone().detach()
            return grad_diff
        optimizer.step(closure)
        # x.data.clamp_(0,1)
        # y.data.clamp_(0,1)
        if iters % 1 == 0: 
            current_loss = closure()
            print(iters, "%.4f" % current_loss.item())
            if current_loss<0.0000001 and not np.isnan(current_loss.detach().numpy()):
                return x.clone().detach() ,res_y
            if (current_loss>10000 and iters>2*args.grad_match_iteration/3 )or(np.isnan(current_loss.detach().cpu().numpy()))or current_loss>100:
                dummy_x = torch.normal(mean=0.0,std=0.001,size=(dummy_node_size,feature_size)).to(args.device).requires_grad_(True)
                # dummy_y = torch.normal(mean=0.0,std=0.001,size = (dummy_node_size,7)).to(args.device).requires_grad_(True)
                dummy_y = y
                # dummy_y = torch.normal(mean=0.0,std=0.001,size = (dummy_node_size,)).requires_grad_(True)
                # dummy_y = torch.randint(0,args.num_classes+1,size = (dummy_node_size,))
                # dummy_y = dummy_y.to(torch.float32)
                # dummy_y = torch.zeros(size=(dummy_node_size,args.num_classes)).requires_grad_(False)
                print('LBFGS损失过大,重新初始化dummy数据')
                if count+1 >3:
                    print('重新初始化dummy数据次数过多,使用之前最优dummy数据')
                    return res,res_y
                return gradients_matching_x(dummy_x,dummy_y,edge_index,gradients,model,args,dummy_node_size,feature_size,count+1,min_loss,res,res_y)
            if current_loss.item() < min_loss:
                min_loss = current_loss.item()
                res = x.clone().detach()
                res_y = y.clone().detach()
            
            # history.append(To_image(dummy_data[0].cpu()))   

    return res ,res_y

def get_p_and_O(args,model,dummy_g):
    # last_input = None
    p = None
    # get O
    model.eval()
    def hook_fn(module, input, output):
        global last_input
        last_input = input[0].clone().detach()

    hook_handle = model.fc2.register_forward_hook(hook_fn)
    
    # 此时已经得到O的值
    # At this point, the value of O has been obtained
    output, emb = model(dummy_g.x,dummy_g.edge_index)
    O = copy.deepcopy(last_input)
    # 移除钩子
    # Remove Hook
    hook_handle.remove()

    p = torch.nn.functional.softmax(output,dim = 1).clone().detach()
    return p.detach(),O.detach()

def set_y(dummy_node_size,dis):
    
    dis = dis*dummy_node_size
    tensor_list = []
    for i in range(len(dis)):
        tmp_tensor = torch.full((int(dis[i]),),i).requires_grad_(False)
        tensor_list.append(tmp_tensor)
    res_tensor = copy.deepcopy(tensor_list[0])
    for i in range(1,len(tensor_list)):
        res_tensor = torch.cat((res_tensor,tensor_list[i]),dim=0)
    max_index = np.argmax(dis)
    if len(res_tensor)<dummy_node_size:
        missing_size = dummy_node_size-len(res_tensor)
        tmp_tensor = torch.full((missing_size,),max_index).requires_grad_(False)
        res_tensor = torch.cat((res_tensor,tmp_tensor),dim=0)
    return res_tensor

                                                                                                            #  0.004
def batch_label_construction(train_data,args,model,gradients,num_nodes,feature_size,dis_y,dummy_node_size=1000,dummy_link_p=0.005):
    # dummy_node_size = num_nodes
    if train_data == None:
        # 暂定随机正态分布
        # Provisional random normal distribution
        # 节点
        # node
        # dummy_x = torch.randn(dummy_node_size,feature_size)
        # 非标准正态分布
        # Nonstandard normal distribution
        # 0.001
        dummy_x = torch.normal(mean=0.0,std=0.001,size=(dummy_node_size,feature_size)).to(args.device).requires_grad_(True)
        if type(dis_y) != np.ndarray:
            dummy_y = torch.normal(mean=0.0,std=0.001,size = (dummy_node_size,args.num_classes)).to(args.device).requires_grad_(True)
        else:
            dummy_y = set_y(dummy_node_size,dis_y)
        # dummy_y = torch.normal(mean=0.0,std=0.001,size = (dummy_node_size,)).requires_grad_(True)
        # dummy_y = torch.randint(0,args.num_classes+1,size = (dummy_node_size,))
        # dummy_y = dummy_y.to(torch.float32).requires_grad_(True)
        # 边
        # Side
        # dummy_y = torch.zeros(size=(dummy_node_size,args.num_classes)).requires_grad_(False)
        edge_index = erdos_renyi_graph(dummy_node_size,dummy_link_p)
        edge_index = edge_index.to(args.device)
        if args.grad_match:
            if type(dis_y) != np.ndarray:
                dummy_x,dummy_y = gradients_matching_x_y(dummy_x,dummy_y,edge_index,gradients,model,args,dummy_node_size,feature_size)
            else:
                dummy_x,dummy_y = gradients_matching_x(dummy_x,dummy_y,edge_index,gradients,model,args,dummy_node_size,feature_size)

        dummy_g = Data(x=dummy_x,y = dummy_y,edge_index=edge_index)
    else:
        dummy_g = train_data

    ps,Os = get_p_and_O(args,copy.deepcopy(model),dummy_g)
    p = ps.clone().detach()
    # ps = ps.mean(dim=0)
    O = (Os.sum(dim=1)).mean()*args.scale_O

    # O = (Os.sum(dim=1))
    # O = 5.5
    # tmp1 = Os.sum(dim=1)
    # tmp2 = torch.Tensor([O for i in range(dummy_node_size)])
    # tmp3 = (torch.Tensor([O for i in range(dummy_node_size)])-Os.sum(dim=1))*1e6
    tmp4 = torch.var(Os.sum(dim=1))

    # K = num_nodes
    K = dummy_node_size
    dW = None
    dW = gradients[-2].sum(dim=1) if len(gradients[-1].size()) == 1 else gradients[-1].sum(dim=1)
    # dW = gradients[-1].sum(dim=1)
    # print('最后一层参数大小：',torch.sum(torch.abs(gradients[-3])))
    # calculate the counts
    # counts = K * ps - K * (dW / O)
    
    # O_down = (Os.mean(dim=1)).mean()
    # O_up = (Os.mean(dim=0)).sum()

    # counts = K*ps*O_up/O_down - K*(dW/O_down)

    O_down = (Os.sum(dim=1)).mean()
    O_up = Os.sum(dim=1)
    # counts = (p*O_up.view(-1,1)).sum(dim = 0)/O_down - K * dW/O_down
    counts = ((p*O_up.view(-1,1)).sum(dim = 0) - K * dW)/O_down


    min_value = counts.min()
    if min_value<0:
        counts-=min_value

    # for i in range(len(counts)):
    #     if counts[i]<0:
    #         counts[i] = 0
    counts = counts/counts.sum()
    # counts = torch.nn.functional.softmax(counts)
    abs_ = sum(abs(torch.Tensor([O.cpu() for i in range(dummy_node_size)])-Os.cpu().sum(dim=1)))
    return counts.cpu().numpy(),O.cpu(),(abs_).cpu(),(tmp4*1e3).cpu()






def gradients_matching_output_layer(x,y,gradients,model,args,dummy_node_size,feature_size,count=0,min_loss=100,res = None):
    optimizer = optim.LBFGS([x, y], lr = 1)
    loss_f = torch.nn.CrossEntropyLoss()
    loss_f_cos = cosine_similarity_loss
    res = None
    min_loss = 100
    for iters in range(args.grad_match_iteration+1):

        def closure():
            optimizer.zero_grad()
            output = model(x)
            # y_one_hot = F.softmax(y, dim=-1)
            loss = loss_f(output,y)
            dummy_grad = torch.autograd.grad(loss,model.parameters(),create_graph=True)

            grad_diff = 0
            # for gx,gy in zip(dummy_grad,gradients):
            #     grad_diff += ((gx - gy) ** 2).sum()
            x_mean = torch.mean(torch.abs(x))
            grad_diff = loss_f_cos(gradients,dummy_grad,args.device)
            # grad_diff += args.grad_match_alpha*x_mean
            clip_x = R_clip_dummy_x(x)
            scale_x = R_scale_dummy_x(x)
            clip_y = R_clip_dummy_y(y)
            scale_y = R_scale_dummy_y(y)
            grad_diff = grad_diff+1e-2*clip_x+1e-2*scale_x + args.grad_match_alpha*x_mean+1e-2*clip_y+1e-2*scale_y
            grad_diff.backward()
            # current_loss = grad_diff.clone().detach()
            return grad_diff
        optimizer.step(closure)
        if iters % 1 == 0: 
            current_loss = closure()
            print(iters, "%.4f" % current_loss.item())
            if current_loss<0.0000001 and not np.isnan(current_loss.detach().numpy()):
                return x.clone().detach() 
            if (current_loss>100 and iters>2*args.grad_match_iteration/3 )or(np.isnan(current_loss.detach().numpy()))or current_loss>100:
                dummy_x = torch.normal(mean=0.0,std=0.001,size=(dummy_node_size,feature_size)).requires_grad_(True)
                dummy_y = torch.normal(mean=0.0,std=0.001,size = (dummy_node_size,args.num_classes)).requires_grad_(True)
                # print('LBFGS损失过大,重新初始化dummy数据')
                print("LBFGS loss is too large, reinitialize dummy data")
                if count+1 >10:
                    # print('重新初始化dummy数据次数过多,使用之前最优dummy数据')
                    print("Reinitialize dummy data too many times, use the previous optimal dummy data")
                    return res
                return gradients_matching_output_layer(copy.deepcopy(dummy_x),dummy_y,gradients,model,args,dummy_node_size,feature_size,count+1,min_loss,res)
            if current_loss.item() < min_loss:
                min_loss = current_loss.item()
                res = x.clone().detach()
            
            # history.append(To_image(dummy_data[0].cpu()))   

    return res 

def get_p_and_O_output_layer(args,model,dummy_x_output_layer):
    params = []
    i = 0
    for param in model.parameters():
        params.append(param.data.clone().detach())
        i+=1
    O = dummy_x_output_layer
    
    out_put_mid = torch.mm(dummy_x_output_layer,params[i-2].T)
    output = out_put_mid+params[i-1]

    p = torch.nn.functional.softmax(output,dim = 1).clone().detach()
    return p.detach(),O.detach()
    

def batch_label_construction_output_layer(args,model,gradients,num_nodes,feature_size,dummy_node_size=1000):
    
    dummy_x_tmp = torch.normal(mean=0.0,std=0.001,size=(dummy_node_size,feature_size)).requires_grad_(False)
    dummy_x_output_layer = torch.clamp(dummy_x_tmp.clone().detach(), min=0, max=1).requires_grad_(True)
    dummy_y_putput_layer = torch.normal(mean=0.0,std=0.001,size = (dummy_node_size,args.num_classes)).requires_grad_(True)
    if args.grad_match:
        gradients = gradients[-2:]
        params = []
        for param in model.parameters():
            params.append(copy.deepcopy(param))
        model_dense = Dense(feature_size,args.num_classes,params[-2:])

        dummy_x_output_layer = gradients_matching_output_layer(dummy_x_output_layer,dummy_y_putput_layer,gradients,model_dense,args,dummy_node_size,feature_size)
    ps,Os = get_p_and_O_output_layer(args,copy.deepcopy(model),dummy_x_output_layer)
    ps = ps.mean(dim=0)
    O = (Os.sum(dim=1)).mean()*args.scale_O

    K = num_nodes
    dW = None
    # dW = gradients[-2].sum(dim=1) if len(gradients[-1].size()) == 1 else gradients[-1].sum(dim=1)
    dW = gradients[-2].sum(dim=1)
    counts = K * ps - K * dW / O

    for i in range(len(counts)):
        if counts[i]<0:
            counts[i] = 0
    counts = counts/counts.sum()
    return counts.numpy(),O