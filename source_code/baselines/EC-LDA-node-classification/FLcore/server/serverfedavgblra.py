from attacks.batch_label_reconstruction import batch_label_construction,batch_label_construction_output_layer
from iLRG.iLRG import batch_label_construction_iLRG
from LLG_star_random.LLG_star import batch_label_construction_LLG_star
from GNN_Infiltrator.attack_label import Attack_Label
from FLcore.client.clientfedavg import clientAVG
from FLcore.client.clientfedavgblra import clientAVGblra
from FLcore.server.serverbase import Server
import sys
import numpy as np
import torch
import copy
from result.view_attack import attack_performance,mean_params_and_metric_and_random,two_diff_y_axis,two_metric,acc_with_global_rounds
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import manhattan_distances

def compare_models(model1,model2):
    for params1,params2 in zip(model1.parameters(),model2.parameters()):
        if not torch.equal(params1,params2):
            print('参数不相同！')
            return False
    print('两个模型参数完全相同')
    return True


def mean_params_and_l2_norm(a,b,random):
    # 创建画布和第一个子图  
    fig, ax1 = plt.subplots()  
    
    # 绘制第一个数据集  
    ax1.plot(a, 'g-')  
    ax1.set_xlabel('X ')  
    ax1.set_ylabel('Y1 mean_params', color='g')  
    
    # 创建第二个子图并绑定到同一X轴  
    ax2 = ax1.twinx()  
    ax2.plot(b, 'b-')  
    ax2.set_ylabel('Y2 metric', color='b')  
    ax2.plot(random)
    
    plt.savefig("save/image.png")
    # 显示图形  
    plt.show()  


def mean_of_params(model):
    # 计算所有参数的绝对值之和的平均
    total_abs_sum = 0.0
    total_params = 0

    for i,param in enumerate(model.parameters()):
        # if i == 1:
        if True:
            total_abs_sum += param.abs().sum()
            total_params += param.numel()

    average_abs_sum = total_abs_sum / total_params
    return average_abs_sum



def mean_of_grad(grads):
    total_abs_sum = 0.0
    total_grads = 0
    layer = 2
    for i ,grad in enumerate(grads):
        # if i == layer:
        if True:
            total_abs_sum+=grad.abs().sum()
            total_grads+=grad.numel()

    average_abs_sum = total_abs_sum / total_grads
    return average_abs_sum

def metrics(restored_dis,true_dis,args,metric=None):
    res = None
    flag=True
    for i in restored_dis:
        if i!=0:
            flag=False
    if flag:
        restored_dis = [1e-64 for i in range(len(restored_dis))]
    return_metric = args.metric if metric == None else metric
    # return_metric = 'l2_norm'
    if return_metric == 'l2_norm':
        res = np.linalg.norm(restored_dis-true_dis)
    elif return_metric == 'cosine_similarity':
        res = np.dot(restored_dis,true_dis)/(np.linalg.norm(restored_dis)*np.linalg.norm(true_dis))
    elif return_metric == 'js_div':
        res_dis = np.array([a if a!=0.0 else 1e-10 for a in restored_dis])
        tre_dis = np.array([a if a!=0.0 else 1e-10 for a in true_dis])
        M = (res_dis+tre_dis)/2
        res = 0.5 * np.sum(res_dis*np.log(res_dis/M)) + 0.5 * np.sum(tre_dis*np.log(tre_dis/M))
    elif return_metric == 'hellinger_dis':
        # M = (res_dis+tre_dis)/2
        res = 1/np.sqrt(2)*np.linalg.norm(np.sqrt(restored_dis)-np.sqrt(true_dis))
    elif return_metric == "manhattan_dist":
        true_dis = true_dis.reshape(1,-1)
        restored_dis = restored_dis.reshape(1,-1)
        res = manhattan_distances(true_dis, restored_dis)
    else:
        print('No such Metric!')
        exit()

    return res

def sub_get_gradient(pre_model,cur_model,lr,local_epoch):
    gradient = copy.deepcopy(pre_model)
    for params,pre,cur in zip(gradient.parameters(),pre_model.parameters(),cur_model.parameters()):
        params.data = (pre.data.clone().detach() - cur.data.clone().detach())/(lr*local_epoch)
    
    ret = []
    for params in gradient.parameters():
        ret.append(params.data.clone().detach())
    return ret

def attack(client,cur_metric,cur_metric_random,args,num_classes,pre_model):
    gradient = client.get_gradient()
    # gradient = sub_get_gradient(pre_model,client.model,args.learning_rate,args.local_epochs)
    avg_grad = mean_of_grad(copy.deepcopy(gradient))
    client.batch_gradient=None
    train_data = client.dataset
    # restored_dis,O = batch_label_construction_output_layer(args,client.pre_model,gradient,client.num_nodes,feature_size=16)
    if args.compare:
        # restored_dis_no_active,O, sub,var  = batch_label_construction(None,args,client.pre_model,gradient,client.num_nodes,feature_size = args.feature_size,dis_y = None)
        # restored_dis_iLRG = batch_label_construction_iLRG(args,client.pre_model,gradient,client.num_nodes)
        # restored_dis_LLG_star = batch_label_construction_LLG_star(args,client.pre_model,gradient,client.num_nodes)
        restored_dis_no_active,O, sub,var  = batch_label_construction(None,args,client.pre_model,gradient,client.num_train_nodes,feature_size = args.feature_size,dis_y = None)
        restored_dis_iLRG = batch_label_construction_iLRG(args,client.pre_model,gradient,client.num_train_nodes)
        restored_dis_LLG_star = batch_label_construction_LLG_star(args,client.pre_model,gradient,client.num_train_nodes)

        O = 0
        sub = 0
        var = 0
    else:
        restored_dis,O, sub,var  = batch_label_construction(None,args,client.pre_model,gradient,client.num_nodes,feature_size = args.feature_size,dis_y = None)
    # restored_dis,O = batch_label_construction(None,args,client.pre_model,gradient,client.num_nodes,feature_size = 1433,dis_y = restored_dis)
    true_dis = client.get_label_distribution()
    
    if args.compare:
        metric_cos_sim_no_active = metrics(restored_dis_no_active,true_dis,args,"cosine_similarity")
        metric_cos_sim_iLRG = metrics(restored_dis_iLRG,true_dis,args,"cosine_similarity")
        metric_cos_sim_LLG_star = metrics(restored_dis_LLG_star,true_dis,args,"cosine_similarity")

        metric_js_div_no_active = metrics(restored_dis_no_active,true_dis,args,"js_div")
        metric_js_div_iLRG = metrics(restored_dis_iLRG,true_dis,args,"js_div")
        metric_js_div_LLG_star = metrics(restored_dis_LLG_star,true_dis,args,"js_div")
        cur_metric.append([[metric_cos_sim_no_active,metric_cos_sim_iLRG,metric_cos_sim_LLG_star],
                          [metric_js_div_no_active,metric_js_div_iLRG,metric_js_div_LLG_star]])
        
        metric_cos_sim = -1
        metric_js_div = -1

    else:
        metric_cos_sim = metrics(restored_dis,true_dis,args,"cosine_similarity")
        metric_js_div = metrics(restored_dis,true_dis,args,"js_div")
        metric_man_dist = metrics(restored_dis,true_dis,args,"manhattan_dist")
        metric_eucli_dist = metrics(restored_dis,true_dis,args,"l2_norm")
        print('\n Inferred distribution',restored_dis)
        print('\True distribution (GT)：',true_dis)
        print('metric {}:{:.4f}'.format('cosine_similarity',metric_cos_sim),end = ' ')
        print("\n")
        print('metric {}:{:.4f}'.format('js_div',metric_js_div),end = ' ')
        print("\n")
        print(f"metric: Manhattan distance {metric_man_dist}")
        print("\n")
        print('metric {}:{:.4f}'.format('Euclidean dist',metric_eucli_dist),end = ' ')
        cur_metric.append([metric_cos_sim,metric_js_div])

    random_guess = np.array([1/num_classes for i in range(num_classes)])
    # l2_norm_random = np.linalg.norm(random_guess-true_dis)

    metric_random_cos_sim = metrics(random_guess,true_dis,args,'cosine_similarity')
    metric_random_js_div = metrics(random_guess,true_dis,args,'js_div')

    cur_metric_random.append([metric_random_cos_sim,metric_random_js_div]) 
    return O,avg_grad,[metric_cos_sim,metric_js_div],sub,var

def report_attack(cur_metric,cur_metric_random,all_metric,all_random,args):
    # cur_metric_mean = sum(cur_metric)/len(cur_metric)
    if not args.compare:
        cur_metric_cos_sim = [row[0] for row in cur_metric]
        cur_metric_js_div = [row[1] for row in cur_metric]
        cur_metric_mean_cos_sim = sum(cur_metric_cos_sim)/len(cur_metric_cos_sim)
        cur_metric_mean_js_div = sum(cur_metric_js_div)/len(cur_metric_js_div)
        # cur_metric_mean_random = sum(cur_metric_random)/len(cur_metric_random)
        cur_metric_random_cos_sim = [row[0] for row in cur_metric_random]
        cur_metric_random_js_div = [row[1] for row in cur_metric_random]
        cur_metric_mean_random_cos_sim = sum(cur_metric_random_cos_sim)/len(cur_metric_random_cos_sim)
        cur_metric_mean_random_js_div = sum(cur_metric_random_js_div)/len(cur_metric_random_js_div)
        all_metric.append([cur_metric_mean_cos_sim,cur_metric_mean_js_div])
        all_random.append([cur_metric_mean_random_cos_sim,cur_metric_mean_random_js_div])
        print('\nmetric: cos-sim',"本轮的攻击与真实分布的平均metric为:",cur_metric_mean_cos_sim)
        print('metric: js_div',"本轮的攻击与真实分布的平均metric为:",cur_metric_mean_js_div)
        
        print('metric: cos-sim',"本轮随机猜测与真实分布的平均metric为:",cur_metric_mean_random_cos_sim)
        print('metric: js_div',"本轮随机猜测与真实分布的平均metric为:",cur_metric_mean_random_js_div)
    else:
        cur_metric_cos_sim = [row[0] for row in cur_metric]
        cur_metric_js_div = [row[1] for row in cur_metric]
        cur_metric_cos_sim_no_active = [row[0] for row in cur_metric_cos_sim]
        cur_metric_cos_sim_iLRG = [row[1] for row in cur_metric_cos_sim]
        cur_metric_cos_sim_LLG_star = [row[2] for row in cur_metric_cos_sim]

        cur_metric_js_div_no_active = [row[0] for row in cur_metric_js_div]
        cur_metric_js_div_iLRG = [row[1] for row in cur_metric_js_div]
        cur_metric_js_div_LLG_star = [row[2] for row in cur_metric_js_div]

        cur_metric_mean_cos_sim_no_active = sum(cur_metric_cos_sim_no_active)/len(cur_metric_cos_sim_no_active)
        cur_metric_mean_cos_sim_iLRG = sum(cur_metric_cos_sim_iLRG)/len(cur_metric_cos_sim_iLRG)
        cur_metric_mean_cos_sim_LLG_star = sum(cur_metric_cos_sim_LLG_star)/len(cur_metric_cos_sim_LLG_star)

        cur_metric_mean_js_div_no_active = sum(cur_metric_js_div_no_active)/len(cur_metric_js_div_no_active)
        cur_metric_mean_js_div_iLRG = sum(cur_metric_js_div_iLRG)/len(cur_metric_js_div_iLRG)
        cur_metric_mean_js_div_LLG_star = sum(cur_metric_js_div_LLG_star)/len(cur_metric_js_div_LLG_star)

        all_metric.append([[cur_metric_mean_cos_sim_no_active,cur_metric_mean_cos_sim_iLRG,cur_metric_mean_cos_sim_LLG_star],
                          [cur_metric_mean_js_div_no_active,cur_metric_mean_js_div_iLRG,cur_metric_mean_js_div_LLG_star]])
        
        print('\nmetric: cos-sim',"LDA 本轮的攻击与真实分布的平均metric为:",cur_metric_mean_cos_sim_no_active)
        print('metric: cos-sim',"iLRG 本轮的攻击与真实分布的平均metric为:",cur_metric_mean_cos_sim_iLRG)
        print('metric: cos-sim',"LLG_star 本轮的攻击与真实分布的平均metric为:",cur_metric_mean_cos_sim_LLG_star)

        print('metric: js_div',"LDA 本轮的攻击与真实分布的平均metric为:",cur_metric_mean_js_div_no_active)
        print('metric: js_div',"iLRG 本轮的攻击与真实分布的平均metric为:",cur_metric_mean_js_div_iLRG)
        print('metric: js_div',"LLG_star 本轮的攻击与真实分布的平均metric为:",cur_metric_mean_js_div_LLG_star)





def report_all_attack(attack_round,metric,args):
    print('metric:',args.metric,' 攻击结果为：')
    if len(attack_round)==0:
        print('本次实验未进行攻击！')
    if not args.compare:
        for i in range(len(attack_round)):
            print('攻击轮次：{:3}   metric:cos-sim: {:.5f}, js-div: {:.5f}'.format(attack_round[i],metric[i][0],metric[i][1]))
    else:
        print("no active")
        for i in range(len(attack_round)):
            print('攻击轮次：{:3}   metric:cos-sim: {:.5f}, js-div: {:.5f}'.format(attack_round[i],metric[i][0][0],metric[i][1][0]))
            # print('攻击轮次：{:3}   本轮的攻击与真实分布的平均metric为: {}'.format(attack_round[i],metric[i]))
        print('iLRG')
        for i in range(len(attack_round)):
            print('攻击轮次：{:3}   metric:cos-sim: {:.5f}, js-div: {:.5f}'.format(attack_round[i],metric[i][0][1],metric[i][1][1]))

        print('LLG_star')
        for i in range(len(attack_round)):
            print('攻击轮次：{:3}   metric:cos-sim: {:.5f}, js-div: {:.5f}'.format(attack_round[i],metric[i][0][2],metric[i][1][2]))



class FedAvgBLRA(Server):
    def __init__(self, args):
        super().__init__(args)
        self.set_clients(clientAVGblra)
        # self.clip_client()
        # self.victim = [i for i in range(self.args.num_clients)]
        self.victim = args.victim_client
        self.pre_model = copy.deepcopy(self.global_model)

    
    # clip全局模型的参数
    def active_attack_clip_params(self):

        params_dict = self.global_model.state_dict()
        dict_size = len(params_dict)
       
        total_norm = 0
        for i,params in enumerate(self.global_model.parameters()):
            if params.requires_grad:
                total_norm+=params.data.norm(2).item() ** 2.
        total_norm = total_norm ** .5
        # clip_coef = min(self.l2_norm_clip / (total_norm + 1e-6), 1.)
        clip_coef = min(self.args.clip_norm / (total_norm + 1e-6), 1.)

        if self.local_epochs == 1:
            param_clip_list = [dict_size-3]
        else:
            param_clip_list = [i for i in range(dict_size)]
        # param_clip_list = [0,1]
        for i,params in enumerate(self.global_model.parameters()):
            # if i in param_clip_list:
            if True:
                if params.requires_grad:
                    params.data.mul_(clip_coef)
                    # params.data.mul_(5)

        # for params in self.global_model.parameters():
        #     print(params.data)

        return clip_coef

    def parameters_mul_var(self):
        for params in self.global_model.parameters():
            params.data.mul_(self.args.variable_mul)
    
    def infiltrator(self):
        cos_sim = []
        js_div = []
        man_dis =[]
        for client in self.clients:
            if client.id in self.victim:
                al = Attack_Label(self.global_model,client.dataset,self.args.device)
                res_dis = al.get_dis(client.num_classes)
                true_dis = client.get_label_distribution()
                cos_sim.append(metrics(res_dis,true_dis,self.args,"cosine_similarity"))
                js_div.append(metrics(res_dis,true_dis,self.args,"js_div"))
        
        return sum(cos_sim)/len(cos_sim), sum(js_div)/len(js_div)

    

    def train(self):
        all_metric = []
        mean_params = []
        all_random = []
        all_clip_coef = []
        all_O = []
        all_O_metric = []
        all_acc = []
        all_sub = []
        round_avg_metric_cos_sim = []
        round_avg_metric_js_div = []
        all_emb_var = []
        all_err = []
        
        for i in range(self.global_rounds):
            sys.stdout.flush()

            self.selected_clients = self.select_clients()

            # 保存上一轮的全局模型
            self.pre_model = copy.deepcopy(self.global_model)
            
            if self.args.task == 'cal_var_err' and i in self.args.attack_global_round:
                self.parameters_mul_var()
            
            elif (self.args.task == 'ALDA' or self.args.task=='defense' or self.args.task=='label_DP') and i in self.args.attack_global_round:
                clip_coef = self.active_attack_clip_params()
                all_clip_coef.append(clip_coef)
            elif self.args.task =='why_active' or self.args.task == 'compare' or self.args.task == 'hops_var':
                pass

                
            self.send_models()
            cur_metric = []
            cur_metric_random = []
            cur_O = []
            cur_var = []

            print('\nExperiment ',self.args.cur_experiment_no,end='')
            print(f"-------------Round number: {i}-------------")
            cur_round_sum_cos_sim = 0
            cur_round_sum_js_div = 0

            cur_round_emb = []
            cur_round_err = []
            
            for client in self.selected_clients:
                # if client.id in self.victim :

                if client.id in self.victim and i in self.args.attack_global_round:
                    emb,err = client.train_gradient()
                    cur_round_emb.append(emb)
                    cur_round_err.append(err)
                    O,avg_grad,O_metric,sub,var=attack(client,cur_metric,cur_metric_random,self.args,client.num_classes,self.pre_model)
                    cur_var.append(var)
                    cur_round_sum_cos_sim+=O_metric[0]
                    cur_round_sum_js_div+=O_metric[1]
                    O_metric = [O,O_metric,sub,var]
                    all_O_metric.append(O_metric) 
                    # print(" O的值为:",O)
                    # print('avg grad:',avg_grad.item())
                    mean_param = mean_of_params(self.global_model)
                    # mean_params.append(avg_grad.item())
                    mean_params.append(mean_param.item())
                    cur_O.append(O)
                    
                else:
                    client.train()
            round_avg_metric_cos_sim.append(cur_round_sum_cos_sim/len(self.victim))
            round_avg_metric_js_div.append(cur_round_sum_js_div/len(self.victim))
            if i in self.args.attack_global_round:
                all_emb_var.append(cur_round_emb)
                all_err.append(cur_round_err)
            
            if len(cur_var) != 0:
                all_sub.append(sum(cur_var)/len(cur_var))    
            # if self.args.model_params_clip:
            #     self.clip_client()

            # if i % self.eval_gap == 0 :  # 几轮测试一次全局模型
            # # if i % self.eval_gap == 0 and i in self.args.attack_global_round:  # 几轮测试一次全局模型
            #     print(f"\n-------------Round number: {i}-------------")
            #     # print("\nEvaluate global model")
            #     self.evaluate()
            #     if i in self.args.attack_global_round:
            #         report_attack(cur_l2_norm,cur_l2_norm_random,all_l2_norm)
            #     # mean = mean_of_params(self.global_model)
            #     # print('本轮模型参数的绝对值平均为：',mean.item())
            #     # mean_params.append(mean.item())

            self.receive_models()
            self.aggregate_parameters()
            # print(cur_metric)

            # if i in self.args.attack_global_round and self.args.task!='why_active':
            #     print('\n主动攻击的分类结果：',i)
            #     self.evaluate()
            if (self.args.task == 'ALDA' or self.args.task == 'defense' or self.args.task=='label_DP') and i in self.args.attack_global_round :
                self.global_model = copy.deepcopy(self.pre_model)

            # if not self.args.compare and i in self.args.attack_global_round and self.args.task!='why_active':
            #     self.global_model = copy.deepcopy(self.pre_model)
                
            
            if i % self.eval_gap == 0 :  # 几轮测试一次全局模型
            # if i % self.eval_gap == 0 and i in self.args.attack_global_round:  # 几轮测试一次全局模型
                # 这个send只是为了做评估，不做评估可以删除下面这一行
                self.send_models()
                print("\nEvaluate global model")
                cur_acc = self.evaluate()
                all_acc.append(cur_acc)
                if i in self.args.attack_global_round:
                    report_attack(cur_metric,cur_metric_random,all_metric,all_random, self.args)
                # mean = mean_of_params(self.global_model)
                # print('本轮模型参数的绝对值平均为：',mean.item())
                # mean_params.append(mean.item())
            all_O.append(cur_O)
        
        if len(all_emb_var[0])!=0:
            avg_emb_var = sum(all_emb_var[0])/len(all_emb_var[0])
        if len(all_err[0])!=0:
            avg_err = sum(all_err[0])/len(all_err[0])
        
        self.send_models()

        
        
        print('\nFinish')
        final_acc = self.evaluate()
        print('Best Acc:','{: .6f}'.format(self.best_acc))
        print('Final Acc:','{: .6f}'.format(final_acc))
        print('victim client:',self.args.victim_client)

        if self.args.task == 'compare':
            cos_sim_infi,js_div_infi = self.infiltrator()
            all_metric[0][0][0] = cos_sim_infi
            all_metric[0][1][0] = js_div_infi
            # return cos_sim_infi, js_div_infi,final_acc, all_acc
        # 画图区域
        # two_diff_y_axis([row[0] for row in all_metric],[row[1] for row in all_metric],'cos_sim','JS-div','Metrics with Global Round')
        # two_diff_y_axis(all_acc,all_sub,'Accuracy','Var * 1e5','Acc with Var of Os')
        # mean_params_and_metric_and_random(mean_params,all_metric,all_random)

        # mean_params_and_l2_norm(mean_params,all_metric,all_random)
        
        # attack_performance(self.args.attack_global_round,all_metric,all_random)
        # 画图区域
        # acc_with_global_rounds(all_acc,self.args)
        report_all_attack(self.args.attack_global_round,all_metric,self.args)
        # if not self.args.compare:
        if self.args.task != "compare":
            print('All Metrics:',all_metric)
            if self.args.task == "ALDA":
                print('method: ALDA')
            all_metric_cos_sim = [row[0] for row in all_metric]
            all_metric_js_div = [row[1] for row in all_metric]
            avg_metric_cos_sim = sum(all_metric_cos_sim)/len(all_metric_cos_sim)
            avg_metric_js_div =  sum(all_metric_js_div)/len(all_metric_js_div)

            print('mean of metric cos-sim:','{: .6f}'.format(sum(all_metric_cos_sim)/len(all_metric_cos_sim)))
            print('mean of metric cos-sim','{}'.format(sum(all_metric_cos_sim)/len(all_metric_cos_sim)))

            print('mean of metric js-div:','{: .6f}'.format(sum(all_metric_js_div)/len(all_metric_js_div)))
            print('mean of metric js-div','{}'.format(sum(all_metric_js_div)/len(all_metric_js_div)))
            # print('\nO:',all_O)
            # print(all_clip_coef)
            # print('value of O,metric[cos-sim,js-div],sub of Os,var of Os')

            # for o_m in all_O_metric:
            #     print(o_m)

            sum_var = 0
            num = 0
            for i in all_O_metric:
                if type(i[3]) == int:
                    sum_var+=i[3]
                else:
                    sum_var+=i[3].item()
                num+=1
            mean_var = sum_var/num
            print('Mean Var: {:.6f}'.format(mean_var))
            print(all_clip_coef)

            return avg_metric_cos_sim, avg_metric_js_div, final_acc, all_acc, round_avg_metric_cos_sim,\
                  round_avg_metric_js_div, mean_var, avg_emb_var, avg_err
        else:
            all_metric_cos_sim_no_active = [row[0][0] for row in all_metric]
            all_metric_cos_sim_iLRG = [row[0][1] for row in all_metric]
            all_metric_cos_sim_LLG_star = [row[0][2] for row in all_metric]

            all_metric_js_div_no_active = [row[1][0] for row in all_metric]
            all_metric_js_div_iLRG = [row[1][1] for row in all_metric]
            all_metric_js_div_LLG_star = [row[1][2] for row in all_metric]

            avg_metric_cos_sim_no_active = sum(all_metric_cos_sim_no_active)/len(all_metric_cos_sim_no_active)
            avg_metric_cos_sim_iLRG = sum(all_metric_cos_sim_iLRG)/len(all_metric_cos_sim_iLRG)
            avg_metric_cos_sim_LLG_star = sum(all_metric_cos_sim_LLG_star)/len(all_metric_cos_sim_LLG_star)

            avg_metric_js_div_no_active =  sum(all_metric_js_div_no_active)/len(all_metric_js_div_no_active)
            avg_metric_js_div_iLRG =  sum(all_metric_js_div_iLRG)/len(all_metric_js_div_iLRG)
            avg_metric_js_div_LLG_star =  sum(all_metric_js_div_LLG_star)/len(all_metric_js_div_LLG_star)
            
            print('cos-sim')
            print('Infi mean of metric cos-sim:','{: .6f}'.format(avg_metric_cos_sim_no_active))
            print('iLRG mean of metric cos-sim:','{: .6f}'.format(avg_metric_cos_sim_iLRG))
            print('LLG_star mean of metric cos-sim:','{: .6f}'.format(avg_metric_cos_sim_LLG_star))

            print('js_div')
            print('Infi mean of metric js_div:','{: .6f}'.format(avg_metric_js_div_no_active))
            print('iLRG mean of metric js_div:','{: .6f}'.format(avg_metric_js_div_iLRG))
            print('LLG_star mean of metric js_div:','{: .6f}'.format(avg_metric_js_div_LLG_star))
            
            mean_var = None

            return [avg_metric_cos_sim_no_active,avg_metric_cos_sim_iLRG,avg_metric_cos_sim_LLG_star], \
                    [avg_metric_js_div_no_active,avg_metric_js_div_iLRG,avg_metric_js_div_LLG_star], \
                    final_acc, all_acc, round_avg_metric_cos_sim, round_avg_metric_js_div, mean_var, avg_emb_var, avg_err


