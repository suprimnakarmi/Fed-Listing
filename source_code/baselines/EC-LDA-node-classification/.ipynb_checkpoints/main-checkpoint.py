import torch
import argparse
import numpy as np
from FLcore.model.GNN import GCN,GAT,GraphSAGE
from FLcore.server.serverfedavg import FedAvg
from FLcore.server.serverfedavgblra import FedAvgBLRA
import os
import random
# torch.manual_seed(0)


# 主要控制数据划分
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def run(args,i):
    setup_seed(args.random_seed[i])
    # algorithm = None
    if args.algorithm == 'GCN':
        algorithm=GCN
    elif args.algorithm == 'GAT':
        algorithm=GAT
    elif args.algorithm == 'GraphSAGE':
        algorithm=GraphSAGE
    else:
        print('No such Algorithm!')
        exit()

    if args.dataset in ['Cora','CiteSeer','PubMed','FacebookPagePage','WikiCS','CoraFull','Reddit2']:
        args.model = algorithm(args.dataset,args.gcn_hops,features_in=args.feature_size,features_out=args.num_classes).to(args.device)
    else:
        print('No such Dataset!')
        exit()

    print(args.model)
    if args.attack:
        server = FedAvgBLRA(args)
    else:
        server = FedAvg(args)
    return server.train()  


if __name__ == '__main__':
    # setup_seed(0)
    
    parser = argparse.ArgumentParser()

    # global_rounds
    parser.add_argument('-gr', "--global_rounds", type=int, default=300,
                        help="Global Round in the FGL")
    # model
    parser.add_argument('-m','--model', type=str, default="gnn",    
                        help='GNN used in training')
    # Algorithm
    parser.add_argument('-al','--algorithm',type=str,default='GCN')     # GCN, GAT, GraphSAGE

    # GCN_hops
    parser.add_argument('-gh','--gcn_hops',type = int ,default=1,
                        help='number of gcn/gat/graphsage layer')
    
    # local_epochs
    parser.add_argument('-li', "--local_epochs", type=int, default=1,
                        help="local epochs of clients")
    
    all_dataset = [
        ['Cora',1433,7],
        ['CiteSeer',3703,6],
        ['PubMed',500,3],
        ['FacebookPagePage',128,4],
        ['CoraFull',8710,70],
        ['WikiCS',300,10]
        # ['Reddit2',602,41],     # 此数据集效果不佳
        # ['PPI']
        # 可以试一下PPI数据集
    ]

    # 数据集
    parser.add_argument('-data', "--dataset", type=str, default="Cora")     # Cora,CiteSeer,PubMed,FacebookPagePage,CoraFull,WikiCS

    # 数据集的feature_size
    parser.add_argument('-fs','--feature_size',type=int,default = 0)     # 1433, 3703, 500, 128, 8710
    
    # 分类任务的label种类数
    parser.add_argument('-ncl', "--num_classes", type=int, default=0)       # 7, 6, 3, 4, 70

    # 测试集和训练集的划分比例,训练集所占的比例
    parser.add_argument('-s','--split',type = float,default=0.8,
                        help = 'train/test dataset split percentage')

    # learning rate
    parser.add_argument('-lr', "--learning_rate", type=float, default=1,
                        help="learning rate for training")
    # 学习率是否要衰减
    parser.add_argument('-lrd','--learning_rate_decay',type=bool,default=not True)
    # 学习率衰减
    parser.add_argument('-lrdg','--learning_rate_decay_gamma',type=float,default=0.999)
    # num_clients
    parser.add_argument('-nc', "--num_clients", type=int, default=10,
                        help="Total number of clients")
    
    # device
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    

    # 经过多少轮进行一次测试
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")  
    
    # 客户端参加的比例(client drift程度)   
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
      
    # 攻击客户端的id
    parser.add_argument('-vc','--victim_client',type=list,default=[0],
                        help='victim client in training')
    # 实施攻击的轮次v
    parser.add_argument('-agr','--attack_global_round',type=list,default=[i for i in range(90,100)])
    
    # 共实施多少次攻击
    # 每次实验攻击一个round
    # 攻击轮次固定，固定一个攻击轮次，50轮
    parser.add_argument('-at','--attack_times',type=int,default=10)

    # O的缩放系数
    parser.add_argument('-so', "--scale_O", type=float, default=1,
                        help="the scale fraction of O")
    
    # 是否执行攻击
    parser.add_argument('-a','--attack',type=bool,default=True,
                        help = 'attack or not')
    # 是否进行梯度匹配迭代更新dummy_x
    parser.add_argument('-gm','--grad_match',type=bool,default= False,
                        help='gradients matching or not')
    # 进行梯度匹配更新的迭代次数 30最好
    parser.add_argument('-gmi','--grad_match_iteration',type=int,default=30,
                        help='gradients matching iterations')
    # 进行梯度匹配更新的正则系数
    parser.add_argument('-gma','--grad_match_alpha',type=float,default=0,
                        help='gradients matching alpha')
    
    # 是否要进行模型参数裁剪
    parser.add_argument('-mpc','--model_params_clip',type=bool,default= False,
                        help='clip the params of model or not')
    
    # 是否要裁剪梯度 
    parser.add_argument('-mgc','--model_gradient_clip',type=bool,default= False,
                        help='clip client model gradient or not')
    
    # 衡量所恢复分布与真实分布的相似度
    parser.add_argument('-mt','--metric',type=str,default='cosine_similarity',            # l2_norm,cosine_similarity,js_div,hellinger_dis
                        help='measure the similarity of the restored \
                            distribution to the true distribution')
    # True, False 
    # 是否进行主动攻击
    parser.add_argument('-aa','--active_attack',type=bool,default=True,
                        help='active attack or not')
    
    # # 是否进行对比
    # parser.add_argument('-comp','--compare',type=bool,default=False,
    #                     help = "compare with other method or not")
    
    # # 要对比算法的名字
    # parser.add_argument('-cn','--compare_name',type=str,default='ALDIA_no_active',                 # iLRG, LLG_star, ALDIA_no_active
    #                     help='select which methods to compare with')
    
    # 做五次实验取平均值，这是五次实验的随机数种子
    parser.add_argument('-rs','--random_seed',type = list,default = [0,1,2,3,4],
                        help = 'random seed for five experiments')
    # 当前是第几次实验
    parser.add_argument('-cen','--cur_experiment_no',type = int,default=0)
    # 是否做重复实验
    parser.add_argument('-re','--repeat_experiments',type = bool,default=True,
                        help = 'do repeat experiments or not')
    # 裁剪界限
    parser.add_argument('-cn','--clip_norm',type = float,default=0.01)

    # 攻击时期
    parser.add_argument('-ap','--attack_period',type = str,choices=['start','mid','end'],default = 'mid')
    
    # 攻击的
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable,using cpu")
        args.device = "cpu"    

    
    # ['Cora',1433,7],
    # ['CiteSeer',3703,6],
    # ['PubMed',500,3],
    # ['FacebookPagePage',128,4],
    # ['CoraFull',8710,70],
    # ['WikiCS',300,10]
    # 节点分类任务时，GraphSAGE对所有层都裁剪效果好一点
    args.algorithm = 'GCN'              # GCN, GAT, GraphSAGE
    args.gcn_hops = 2
    args.local_epochs = 5
    args.dataset = 'FacebookPagePage'
    # True, False
    args.compare = False
    args.repeat_experiments = True
    args.learning_rate = 0.1
    args.clip_norm = 0.01
    args.attack_period = 'mid'

    if args.dataset == 'CoraFull':
        args.learning_rate = 0.5

    if args.dataset == 'WikiCS':
        args.learning_rate = 0.5
    
    if args.attack_period == 'start':
        args.attack_global_round = [0]
    elif args.attack_period == 'mid':
        args.attack_global_round = [args.global_rounds/2]
    elif args.attack_period == 'end':
        args.attack_global_round = [args.global_rounds-1]
    else:
        print('\nWrong attack_period!')
        exit()

        # args.global_rounds = 500
    # args.active_attack = True
    # args.compare = False
    # args.compare_name = 'iLRG'          # iLRG, LLG_star, ALDIA_no_active

    # if args.compare:
    #     args.active_attack = False
    #     a=0
    
    # if args.compare_name == 'ALDIA_no_active':
    #     args.compare = False
    #     args.active_attack = False

    # attack_list = random.sample(range(1, 10), 5)
    # args.attack_global_round = sorted(random.sample(range(0, args.global_rounds), args.attack_times))
    
    

    flag = False
    for l in all_dataset:
        if args.dataset == l[0]:
            args.feature_size = l[1]
            args.num_classes = l[2]
            flag = True
            break
    if not flag:
        print("No such Dataset!")
        exit()
    
    # args.attack_global_round = [50]

    args.victim_client = [i for i in range(args.num_clients)]
    

    avg_cos_sim = []
    avg_js_div = []
    best_acc = [] 
    print(args)

    if args.repeat_experiments:
        for i in range(len(args.random_seed)):
            print("\n----------------Experiment ",i,":----------------")
            args.cur_experiment_no = i
            cos_sim, js_div, acc = run(args,i)
            avg_cos_sim.append(cos_sim)
            avg_js_div.append(js_div)
            best_acc.append(acc)
    else:
        cos_sim, js_div, acc = run(args,0)
        avg_cos_sim.append(cos_sim)
        avg_js_div.append(js_div)
        best_acc.append(acc)
    print(f"\n----------------Done! Report:----------------")
    # print('Done! Report:')
    if not args.compare:
        print("avg_all_cos_sim:",avg_cos_sim)
        print("avg_all_js_div:",avg_js_div)
        print("all_best_acc:",best_acc)

        print('avg_cos_sim:',sum(avg_cos_sim)/len(avg_cos_sim))
        print('avg_JS_div:',sum(avg_js_div)/len(avg_js_div))
        print('avg_best_acc:',sum(best_acc)/len(best_acc))

        print('avg_cos_sim:{: .3f}'.format(sum(avg_cos_sim)/len(avg_cos_sim)))
        print('avg_JS_div:{: .3f}'.format(sum(avg_js_div)/len(avg_js_div)))
        print('avg_best_acc:{: .3f}'.format(sum(best_acc)/len(best_acc)))
    else:
        avg_cos_sim_no_active = [row[0] for row in avg_cos_sim]
        avg_cos_sim_iLRG = [row[1] for row in avg_cos_sim]
        avg_cos_sim_LLG_star = [row[2] for row in avg_cos_sim]

        avg_js_div_no_active = [row[0] for row in avg_js_div]
        avg_js_div_iLRG = [row[1] for row in avg_js_div]
        avg_js_div_LLG_star = [row[2] for row in avg_js_div]

        print('cos_sim')
        print('no active:',avg_cos_sim_no_active)
        print('iLRG:',avg_cos_sim_iLRG)
        print('LLG_star:',avg_cos_sim_LLG_star)

        print('js_div')
        print('no active:',avg_js_div_no_active)
        print('iLRG:',avg_js_div_iLRG)
        print('LLG_star:',avg_js_div_LLG_star)
        print("all_best_acc:",best_acc)

        print('avg_cos_sim')
        print('no active:{: .3f}'.format(sum(avg_cos_sim_no_active)/len(avg_cos_sim_no_active)))
        print('iLRG:{: .3f}'.format(sum(avg_cos_sim_iLRG)/len(avg_cos_sim_iLRG)))
        print('LLG_star:{: .3f}'.format(sum(avg_cos_sim_LLG_star)/len(avg_cos_sim_LLG_star)))

        print('js_div')
        print('no active:{: .3f}'.format(sum(avg_js_div_no_active)/len(avg_js_div_no_active)))
        print('iLRG:{: .3f}'.format(sum(avg_js_div_iLRG)/len(avg_js_div_iLRG)))
        print('LLG_star:{: .3f}'.format(sum(avg_js_div_LLG_star)/len(avg_js_div_LLG_star)))

        print('avg_best_acc:{: .3f}'.format(sum(best_acc)/len(best_acc)))


    print(args)
    
