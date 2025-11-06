import torch
import argparse
import numpy as np
from FLcore.model.GNN import GCN,GAT,GraphSAGE, GIN
from FLcore.server.serverfedavg import FedAvg
from FLcore.server.serverfedavgblra import FedAvgBLRA
import os
import random
from result.view_attack import acc_with_two_metric_mean_var
from result.utils import compute_mean_var
# torch.manual_seed(0)


# 主要控制数据划分
# Main control data division
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def run(args,i):
    setup_seed(args.random_seed[i])
    # algorithm = None
    if args.algorithm == 'GCN':
        algorithm=GCN
    elif args.algorithm == 'GAT':
        algorithm=GAT
    elif args.algorithm == 'GraphSAGE':
        algorithm = GraphSAGE
    elif args.algorithm == 'GIN':
        algorithm = GIN
    else:
        print('No such Algorithm!')
        exit()

    if args.dataset in ['Cora','CiteSeer','PubMed','FacebookPagePage','WikiCS','CoraFull','LastFM', 'AmazonComputers']:
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
    parser.add_argument('-al','--algorithm',type=str,default='GraphSAGE')     # GCN, GAT, GraphSAGE

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
        ['WikiCS',300,10],
        ['LastFM',128,18],
        ['AmazonComputers',767,10]
        # ['Reddit2',602,41],     
        # ['PPI']
        # 可以试一下PPI数据集
        # You can try the PPI dataset
    ]

    # 数据集
    # Dataset
    parser.add_argument('-data', "--dataset", type=str, default="PubMed")     # Cora,CiteSeer,PubMed,FacebookPagePage,CoraFull,WikiCS

    # 数据集的feature_size
    # Feature size of the dataset
    parser.add_argument('-fs','--feature_size',type=int,default = 0)     # 1433, 3703, 500, 128, 8710
    
    # 分类任务的label种类数
    # The number of label types for classification tasks
    parser.add_argument('-ncl', "--num_classes", type=int, default=0)       # 7, 6, 3, 4, 70

    # 测试集和训练集的划分比例,训练集所占的比例
    # The ratio of the test set to the training set, and the proportion of the training set
    parser.add_argument('-s','--split',type = float,default=0.8,
                        help = 'train/test dataset split percentage')

    # learning rate
    parser.add_argument('-lr', "--learning_rate", type=float, default=1,
                        help="learning rate for training")
    # 学习率是否要衰减
    # Should the learning rate be decayed?
    parser.add_argument('-lrd','--learning_rate_decay',type=bool,default=False)
    # 学习率衰减
    # Learning rate decay
    parser.add_argument('-lrdg','--learning_rate_decay_gamma',type=float,default=0.999)
    # num_clients
    parser.add_argument('-nc', "--num_clients", type=int, default=10,
                        help="Total number of clients")
    
    # device
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    

    # 经过多少轮进行一次测试
    # After how many rounds of testing
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")  
    
    # 客户端参加的比例(client drift程度)   
    # Client participation rate (client drift level)
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
      
    # 攻击客户端的id
    # Attacking client ID (Victim client)
    parser.add_argument('-vc','--victim_client',type=list,default=[0],
                        help='victim client in training')
    # 实施攻击的轮次v
    # The attack round v
    parser.add_argument('-agr','--attack_global_round',type=list,default=[i for i in range(90,100)])
    
    # 共实施多少次攻击
    # 每次实验攻击一个round
    # 攻击轮次固定，固定一个攻击轮次，50轮
# Total number of attacks
# Each attack is one round
# Fixed attack round number, 50 rounds
    parser.add_argument('-at','--attack_times',type=int,default=10)

    # O的缩放系数
    # Scaling factor of O
    parser.add_argument('-so', "--scale_O", type=float, default=1,
                        help="the scale fraction of O")
    
    # 是否执行攻击
    # Whether to execute the attack
    parser.add_argument('-a','--attack',type=bool,default=True,
                        help = 'attack or not')
    # 是否进行梯度匹配迭代更新dummy_x
    # Whether to perform gradient matching iterative update of dummy_x
    parser.add_argument('-gm','--grad_match',type=bool,default= False,
                        help='gradients matching or not')
    # 进行梯度匹配更新的迭代次数 30最好
    # The number of iterations for gradient matching update is 30, which is the best
    parser.add_argument('-gmi','--grad_match_iteration',type=int,default=30,
                        help='gradients matching iterations')
    # 进行梯度匹配更新的正则系数
    # Regularization coefficient for gradient matching update
    parser.add_argument('-gma','--grad_match_alpha',type=float,default=0,
                        help='gradients matching alpha')
    
    # 是否要进行模型参数裁剪
    # Whether to perform model parameter pruning
    parser.add_argument('-mpc','--model_params_clip',type=bool,default= False,
                        help='clip the params of model or not')
    
    # 是否要裁剪梯度 
    # Whether to clip gradients
    parser.add_argument('-mgc','--model_gradient_clip',type=bool,default= False,
                        help='clip client model gradient or not')
    
    # 衡量所恢复分布与真实分布的相似度
    # Measure the similarity between the restored distribution and the true distribution
    parser.add_argument('-mt','--metric',type=str,default='cosine_similarity',            # l2_norm,cosine_similarity,js_div,hellinger_dis
                        help='measure the similarity of the restored \
                            distribution to the true distribution')
    # True, False 
    # 是否进行主动攻击
    # Whether to conduct active attacks
    parser.add_argument('-aa','--active_attack',type=bool,default=True,
                        help='active attack or not')
    
    # # 是否进行对比
    # parser.add_argument('-comp','--compare',type=bool,default=False,
    #                     help = "compare with other method or not")
    
    # # 要对比算法的名字
    # parser.add_argument('-cn','--compare_name',type=str,default='ALDIA_no_active',                 # iLRG, LLG_star, ALDIA_no_active
    #                     help='select which methods to compare with')
    
    # 做五次实验取平均值，这是五次实验的随机数种子
    # Do five experiments and take the average value, which is the random number seed of the five experiments
    parser.add_argument('-rs','--random_seed',type = list,default = [0,1,2,3,4],
                        help = 'random seed for five experiments')
    # 当前是第几次实验
    # This is the first experiment
    parser.add_argument('-cen','--cur_experiment_no',type = int,default=0)
    # 是否做重复实验
    # Do repeated experiments?
    parser.add_argument('-re','--repeat_experiments',type = bool,default=True,
                        help = 'do repeat experiments or not')
    # 裁剪界限
    # Clipping limit
    parser.add_argument('-cn','--clip_norm',type = float,default=0.01)

    # 攻击时期
    # Attack period
    parser.add_argument('-ap','--attack_period',type = str,choices=['start','mid','end'],default = 'mid')
    # False, True

    # 是否进行防御
    # Whether to defend
    parser.add_argument('-de','--defense',type=bool,default= False)

    # DP lr
    parser.add_argument('-lrdp','--learning_rate_dp',type = float,default=1)
    # ALDA ：消融实验      
    # why_active ：为什么使用主动攻击——生成图片（LDA攻击效果和准确率随着global rounds变化）
    # cal_var_err :在模型的所以层乘以一个参数，观察攻击效果和方差变化和误差
    # compara : 对比试验(iLRG,LLG*)
    # defense :使用DP防御
    # --infi    :GNN_Infiltrator对比实验
    # label_DP
    # hops_var
    # task    ALDA,        why_active,cal_var_err
    parser.add_argument('-ta','--task',type = str,default='why_active')

    # 采样率
    # Sampling rate
    parser.add_argument('-sr','--sample_rate',type=float,default=1)

    # 隐私预算
    # Privacy Budget
    parser.add_argument('-eps','--epsilon',type=float,default=5)             

    # 计算方差任务中，模型参数乘以的变量值
    # The variable value by which the model parameters are multiplied in the variance calculation task
    parser.add_argument('-vm','--variable_mul',type = float,default=0.1)

    # node DP的节点度数限制
    # The node degree limit
    parser.add_argument('-DPdl','--DP_degree_limit',type = int,default=5)

    # label DP
    parser.add_argument('-ld','--defense_label_DP',type=bool,default=False)
    
    # 攻击的
    # Attack
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable,using cpu")
        args.device = "mps"    

    # 数据集  global_rounds  lr
    # Dataset global_rounds lr
    train_setting = [
        ['Cora',200,0.1],
        ['FacebookPagePage',200,0.1],
        ['CiteSeer',100,0.1],
        ['WikiCS',300,0.5],
        ['LastFM',200,0.1],
        ['PubMed',100,0.1],
        ['Computers',100,0.1],
        # ['LastFM',2,0.1],
        ['CoraFull',200,0.5],
        ['AmazonComputers', 100, 0.1]
    ]

    args.task='ALDA'
    print("HHHH",args)
    # args.task='defense'
    # ['Cora',1433,7],
    # ['CiteSeer',3703,6],
    # ['PubMed',500,3],
    # ['FacebookPagePage',128,4],
    # ['CoraFull',8710,70],
    # ['WikiCS',300,10]
    # ['LastFM',,]
    # 节点分类任务时，GraphSAGE对所有层都裁剪效果好一点
    # For node classification tasks, GraphSAGE performs better by pruning all layers.
    if args.task=='ALDA' or args.task=='defense' or args.task=='compare' or args.task=='hops_var' or args.task=='label_DP':
        args.algorithm = 'GIN'              # GCN, GAT, GraphSAGE
        args.gcn_hops = 2
        args.local_epochs = 5
        args.dataset = 'AmazonComputers'
        print("Yeta")
        # True, False
        if args.task == 'compare':
            args.compare = True
        else:
            args.compare = False
        
        if args.task == 'defense':
            args.defense = True
            args.local_epochs = 1
            if args.dataset == 'Cora':
                print("here")
                args.learning_rate_dp = 0.5 
                # args.sample_rate = 0.1
            elif args.dataset == 'CiteSeer':
                args.learning_rate_dp = 0.5
                # args.sample_rate = 0.2
            elif args.dataset == 'LastFM':
                args.learning_rate_dp = 0.5
            elif args.dataset == 'FacebookPagePage':
                args.learning_rate_dp = 0.5
            elif args.dataset == 'WikiCS':
                args.learning_rate_dp = 1
            elif args.dataset == 'CoraFull':
                args.learning_rate_dp = 1
            else:
                print('请输入正确的数据集！')
                exit()
        else:
            args.defense = False
        
        if args.task == 'label_DP':
            args.defense_label_DP=True

        args.num_clients = 10
        args.repeat_experiments = False
        # args.repeat_experiments = True
        # args.global_rounds = 200
        # args.learning_rate = 0.1
        args.clip_norm = 0.01
        args.attack_period = 'mid'

        flag = False
        for setting in train_setting:
            if args.dataset == setting[0]:
                args.global_rounds = setting[1]
                args.learning_rate = setting[2]
                flag = True
                break
        if not flag:
            print("No such Dataset!")
            exit()
        
        if args.attack_period == 'start':
            args.attack_global_round = [0]
        elif args.attack_period == 'mid':
            args.attack_global_round = [int(args.global_rounds/2)]
        elif args.attack_period == 'end':
            args.attack_global_round = [args.global_rounds-1]
        else:
            print('\nWrong attack_period!')
            exit()



    elif args.task == 'why_active':
        # args.defense = False
        args.algorithm = 'GraphSAGE'              # GCN, GAT, GraphSAGE
        args.gcn_hops = 2
        args.local_epochs = 5
        args.dataset = 'WikiCS'

        args.defense = False
        args.compare = False
        args.repeat_experiments = True
        args.global_rounds = 300
        # test
        # args.global_rounds = 3
        args.learning_rate = 0.5
        args.clip_norm = 0.01
        args.attack_global_round = [i for i in range(args.global_rounds)]
    
    elif args.task == 'cal_var_err':
        args.algorithm = 'GCN'              # GCN, GAT, GraphSAGE
        args.gcn_hops = 2
        args.local_epochs = 1
        # args.dataset = 'WikiCS'
        args.dataset = 'WikiCS'

        args.defense = False
        args.compare = False
        args.repeat_experiments = True
        args.global_rounds = 300
        # test
        # args.global_rounds = 3
        args.learning_rate = 0.5
        args.clip_norm = 0.01
        # args.attack_global_round = [i for i in range(args.global_rounds)]
        args.attack_global_round = [int(args.global_rounds/2)]
        # args.global_rounds = int(args.global_rounds/2) + 1
    # elif args.task == 'hops_var':
    #     args.algorithm = 'GCN'              # GCN, GAT, GraphSAGE
    #     args.gcn_hops = 1
    #     args.local_epochs = 5
    #     # args.dataset = 'WikiCS'
    #     args.dataset = 'LastFM'

    #     args.defense = False
    #     args.compare = False
    #     args.repeat_experiments = True
    #     flag = False
    #     for setting in train_setting:
    #         if args.dataset == setting[0]:
    #             args.global_rounds = setting[1]
    #             args.learning_rate = setting[2]
    #             flag = True
    #             break
    #     if not flag:
    #         print("No such Dataset!")
    #         exit()
    #     # args.clip_norm = 0.01
    #     # args.attack_global_round = [i for i in range(args.global_rounds)]
    #     args.attack_global_round = [int(args.global_rounds/2)]
    #     # args.global_rounds = int(args.global_rounds/2) + 1

    
    else:
        print('No such task!')
        exit()

    # if args.dataset == 'CoraFull':
    #     args.learning_rate = 0.5

    # if args.dataset == 'WikiCS':
    #     args.learning_rate = 0.5

    # output_file = 'why_active_WikiCS_300_2_'+str(args.local_epochs)+'_0.5.txt'

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
    # if args.defense:
    #     args.local_epochs = 1
    
    if len(args.attack_global_round)>1:
        print('仅允许攻击一次！')
        exit()

    avg_cos_sim = []
    avg_js_div = []
    final_acc = [] 
    five_all_acc = []
    five_all_cos_sim = []
    five_all_js_div = []
    five_all_emb_var = []
    five_all_err = []
    all_var = []
    all_err = []
    print("I",args)

    if args.repeat_experiments:
        for i in range(len(args.random_seed)):
            print("\nExperiment ",i,":----------------")
            args.cur_experiment_no = i

            cos_sim, js_div, acc, all_acc, all_cos_sim, all_js_div, var, emb_var, err = run(args,i)
            avg_cos_sim.append(cos_sim)
            avg_js_div.append(js_div)
            final_acc.append(acc)
            five_all_acc.append(all_acc)
            five_all_cos_sim.append(all_cos_sim)
            five_all_js_div.append(all_js_div)
            five_all_emb_var.append(emb_var)
            five_all_err.append(err)
            all_var.append(var)

    else:
        
        cos_sim, js_div, acc, all_acc, all_cos_sim, all_js_div, var, emb_var, err = run(args,0)
        avg_cos_sim.append(cos_sim)
        avg_js_div.append(js_div)
        final_acc.append(acc)
        five_all_acc.append(all_acc)
        five_all_cos_sim.append(all_cos_sim)
        five_all_js_div.append(all_js_div)
        five_all_emb_var.append(emb_var)
        five_all_err.append(err)
        all_var.append(var)

    


    if args.task == 'cal_var_err':
        print('all_var:',all_var)
        print('\navg var:{: .3f}'.format(sum(all_var)/len(all_var)))
        print('all_err:',five_all_err)
        print('\navg err:{: .3f}'.format(sum(five_all_err)/len(five_all_err)))
        

    if args.task == 'why_active':
        mean_acc,var_acc = compute_mean_var(five_all_acc)
        mean_cos_sim, var_cos_sim = compute_mean_var(five_all_cos_sim)
        mean_js_div, var_js_div = compute_mean_var(five_all_js_div)
        # 指定输出文件路径
        #                         数据集  globalround  gcn_layer local_epochs lr
        # output_file = 'why_active_WikiCS_300_2_5_0.5.txt'
        output_file = 'why_active_WikiCS_300_2_'+str(args.local_epochs)+'_0.5.txt'

        # 将四个列表写入文本文件
        with open(output_file, 'w') as f:
            f.write("mean_acc:\n")
            for item in mean_acc:
                f.write("%s\n" % item)
            f.write("\n var_acc:\n")
            for item in var_acc:
                f.write("%s\n" % item)
            f.write("\n mean_cos_sim:\n")
            for item in mean_cos_sim:
                f.write("%s\n" % item)
            f.write("\n var_cos_sim:\n")
            for item in var_cos_sim:
                f.write("%s\n" % item)
            f.write("\n mean_js_div:\n")
            for item in mean_js_div:
                f.write("%s\n" % item)
            f.write("\n var_js_div:\n")
            for item in var_js_div:
                f.write("%s\n" % item)

        acc_with_two_metric_mean_var(mean_acc,var_acc,mean_cos_sim, var_cos_sim,mean_js_div, var_js_div,args.dataset)

    print(f"\n----------------Done! Report:----------------")
    # print('Done! Report:')
    if not args.compare:
        print("avg_all_cos_sim:",avg_cos_sim)
        print("avg_all_js_div:",avg_js_div)
        print("all_final_acc:",final_acc)

        print('avg_cos_sim:',sum(avg_cos_sim)/len(avg_cos_sim))
        print('avg_JS_div:',sum(avg_js_div)/len(avg_js_div))
        print('avg_final_acc:',sum(final_acc)/len(final_acc))

        print('avg_cos_sim:{: .3f}'.format(sum(avg_cos_sim)/len(avg_cos_sim)))
        print('avg_JS_div:{: .3f}'.format(sum(avg_js_div)/len(avg_js_div)))
        print('avg_final_acc:{: .3f}'.format(sum(final_acc)/len(final_acc)))
    else:
        avg_cos_sim_no_active = [row[0] for row in avg_cos_sim]
        avg_cos_sim_iLRG = [row[1] for row in avg_cos_sim]
        avg_cos_sim_LLG_star = [row[2] for row in avg_cos_sim]

        avg_js_div_no_active = [row[0] for row in avg_js_div]
        avg_js_div_iLRG = [row[1] for row in avg_js_div]
        avg_js_div_LLG_star = [row[2] for row in avg_js_div]

        print('cos_sim')
        print('Infi:',avg_cos_sim_no_active)
        print('iLRG:',avg_cos_sim_iLRG)
        print('LLG_star:',avg_cos_sim_LLG_star)

        print('js_div')
        print('Infi:',avg_js_div_no_active)
        print('iLRG:',avg_js_div_iLRG)
        print('LLG_star:',avg_js_div_LLG_star)
        print("all_final_acc:",final_acc)

        print('avg_cos_sim')
        print('Infi:{: .3f}'.format(sum(avg_cos_sim_no_active)/len(avg_cos_sim_no_active)))
        print('iLRG:{: .3f}'.format(sum(avg_cos_sim_iLRG)/len(avg_cos_sim_iLRG)))
        print('LLG_star:{: .3f}'.format(sum(avg_cos_sim_LLG_star)/len(avg_cos_sim_LLG_star)))

        print('js_div')
        print('Infi:{: .3f}'.format(sum(avg_js_div_no_active)/len(avg_js_div_no_active)))
        print('iLRG:{: .3f}'.format(sum(avg_js_div_iLRG)/len(avg_js_div_iLRG)))
        print('LLG_star:{: .3f}'.format(sum(avg_js_div_LLG_star)/len(avg_js_div_LLG_star)))

        print('avg_final_acc:{: .3f}'.format(sum(final_acc)/len(final_acc)))
    
    if args.task == 'hops_var':
        print("emb_avg:{: .3f}".format(sum(five_all_emb_var)/len(five_all_emb_var)))


    print(args)
    print('DONE!')

    
