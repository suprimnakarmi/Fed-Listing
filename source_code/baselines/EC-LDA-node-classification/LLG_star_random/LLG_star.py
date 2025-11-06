from LLG_star_random.predictor import *
import copy

def batch_label_construction_LLG_star(args,model,gradients,num_nodes):
    prediction = LLG_star_random_prediction(args,gradients,copy.deepcopy(model),num_nodes)

    prediction.sort()
    dis = [0 for i in range(args.num_classes)]
    for i in prediction:
        dis[i]+=1
    sum_nodes = sum(dis)
    dis = [i/sum_nodes for i in dis]
    return dis     

    