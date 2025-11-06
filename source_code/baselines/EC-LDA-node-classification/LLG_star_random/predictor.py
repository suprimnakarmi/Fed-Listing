import torch
import numpy as np
from torch_geometric.utils import erdos_renyi_graph
import copy

def get_gradient_dummy_random(args,model,feature_size,dummy_node_size=100,dummy_link_p=0.004):
    dummy_x = torch.normal(mean=0.0,std=0.001,size=(dummy_node_size,feature_size)).to(args.device).requires_grad_(True)
    dummy_y = torch.normal(mean=0.0,std=0.001,size = (dummy_node_size,args.num_classes)).to(args.device).requires_grad_(True)
    edge_index = erdos_renyi_graph(dummy_node_size,dummy_link_p)
    edge_index = edge_index.to(args.device)

    local_iterations=1
    gradients = []
    seperated_gradients=[]
    loss_f = torch.nn.CrossEntropyLoss()
    for i in range(local_iterations):
        model.to(args.device)
        orig_out,emb = model(dummy_x,edge_index)
        loss = loss_f(orig_out,dummy_y)
        grad = torch.autograd.grad(loss,model.parameters())

        seperated_gradients.append(list((_.detach().clone() for _ in grad)))
    
    # Copy the structure of a grad, but make it zeroes
    aggregated = list(x.zero_() for x in grad)

    # iterate over the gradients for each local iteration
    for grad in seperated_gradients:
        # there iterate through the gradients and add to the aggregator
        for i_g,g in enumerate(grad):
            aggregated[i_g] = torch.add(aggregated[i_g], g)
    
    gradient = list(torch.div(x, 1) for x in aggregated)

    return gradient




def LLG_star_random_prediction(args,gradients,model,batch_node_size):
    gradients_for_prediction = torch.sum(gradients[-2], dim=-1).clone()

    h1_extraction = []

    # do h1 extraction
    for i_cg, class_gradient in enumerate(gradients_for_prediction):
        if class_gradient < 0:
            h1_extraction.append((i_cg, class_gradient))
        
    impact = 0
    acc_impact = 0
    acc_offset = np.zeros(args.num_classes)
    n = 10

    for _ in range(n):
        tmp_gradients = []
        impact = 0
        for i in range(args.num_classes):
            gradients_dummy = get_gradient_dummy_random(args,copy.deepcopy(model),args.feature_size)
            tmp_gradients = torch.sum(gradients_dummy[-2], dim=-1).cpu().detach().numpy()
            impact += torch.sum(gradients_dummy[-2], dim=-1)[i].item()
            for j in range(args.num_classes):
                if j == i:
                    continue
                else:
                    acc_offset[j] +=tmp_gradients[j]

            impact /= (args.num_classes * batch_node_size)
            acc_impact += impact
    impact = (acc_impact / n) * (1 + 1/args.num_classes) / args.local_epochs
    
    acc_offset = np.divide(acc_offset, n*(args.num_classes-1))
    offset = torch.Tensor(acc_offset).to(args.device)

    gradients_for_prediction -= offset

    prediction=[]

    for (i_c, _) in h1_extraction:
        prediction.append(i_c)
        gradients_for_prediction[i_c] = gradients_for_prediction[i_c].add(-impact)
    
    # predict the rest
    for _ in range(batch_node_size * args.local_epochs - len(prediction)):
        # add minimal candidat, likely to be doubled, to prediction
        min_id = torch.argmin(gradients_for_prediction).item()
        prediction.append(min_id)

        # add the mean value of one accurance to the candidate
        gradients_for_prediction[min_id] = gradients_for_prediction[min_id].add(-impact)
    
    return prediction