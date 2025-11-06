from iLRG.methods import *

def post_process_emb(embedding, model, device, alpha=0.01):
    # embedding = embedding.to(device)
    # Feed embedding into FC-Layer to get probabilities.
    out = model.fc2(embedding) * alpha
    prob = torch.softmax(out, dim=-1)
    return prob

def get_irlg_res(cls_rec_probs, b_grad, num_classes, num_nodes):
    # labels, existences, num_instances, num_instances_nonzero = get_label_stats(gt_label, num_classes)
    # Recovered Labels
    rec_instances, mod_rec_instances = iLRG(
        cls_rec_probs,
        b_grad,
        num_classes,
        num_nodes)
    sum_rec = rec_instances.sum()
    for i in range(len(rec_instances)):
        if rec_instances[i] == -0.0:
            rec_instances[i] = 0.0

    min_value = rec_instances.min()
    if min_value<0:
        rec_instances-=min_value

    if sum_rec == 0:
        rec_instances = [0 for i in range(len(rec_instances))]
    else:
        rec_instances = [i/sum_rec for i in rec_instances]
        
    return rec_instances


