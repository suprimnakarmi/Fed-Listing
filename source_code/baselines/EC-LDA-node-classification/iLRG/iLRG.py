from iLRG.methods import *
from iLRG.utils import *

def batch_label_construction_iLRG(args,model,gradients,num_nodes):
    gradients = [gra.cpu()for gra in gradients]
    model = model.cpu()
    w_grad, b_grad = gradients[-2], gradients[-1]
    cls_rec_probs = []
    for i in range(args.num_classes):
        # Recover class-specific embeddings and probabilities
        cls_rec_emb = get_emb(w_grad[i], b_grad[i])
        # if (not args.silu) and (not args.leaky_relu):
        #     cls_rec_emb = torch.where(cls_rec_emb < 0., torch.full_like(cls_rec_emb, 0), cls_rec_emb)
        # cls_rec_emb = torch.where(w_grad[i] < 0., torch.full_like(w_grad[i], 0), w_grad[i])
        cls_rec_prob = post_process_emb(embedding=cls_rec_emb,
                                        model=model,
                                        device=args.device,
                                        alpha=1)
        cls_rec_probs.append(cls_rec_prob)

    res = get_irlg_res(cls_rec_probs=cls_rec_probs,
                                    b_grad=b_grad,
                                    num_classes=args.num_classes,
                                    num_nodes=num_nodes)

    return res