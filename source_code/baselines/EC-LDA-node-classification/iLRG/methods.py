import torch
import numpy as np

# Recover embeddings
def get_emb(grad_w, grad_b, exp_thre=10):
    # Split scientific count notation
    sc_grad_b = '%e' % grad_b
    sc_grad_w = ['%e' % w for w in grad_w]
    real_b, exp_b = float(sc_grad_b.split('e')[0]), int(sc_grad_b.split('e')[1])
    real_w, exp_w = np.array([float(sc_w.split('e')[0]) for sc_w in sc_grad_w]), \
                    np.array([int(sc_w.split('e')[1]) for sc_w in sc_grad_w])
    # Deal with 0 case
    if real_b == 0.:
        real_b = 1
        exp_b = -64
    # Deal with exponent value
    exp = exp_w - exp_b
    exp = np.where(exp > exp_thre, exp_thre, exp)
    exp = np.where(exp < -1 * exp_thre, -1 * exp_thre, exp)

    def get_exp(x):
        return 10 ** x if x >= 0 else 1. / 10 ** (-x)

    exp = np.array(list(map(get_exp, exp)))
    # Calculate recovered average embeddings for batch_i (samples of class i)
    res = (1. / real_b) * real_w * exp
    res = torch.from_numpy(res).to(torch.float32)
    return res


# Recover Labels
def iLRG(probs, grad_b, n_classes, n_images):
    # Solve linear equations to recover labels
    coefs, values = [], []
    # Add the first equation: k1+k2+...+kc=K
    coefs.append([1 for _ in range(n_classes)])
    values.append(n_images)
    # Add the following equations
    for i in range(n_classes):
        coef = []
        for j in range(n_classes):
            if j != i:
                coef.append(probs[j][i].item())
            else:
                coef.append(probs[j][i].item() - 1)
        coefs.append(coef)
        values.append(n_images * grad_b[i])
    # Convert into numpy ndarray
    coefs = np.array(coefs)
    values = np.array(values)
    # Solve with Moore-Penrose pseudoinverse
    res_float = np.linalg.pinv(coefs).dot(values)
    # Filter negative values
    res = np.where(res_float > 0, res_float, 0)
    # Round values
    res = np.round(res).astype(int)
    res = np.where(res <= n_images, res, 0)
    err = res - res_float
    num_mod = np.sum(res) - n_images
    if num_mod > 0:
        inds = np.argsort(-err)
        mod_inds = inds[:num_mod]
        mod_res = res.copy()
        mod_res[mod_inds] -= 1
    elif num_mod < 0:
        inds = np.argsort(err)
        mod_inds = inds[:num_mod]
        mod_res = res.copy()
        mod_res[mod_inds] += 1
    else:
        mod_res = res

    return res, mod_res

