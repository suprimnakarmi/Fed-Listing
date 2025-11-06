import torch

def R_clip_dummy_x(dummy_x):
    # clip
    clip_dummy_x = torch.clamp(dummy_x, min=0, max=1)
    diff = dummy_x-clip_dummy_x
    return torch.norm(diff)

def R_scale_dummy_x(dummy_x):
    min_val = torch.min(dummy_x)
    max_val = torch.max(dummy_x)
    return torch.norm(dummy_x-(dummy_x - min_val)/(max_val - min_val))

    # max_values, _ = torch.max(dummy_x, dim=1)
    # min_values, _ = torch.min(dummy_x, dim=1)

    # # 对每一行的每个元素执行操作
    # normalized_tensor = (dummy_x - min_values[:, None]) / (max_values - min_values)[:, None]
    # return torch.norm(dummy_x-normalized_tensor)

def R_clip_dummy_y(dummy_x):
    # clip
    clip_dummy_x = torch.clamp(dummy_x, min=0, max=1)
    diff = dummy_x-clip_dummy_x
    return torch.norm(diff)

def R_scale_dummy_y(dummy_x):
    min_val = torch.min(dummy_x)
    max_val = torch.max(dummy_x)
    return torch.norm(dummy_x-(dummy_x - min_val)/(max_val - min_val))

    
    # # 找出每一行的最大值和最小值
    # max_values, _ = torch.max(dummy_x, dim=1)
    # min_values, _ = torch.min(dummy_x, dim=1)

    # # 对每一行的每个元素执行操作
    # normalized_tensor = (dummy_x - min_values[:, None]) / (max_values - min_values)[:, None]
    # return torch.norm(dummy_x-normalized_tensor)

def R_positive_x(dummy_x):
    pass

def R_sum_equal_one(dummy_y):
    sum_sample = torch.sum(dummy_y, dim=1)
    ones_tensor = torch.ones_like(sum_sample)
    return (sum_sample-ones_tensor).abs().mean()

def R_centralized(dummy_y):
    row_variances = torch.var(dummy_y, dim=1)
    return torch.exp(-row_variances).mean()
# def R_tv(dummy_x):


# TODO,设计正则项