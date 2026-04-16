import torch
from .default import encode

@torch.no_grad()
def normalized_int_coords(coords:torch.FloatTensor, depth:int)->torch.LongTensor:
    assert coords.dim() == 3 and coords.shape[2] == 3, f"输入坐标张量 `coords` 形状错误 {coords.shape}。"
    min_vals, _ = torch.min(coords, dim=1, keepdim=True)  # (B, 1, 3)
    max_vals, _ = torch.max(coords, dim=1, keepdim=True)  # (B, 1, 3)
    scale = max_vals - min_vals
    scale[scale == 0] = 1.0 # 防止除以零，如果一个维度上的所有点都相同
    norm_coords = (coords - min_vals) / scale # 将坐标归一化到 [0, 1] 区间
    max_coord_val = 2**depth - 1 # 根据阶数 p 计算整数坐标的最大值
    int_coords = (norm_coords * max_coord_val).long() # 将归一化坐标转换为整数坐标
    return int_coords

@torch.no_grad()
def get_morton_code(coords:torch.Tensor, depth=15):
    int_coords = normalized_int_coords(coords, depth=depth)
    code = encode(int_coords.contiguous(), depth=depth, order="z")
    return code
