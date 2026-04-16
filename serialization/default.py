import torch
from .z_order import xyz2key as z_order_encode_


@torch.inference_mode()
def encode(grid_coord, depth=16, order="z"):
    assert order in {"z", "z-trans", "hilbert", "hilbert-trans"}
    if order == "z":
        code = z_order_encode(grid_coord, depth=depth)
    else:
        raise NotImplementedError
    return code

def z_order_encode(grid_coord: torch.Tensor, depth: int = 16):
    x, y, z = grid_coord[:, :, 0].long(), grid_coord[:, :, 1].long(), grid_coord[:, :, 2].long()
    # we block the support to batch, maintain batched code in Point class
    code = z_order_encode_(x, y, z, depth=depth)
    return code

