import numpy as np
import mindspore
from mindspore import Tensor, dtype as mstype, ops
import mindspore.mint as mint
import torch

def test_cummax_random_dtype_support():
    """
    (1a) 不同 dtype
    """
    print("===== cummax random dtype support test =====")
    for dt in [mstype.float16, mstype.float32, mstype.int32]:
        x_np = np.random.randint(-5,5,size=(3,3)) if dt==mstype.int32 else np.random.randn(3,3)
        x_np = x_np.astype(mindspore.dtype_to_nptype(dt) if dt!=mstype.int32 else np.int32)
        print(f"MindSpore dtype={dt}")
        try:
            val_ms, idx_ms = mint.cummax(Tensor(x_np, dt), dim=1)
            print("  values shape:", val_ms.shape, "indices shape:", idx_ms.shape)
        except Exception as e:
            print("  MindSpore error:", e)

        # PyTorch
        torch_dt = torch.float16 if dt==mstype.float16 else torch.float32
        if dt==mstype.int32:
            torch_dt = torch.int32
        x_torch = torch.tensor(x_np, dtype=torch_dt)
        print(f"PyTorch dtype={torch_dt}")
        try:
            val_pt, idx_pt = torch.cummax(x_torch, dim=1)
            print("  values shape:", val_pt.shape, "indices shape:", idx_pt.shape)
        except Exception as e:
            print("  PyTorch error:", e)
        print("---------------------------------------")


def test_cummax_fixed_dtype_output_equality():
    """
    (1b) 固定dtype=float32, 随机输入, 比较输出
    """
    print("===== cummax fixed dtype output equality test =====")
    x_np = np.random.randint(-5,5,size=(3,3)).astype(np.float32)
    val_ms, idx_ms = mint.cummax(Tensor(x_np), dim=0)
    val_pt, idx_pt = torch.cummax(torch.tensor(x_np), dim=0)

    val_diff = np.abs(val_ms.asnumpy() - val_pt.numpy()).max()
    idx_diff = np.abs(idx_ms.asnumpy() - idx_pt.numpy()).max()
    print("val diff:", val_diff, "idx diff:", idx_diff)
    assert val_diff < 1e-3
    assert idx_diff < 1e-3


def test_cummax_fixed_shape_diff_params():
    """
    (1c) 测试 dim=-1, / dim 超范围
    """
    print("===== cummax fixed shape diff params test =====")
    x = Tensor(np.array([[3,3,1]], np.float32))
    val_ms, idx_ms = mint.cummax(x, dim=-1)
    print("dim=-1 => val:", val_ms.asnumpy(), "idx:", idx_ms.asnumpy())

    # 超范围
    try:
        mint.cummax(x, dim=2)
    except Exception as e:
        print("dim out of range:", e)


def test_cummax_error_messages():
    """
    (1d) 测试随机混乱输入
    """
    print("===== cummax error messages test =====")
    # 输入非tensor
    try:
        mint.cummax([1,2,3], dim=0)
    except Exception as e:
        print("non-tensor error:", e)

    # 输入维度0
    try:
        empty = Tensor(np.empty((0,3)), mstype.float32)
        val, idx = mint.cummax(empty, dim=0)
        print("Empty shape result => val:", val.shape, "idx:", idx.shape)
    except Exception as e:
        print("empty input error:", e)

    # dim非int
    try:
        mint.cummax(Tensor(np.array([1,2,3], np.float32)), dim="0")
    except Exception as e:
        print("dim not int error:", e)


def test_cummax_network_forward_backward():
    """
    (2b,2c) 测试cummax的反向梯度
    """
    print("===== cummax forward/backward test =====")

    # PyTorch
    x_pt = torch.tensor([5.,3.], requires_grad=True)
    val_pt, idx_pt = torch.cummax(x_pt, dim=0)
    loss_pt = val_pt.sum()
    loss_pt.backward()
    grad_pt = x_pt.grad.numpy()
    print("PyTorch val:", val_pt.detach().numpy(), "grad:", grad_pt)

    # MindSpore
    x_ms = Tensor(np.array([5.,3.], np.float32))
    x_ms.requires_grad = True
    def forward_fn(inp):
        val, idx = mint.cummax(inp, 0)
        return val.sum()
    grad_fn = ops.grad(forward_fn, grad_position=0)
    grad_ms = grad_fn(x_ms).asnumpy()
    print("MindSpore grad:", grad_ms)
