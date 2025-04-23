import numpy as np
import mindspore
from mindspore import Tensor, dtype as mstype, ops
import mindspore.mint.nn.functional as F_ms
import torch
import torch.nn.functional as F_torch

def test_leaky_relu_random_dtype_support():
    """
    (1a) 测试random输入不同dtype
    """
    print("===== LeakyReLU random dtype support test =====")
    for dt in [mstype.float16, mstype.float32, mstype.int32]:
        x_np = np.random.randn(3,3)
        if dt == mstype.int32:
            x_np = x_np.astype(np.int32)
        else:
            x_np = x_np.astype(mindspore.dtype_to_nptype(dt))
        try:
            out_ms = F_ms.leaky_relu(Tensor(x_np, dt), negative_slope=0.1)
            print(f"MindSpore dtype={dt}, shape={out_ms.shape}")
        except Exception as e:
            print("MindSpore error:", e)

        torch_dt = torch.float16 if dt == mstype.float16 else torch.float32
        x_torch = torch.tensor(x_np, dtype=torch_dt)
        try:
            out_pt = F_torch.leaky_relu(x_torch, negative_slope=0.1)
            print(f"PyTorch dtype={torch_dt}, shape={out_pt.shape}")
        except Exception as e:
            print("PyTorch error:", e)
        print("------------------------------------")


def test_leaky_relu_fixed_dtype_output_equality():
    """
    (1b) 固定dtype=float32 随机输入，对比输出
    """
    print("===== LeakyReLU fixed dtype output equality test =====")
    x_np = np.random.randn(4,4).astype(np.float32)
    ms_in = Tensor(x_np, mstype.float32)
    torch_in = torch.tensor(x_np, dtype=torch.float32)

    out_ms = F_ms.leaky_relu(ms_in, 0.1).asnumpy()
    out_pt = F_torch.leaky_relu(torch_in, 0.1).numpy()

    diff = np.abs(out_ms - out_pt).max()
    print("Max diff:", diff)
    assert diff < 1e-3


def test_leaky_relu_fixed_shape_diff_params():
    """
    (1c) 测试 negative_slope 参数不同类型
    """
    print("===== LeakyReLU fixed shape diff params test =====")
    x = Tensor(np.array([-2., -1., 0., 1., 2.], np.float32))

    out_slope_default = F_ms.leaky_relu(x)          # default=0.01
    out_slope_float   = F_ms.leaky_relu(x, 0.2)
    out_slope_int     = F_ms.leaky_relu(x, 1)
    out_slope_bool    = F_ms.leaky_relu(x, True)  # True => 1

    print("default=0.01:", out_slope_default.asnumpy())
    print("0.2:", out_slope_float.asnumpy())
    print("1:", out_slope_int.asnumpy())
    print("True(=1):", out_slope_bool.asnumpy())

    # string slope => error
    try:
        F_ms.leaky_relu(x, negative_slope="0.1")
    except Exception as e:
        print("slope=string error:", e)


def test_leaky_relu_error_messages():
    """
    (1d) 测试随机无效输入
    """
    print("===== LeakyReLU error messages test =====")
    # 输入非Tensor
    try:
        F_ms.leaky_relu([-1,0,1], negative_slope=0.1)
    except Exception as e:
        print("non-tensor input error:", e)


def test_leaky_relu_network_forward_backward():
    """
    (2b,2c) 使用LeakyReLU验证前向输出 & 反向梯度
    """
    print("===== LeakyReLU forward/backward test =====")

    # PyTorch
    x_pt = torch.tensor([-1.,0.,1.], requires_grad=True)
    out_pt = F_torch.leaky_relu(x_pt, 0.2)
    out_pt.sum().backward()
    grad_pt = x_pt.grad.numpy()

    # MindSpore
    x_ms = Tensor(np.array([-1.,0.,1.], np.float32))
    x_ms.requires_grad = True
    def forward_fn(inp):
        return F_ms.leaky_relu(inp, 0.2).sum()
    grad_fn = ops.grad(forward_fn, grad_position=0)
    grad_ms = grad_fn(x_ms).asnumpy()

    print("PyTorch grad:", grad_pt)
    print("MindSpore grad:", grad_ms)