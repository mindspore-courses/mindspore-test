# test_sin.py

import pytest
import numpy as np
import mindspore
from mindspore import Tensor, nn, ops
from mindspore import dtype as mstype
from mindspore.mint import sin
import torch

def print_env_info():
    """
    打印 MindSpore 与 PyTorch 的版本信息，方便后续排查。
    """
    print("===== Environment Info =====")
    try:
        print("MindSpore version:", mindspore.__version__)
    except:
        print("MindSpore version: Unknown")
    try:
        print("PyTorch version:", torch.__version__)
    except:
        print("PyTorch version: Unknown")
    print("============================\n")

def test_sin_error_input():
    """
    (1d) 测试字符串/None等异常输入
    """
    print_env_info()
    bad_inputs = ["this is a string", None, [1, 2, 3]]
    for inp in bad_inputs:
        try:
            _ = sin(inp)
        except Exception as e:
            print(f"Sin error with input={inp} =>", e)

def test_sin_calculation_fixed_dtype():
    """
    (1b) 使用固定 dtype(float32) + 固定输入，对比 MindSpore/PyTorch
    """
    print_env_info()
    arr = np.array([0.0, np.pi/6, np.pi/2, np.pi], dtype=np.float32)
    ms_out = sin(Tensor(arr, mstype.float32))
    torch_out = torch.sin(torch.tensor(arr, dtype=torch.float32))
    assert np.allclose(ms_out.asnumpy(), torch_out.detach().numpy(), atol=1e-3)

def test_sin_calculation_random_dtype():
    """
    (1a) 随机输入，不同 dtype(float16/32/64)
    """
    print_env_info()
    dmap = {
        mstype.float16: torch.float16,
        mstype.float32: torch.float32,
        mstype.float64: torch.float64
    }
    for ms_dt, torch_dt in dmap.items():
        arr = np.random.randn(5, 5).astype(mindspore.dtype_to_nptype(ms_dt))
        ms_res = sin(Tensor(arr, ms_dt))
        torch_res = torch.sin(torch.tensor(arr, dtype=torch_dt))
        assert np.allclose(ms_res.asnumpy(), torch_res.detach().numpy(), atol=1e-3)

def test_sin_calculation_fixed_shape_diff_param():
    """
    (1c) 固定输入 + 多余字符串参数
    """
    print_env_info()
    arr = np.array([0, 1], dtype=np.float32)
    try:
        _ = sin(Tensor(arr), "extra_param")
    except Exception as e:
        print("Sin extra param error:", e)

def test_sin_calculation_empty():
    print_env_info()
    arr = np.array([], dtype=np.float32)
    ms_res = sin(Tensor(arr))
    torch_res = torch.sin(torch.tensor(arr))
    assert ms_res.shape == (0,)
    assert torch_res.shape == (0,)

def test_sin_calculation_extreme_values():
    """
    扩展：极端值(非常大/非常小)
    """
    print_env_info()
    arr = np.array([1e10, -1e10, np.pi*2], dtype=np.float32)
    ms_res = sin(Tensor(arr))
    torch_res = torch.sin(torch.tensor(arr))
    print("MindSpore:", ms_res.asnumpy())
    print("PyTorch:", torch_res.detach().numpy())


class SinNetMindspore(nn.Cell):
    """
    (2a,2b,2c) 在网络中使用 sin
    """
    def __init__(self):
        super(SinNetMindspore, self).__init__()
        self.act = sin

    def construct(self, x):
        return self.act(x)

def test_sin_nn_inference_compare_with_torch():
    """
    (2b) 比较网络前向推理
    """
    print_env_info()
    net_ms = SinNetMindspore()
    data = np.random.randn(4, 3).astype(np.float32)
    ms_out = net_ms(Tensor(data)).asnumpy()

    torch_out = torch.sin(torch.tensor(data, dtype=torch.float32)).detach().numpy()
    assert np.allclose(ms_out, torch_out, atol=1e-3)

def test_sin_function_grad():
    """
    (2c) 测试 sin 对输入的梯度
    """
    print_env_info()
    arr = np.random.randn(4, 3).astype(np.float32)
    x_ms = Tensor(arr)
    x_ms.requires_grad = True

    grad_op = ops.GradOperation(get_all=True)(sin)
    grad_ms = grad_op(x_ms)[0].asnumpy()

    x_torch = torch.tensor(arr, dtype=torch.float32, requires_grad=True)
    y_torch = torch.sin(x_torch)
    y_torch.sum().backward()
    grad_torch = x_torch.grad.detach().numpy()

    assert np.allclose(grad_ms, grad_torch, atol=1e-3)
