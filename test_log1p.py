# test_log1p.py

import pytest
import numpy as np
import mindspore
from mindspore import Tensor, nn, ops
from mindspore import dtype as mstype
from mindspore.mint import log1p
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

def test_log1p_error_input():
    """
    (1d) 测试字符串/None等异常输入
    """
    print_env_info()
    for item in ["bad", None]:
        try:
            _ = log1p(item)
        except Exception as e:
            print(f"log1p error with input={item}:", e)

def test_log1p_calculation_fixed_dtype():
    """
    (1b) 固定 dtype(float32)+固定输入
    """
    print_env_info()
    arr = np.array([0.0, 0.5, 1.0, 2.0], dtype=np.float32)
    ms_out = log1p(Tensor(arr, mstype.float32))
    torch_out = torch.log1p(torch.tensor(arr, dtype=torch.float32))
    assert np.allclose(ms_out.asnumpy(), torch_out.detach().numpy(), atol=1e-4)

def test_log1p_calculation_random_dtype():
    """
    (1a) 随机输入 不同 dtype，保证1+x>0
    """
    print_env_info()
    dmap = {
        mstype.float16: torch.float16,
        mstype.float32: torch.float32,
        mstype.float64: torch.float64
    }
    for ms_dt, torch_dt in dmap.items():
        arr = np.random.uniform(-0.99, 5.0, size=(5, 5)).astype(mindspore.dtype_to_nptype(ms_dt))
        ms_out = log1p(Tensor(arr, ms_dt))
        torch_out = torch.log1p(torch.tensor(arr, dtype=torch_dt))
        assert np.allclose(ms_out.asnumpy(), torch_out.detach().numpy(), atol=1e-3)

def test_log1p_calculation_fixed_shape_diff_param():
    """
    (1c) 固定输入 + 多余参数
    """
    print_env_info()
    arr = np.array([0.1, 0.2], dtype=np.float32)
    try:
        _ = log1p(Tensor(arr), "extra_param")
    except Exception as e:
        print("MindSpore log1p extra param error:", e)

def test_log1p_calculation_edge_values():
    """
    扩展： 测试非常靠近 -1 的值，以及非常大的正数
    """
    print_env_info()
    arr = np.array([-0.9999, -0.5, 1e10, 1.0], dtype=np.float32)
    ms_out = log1p(Tensor(arr, mstype.float32))
    torch_out = torch.log1p(torch.tensor(arr, dtype=torch.float32))
    print("MindSpore:", ms_out.asnumpy())
    print("PyTorch:", torch_out.detach().numpy())

class Log1pNetMindspore(nn.Cell):
    """(2a,2b,2c) 在网络中使用 log1p"""
    def __init__(self):
        super(Log1pNetMindspore, self).__init__()
        self.op = log1p

    def construct(self, x):
        return self.op(x)

def test_log1p_nn_inference_compare_with_torch():
    """
    (2b) 比较网络前向推理
    """
    print_env_info()
    net_ms = Log1pNetMindspore()
    data = np.random.uniform(-0.9, 2.0, size=(3, 2)).astype(np.float32)
    ms_out = net_ms(Tensor(data)).asnumpy()
    torch_out = torch.log1p(torch.tensor(data, dtype=torch.float32)).detach().numpy()
    assert np.allclose(ms_out, torch_out, atol=1e-3)

def test_log1p_function_grad():
    """
    (2c) 测试 log1p 对输入的梯度
    """
    print_env_info()
    x_ms = Tensor(np.random.uniform(-0.9, 2.0, size=(2, 2)).astype(np.float32))
    x_ms.requires_grad = True

    grad_op = ops.GradOperation(get_all=True)(log1p)
    grad_ms = grad_op(x_ms)[0].asnumpy()

    x_torch = torch.tensor(x_ms.asnumpy(), dtype=torch.float32, requires_grad=True)
    y_torch = torch.log1p(x_torch)
    y_torch.sum().backward()
    grad_torch = x_torch.grad.detach().numpy()
    assert np.allclose(grad_ms, grad_torch, atol=1e-3)
