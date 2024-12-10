# test_logaddexp.py

import pytest
import numpy as np
import mindspore
from mindspore import Tensor, nn, ops
from mindspore import dtype as mstype
from mindspore.mint import logaddexp
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

def test_logaddexp_error_input():
    """
    (1d) 字符串等异常输入
    """
    print_env_info()
    for item in ["a", None]:
        try:
            _ = logaddexp(item, item)
        except Exception as e:
            print("logaddexp error with input:", e)

def test_logaddexp_calculation_fixed_dtype():
    """
    (1b) 固定 dtype(float32)+固定输入
    """
    print_env_info()
    a_arr = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    b_arr = np.array([2.0, 1.0, 0.0], dtype=np.float32)
    ms_out = logaddexp(Tensor(a_arr), Tensor(b_arr))
    torch_out = torch.logaddexp(torch.tensor(a_arr), torch.tensor(b_arr))
    assert np.allclose(ms_out.asnumpy(), torch_out.detach().numpy(), atol=1e-4)

def test_logaddexp_calculation_random_dtype():
    """
    (1a) 随机输入，不同 dtype
    """
    print_env_info()
    dmap = {
        mstype.float16: torch.float16,
        mstype.float32: torch.float32,
        mstype.float64: torch.float64
    }
    for ms_dt, torch_dt in dmap.items():
        a_arr = np.random.randn(4, 4).astype(mindspore.dtype_to_nptype(ms_dt))
        b_arr = np.random.randn(4, 4).astype(mindspore.dtype_to_nptype(ms_dt))
        ms_out = logaddexp(Tensor(a_arr, ms_dt), Tensor(b_arr, ms_dt))
        torch_out = torch.logaddexp(torch.tensor(a_arr, dtype=torch_dt),
                                    torch.tensor(b_arr, dtype=torch_dt))
        assert np.allclose(ms_out.asnumpy(), torch_out.detach().numpy(), atol=1e-3)

def test_logaddexp_calculation_fixed_shape_diff_param():
    """
    (1c) 固定输入 + 多余参数
    """
    print_env_info()
    a_arr = np.array([1.0, 2.0], dtype=np.float32)
    b_arr = np.array([2.0, 1.0], dtype=np.float32)
    try:
        _ = logaddexp(Tensor(a_arr), Tensor(b_arr), "extra_param")
    except Exception as e:
        print("MindSpore logaddexp extra param error:", e)

def test_logaddexp_calculation_broadcast():
    """
    扩展：广播形状
    """
    print_env_info()
    a_arr = np.random.randn(2, 1).astype(np.float32)
    b_arr = np.random.randn(1, 2).astype(np.float32)
    ms_out = logaddexp(Tensor(a_arr), Tensor(b_arr))
    torch_out = torch.logaddexp(torch.tensor(a_arr), torch.tensor(b_arr))
    assert np.allclose(ms_out.asnumpy(), torch_out.detach().numpy(), atol=1e-3)

class LogAddExpNetMindspore(nn.Cell):
    """(2a,2b,2c) 在网络中使用 logaddexp"""
    def __init__(self):
        super(LogAddExpNetMindspore, self).__init__()
        self.func = logaddexp

    def construct(self, x, y):
        return self.func(x, y)

def test_logaddexp_nn_inference_compare_with_torch():
    """
    (2b) 比较网络前向推理
    """
    print_env_info()
    net_ms = LogAddExpNetMindspore()
    a_arr = np.random.randn(2, 2).astype(np.float32)
    b_arr = np.random.randn(2, 2).astype(np.float32)
    ms_out = net_ms(Tensor(a_arr), Tensor(b_arr)).asnumpy()
    torch_out = torch.logaddexp(torch.tensor(a_arr), torch.tensor(b_arr)).detach().numpy()
    assert np.allclose(ms_out, torch_out, atol=1e-3)

def test_logaddexp_function_grad():
    """
    (2c) 测试 logaddexp 对输入的梯度
    """
    print_env_info()
    a = Tensor(np.random.randn(2, 2).astype(np.float32))
    b = Tensor(np.random.randn(2, 2).astype(np.float32))
    a.requires_grad = True
    b.requires_grad = True

    def forward_fn(x, y):
        return logaddexp(x, y)

    grad_op = ops.GradOperation(get_all=True)(forward_fn)
    grad_ms = grad_op(a, b)
    ms_a = grad_ms[0].asnumpy()
    ms_b = grad_ms[1].asnumpy()

    a_torch = torch.tensor(a.asnumpy(), dtype=torch.float32, requires_grad=True)
    b_torch = torch.tensor(b.asnumpy(), dtype=torch.float32, requires_grad=True)
    y_torch = torch.logaddexp(a_torch, b_torch).sum()
    y_torch.backward()
    torch_a = a_torch.grad.detach().numpy()
    torch_b = b_torch.grad.detach().numpy()

    assert np.allclose(ms_a, torch_a, atol=1e-3)
    assert np.allclose(ms_b, torch_b, atol=1e-3)
