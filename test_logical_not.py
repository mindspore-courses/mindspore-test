# test_logical_not.py

import pytest
import numpy as np
import mindspore
from mindspore import Tensor, nn, ops
from mindspore import dtype as mstype
from mindspore.mint import logical_not
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

def test_logical_not_error_input():
    """
    (1d) 字符串、None等异常输入
    """
    print_env_info()
    for item in ["a string", None]:
        try:
            _ = logical_not(item)
        except Exception as e:
            print("logical_not error with input:", e)

def test_logical_not_calculation_fixed_dtype():
    """
    (1b) 固定 bool 输入，对比
    """
    print_env_info()
    arr = np.array([[True, False], [False, True]], dtype=bool)
    ms_out = logical_not(Tensor(arr, mstype.bool_))
    torch_out = torch.logical_not(torch.tensor(arr))
    assert np.all(ms_out.asnumpy() == torch_out.detach().cpu().numpy())

def test_logical_not_calculation_random_shape():
    """
    (1a/1b) 随机 bool 输入，对比
    """
    print_env_info()
    arr = np.random.choice([True, False], size=(3, 3))
    ms_out = logical_not(Tensor(arr, mstype.bool_))
    torch_out = torch.logical_not(torch.tensor(arr))
    assert np.all(ms_out.asnumpy() == torch_out.detach().cpu().numpy())

def test_logical_not_calculation_fixed_shape_diff_param():
    """
    (1c) 多余参数
    """
    print_env_info()
    arr = np.array([True, False], dtype=bool)
    try:
        _ = logical_not(Tensor(arr), "extra_param")
    except Exception as e:
        print("logical_not extra param error:", e)

def test_logical_not_calculation_empty():
    """
    扩展：空张量
    """
    print_env_info()
    arr = np.array([], dtype=bool)
    ms_out = logical_not(Tensor(arr))
    torch_out = torch.logical_not(torch.tensor(arr))
    assert ms_out.shape == (0,)
    assert torch_out.shape == (0,)

class LogicalNotNetMindspore(nn.Cell):
    """(2a,2b,2c) 网络中使用 logical_not"""
    def __init__(self):
        super(LogicalNotNetMindspore, self).__init__()
        self.op = logical_not

    def construct(self, x):
        return self.op(x)

def test_logical_not_nn_inference_compare_with_torch():
    """
    (2b) 比较网络前向
    """
    print_env_info()
    net_ms = LogicalNotNetMindspore()
    arr = np.random.choice([True, False], size=(2, 2))
    ms_out = net_ms(Tensor(arr, mstype.bool_)).asnumpy()
    torch_out = torch.logical_not(torch.tensor(arr)).detach().cpu().numpy()
    assert np.all(ms_out == torch_out)

def test_logical_not_function_grad():
    """
    (2c) 对 bool 运算做反向意义不大，这里演示强行梯度
    """
    print_env_info()
    float_arr = Tensor(np.random.rand(2, 2).astype(np.float32))
    float_arr.requires_grad = True

    def forward_fn(x):
        return logical_not(x > 0.5).astype(mstype.float32).sum()

    grad_op = ops.GradOperation(get_all=True)(forward_fn)
    try:
        grad_res = grad_op(float_arr)
        print("logical_not grad:", grad_res[0].asnumpy())
    except Exception as e:
        print("logical_not grad error:", e)
