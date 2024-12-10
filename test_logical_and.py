# test_logical_and.py

import pytest
import numpy as np
import mindspore
from mindspore import Tensor, nn, ops
from mindspore import dtype as mstype
from mindspore.mint import logical_and
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

def test_logical_and_error_input():
    """
    (1d) 测试字符串、None等输入
    """
    print_env_info()
    for item in ["a", None]:
        try:
            _ = logical_and(item, item)
        except Exception as e:
            print("logical_and error input:", e)

def test_logical_and_calculation_fixed_dtype():
    """
    (1b) 固定 bool 输入，对比 MindSpore 与 PyTorch
    """
    print_env_info()
    a_np = np.array([[True, False], [True, True]])
    b_np = np.array([[False, False], [True, False]])
    ms_out = logical_and(Tensor(a_np, mstype.bool_), Tensor(b_np, mstype.bool_))
    torch_out = torch.logical_and(torch.tensor(a_np), torch.tensor(b_np))
    assert np.all(ms_out.asnumpy() == torch_out.detach().cpu().numpy())

def test_logical_and_calculation_random_shape():
    """
    (1a/1b) 随机 bool 张量，对比
    """
    print_env_info()
    a_np = np.random.choice([True, False], size=(4, 4))
    b_np = np.random.choice([True, False], size=(4, 4))
    ms_out = logical_and(Tensor(a_np), Tensor(b_np))
    torch_out = torch.logical_and(torch.tensor(a_np), torch.tensor(b_np))
    assert np.all(ms_out.asnumpy() == torch_out.detach().cpu().numpy())

def test_logical_and_calculation_fixed_shape_diff_param():
    """
    (1c) 多余参数
    """
    print_env_info()
    a_np = np.array([True, False], dtype=bool)
    b_np = np.array([False, True], dtype=bool)
    try:
        _ = logical_and(Tensor(a_np), Tensor(b_np), "extra_param")
    except Exception as e:
        print("MindSpore logical_and extra param error:", e)

def test_logical_and_calculation_broadcast():
    """
    扩展：广播形状 (2,1) vs (1,2)
    """
    print_env_info()
    a_np = np.random.choice([True, False], size=(2,1))
    b_np = np.random.choice([True, False], size=(1,2))
    ms_out = logical_and(Tensor(a_np), Tensor(b_np))
    torch_out = torch.logical_and(torch.tensor(a_np), torch.tensor(b_np))
    assert ms_out.shape == (2,2)
    assert np.all(ms_out.asnumpy() == torch_out.detach().cpu().numpy())

class LogicalAndNetMindspore(nn.Cell):
    """(2a,2b,2c) 在网络中使用 logical_and"""
    def __init__(self):
        super(LogicalAndNetMindspore, self).__init__()
        self.op = logical_and

    def construct(self, x, y):
        return self.op(x, y)

def test_logical_and_nn_inference_compare_with_torch():
    """
    (2b) 网络中逻辑与，对比前向
    """
    print_env_info()
    net_ms = LogicalAndNetMindspore()
    a_np = np.array([[True, False], [True, True]])
    b_np = np.array([[True, True], [False, True]])
    ms_out = net_ms(Tensor(a_np), Tensor(b_np)).asnumpy()
    torch_out = torch.logical_and(torch.tensor(a_np), torch.tensor(b_np)).detach().cpu().numpy()
    assert np.all(ms_out == torch_out)

def test_logical_and_function_grad():
    """
    (2c) 对 bool 运算做反向意义不大，这里演示强行梯度
    """
    print_env_info()
    a_float = Tensor(np.random.rand(4, 4).astype(np.float32))
    b_float = Tensor(np.random.rand(4, 4).astype(np.float32))
    a_float.requires_grad = True
    b_float.requires_grad = True

    def forward_fn(x, y):
        return logical_and(x > 0.5, y > 0.5).astype(mstype.float32).sum()

    grad_op = ops.GradOperation(get_all=True)(forward_fn)
    try:
        grad_res = grad_op(a_float, b_float)
        print("logical_and grad:", [g.asnumpy() for g in grad_res])
    except Exception as e:
        print("logical_and grad error:", e)
