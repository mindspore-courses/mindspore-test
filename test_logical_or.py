# test_logical_or.py

import pytest
import numpy as np
import mindspore
from mindspore import Tensor, nn, ops
from mindspore import dtype as mstype
from mindspore.mint import logical_or
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

def test_logical_or_error_input():
    """
    (1d) 字符串等异常输入
    """
    print_env_info()
    for item in ["a", None]:
        try:
            _ = logical_or(item, item)
        except Exception as e:
            print("logical_or error input:", e)

def test_logical_or_calculation_fixed_dtype():
    """
    (1b) 固定 bool 输入，对比 MindSpore/PyTorch
    """
    print_env_info()
    a_np = np.array([True, False, False])
    b_np = np.array([False, False, True])
    ms_out = logical_or(Tensor(a_np), Tensor(b_np))
    torch_out = torch.logical_or(torch.tensor(a_np), torch.tensor(b_np))
    assert np.all(ms_out.asnumpy() == torch_out.detach().cpu().numpy())

def test_logical_or_calculation_random_shape():
    """
    (1a/1b) 随机 bool 输入，对比
    """
    print_env_info()
    a_np = np.random.choice([True, False], size=(4, 4))
    b_np = np.random.choice([True, False], size=(4, 4))
    ms_out = logical_or(Tensor(a_np), Tensor(b_np))
    torch_out = torch.logical_or(torch.tensor(a_np), torch.tensor(b_np))
    assert np.all(ms_out.asnumpy() == torch_out.detach().cpu().numpy())

def test_logical_or_calculation_fixed_shape_diff_param():
    """
    (1c) 多余参数
    """
    print_env_info()
    a_np = np.array([True, False], dtype=bool)
    b_np = np.array([False, True], dtype=bool)
    try:
        _ = logical_or(Tensor(a_np), Tensor(b_np), "extra_param")
    except Exception as e:
        print("logical_or extra param error:", e)

def test_logical_or_calculation_broadcast():
    """
    扩展：广播形状 (2,1) vs (1,2)
    """
    print_env_info()
    a_np = np.random.choice([True, False], size=(2,1))
    b_np = np.random.choice([True, False], size=(1,2))
    ms_out = logical_or(Tensor(a_np), Tensor(b_np))
    torch_out = torch.logical_or(torch.tensor(a_np), torch.tensor(b_np))
    assert ms_out.shape == (2,2)
    assert np.all(ms_out.asnumpy() == torch_out.detach().cpu().numpy())

class LogicalOrNetMindspore(nn.Cell):
    """(2a,2b,2c) 在网络中使用 logical_or"""
    def __init__(self):
        super(LogicalOrNetMindspore, self).__init__()
        self.op = logical_or

    def construct(self, x, y):
        return self.op(x, y)

def test_logical_or_nn_inference_compare_with_torch():
    """
    (2b) 比较网络中 logical_or 的前向
    """
    print_env_info()
    net_ms = LogicalOrNetMindspore()
    a_np = np.array([[True, False], [False, False]])
    b_np = np.array([[False, True], [False, True]])
    ms_out = net_ms(Tensor(a_np), Tensor(b_np)).asnumpy()
    torch_out = torch.logical_or(torch.tensor(a_np), torch.tensor(b_np)).detach().cpu().numpy()
    assert np.all(ms_out == torch_out)

def test_logical_or_function_grad():
    """
    (2c) 对 bool 运算做反向意义不大，这里演示强行梯度
    """
    print_env_info()
    a_float = Tensor(np.random.rand(2, 2).astype(np.float32))
    b_float = Tensor(np.random.rand(2, 2).astype(np.float32))
    a_float.requires_grad = True
    b_float.requires_grad = True

    def forward_fn(x, y):
        return logical_or(x > 0.5, y > 0.5).astype(mstype.float32).sum()

    grad_op = ops.GradOperation(get_all=True)(forward_fn)
    try:
        grad_res = grad_op(a_float, b_float)
        for i, g in enumerate(grad_res):
            print(f"Grad[{i}]: {g.asnumpy()}")
    except Exception as e:
        print("logical_or grad error:", e)
