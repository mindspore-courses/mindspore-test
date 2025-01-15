# test_sinh.py

import pytest
import numpy as np
import mindspore
from mindspore import Tensor, nn, ops
from mindspore import dtype as mstype
from mindspore.mint import sinh
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

def test_sinh_error_input():
    """
    (1d) 混乱输入，如字符串, None
    """
    print_env_info()
    for err in ["this is a string", None]:
        try:
            _ = sinh(err)
        except Exception as e:
            print(f"sinh error with input={err}:", e)

def test_sinh_calculation_fixed_dtype():
    """
    (1b) 固定 dtype(float32)+固定输入
    """
    print_env_info()
    arr = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    ms_res = sinh(Tensor(arr, mstype.float32))
    torch_res = torch.sinh(torch.tensor(arr, dtype=torch.float32))
    assert np.allclose(ms_res.asnumpy(), torch_res.detach().numpy(), atol=1e-3)

def test_sinh_calculation_random_dtype():
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
        arr = np.random.randn(5, 5).astype(mindspore.dtype_to_nptype(ms_dt))
        ms_out = sinh(Tensor(arr, ms_dt))
        torch_out = torch.sinh(torch.tensor(arr, dtype=torch_dt))
        assert np.allclose(ms_out.asnumpy(), torch_out.detach().numpy(), atol=1e-3)

def test_sinh_calculation_fixed_shape_diff_param():
    """
    (1c) 固定输入 + 多余参数
    """
    print_env_info()
    arr = np.array([0, 1], dtype=np.float32)
    try:
        _ = sinh(Tensor(arr), "extra_param")
    except Exception as e:
        print("MindSpore sinh extra param error:", e)

def test_sinh_calculation_extreme_values():
    """
    扩展：极端值，比如非常大的正负数
    """
    print_env_info()
    arr = np.array([1e10, -1e10, 1e-10, -1e-10], dtype=np.float32)
    ms_res = sinh(Tensor(arr, mstype.float32))
    torch_res = torch.sinh(torch.tensor(arr, dtype=torch.float32))
    # 可能会出现 inf 或 -inf，对比时要稍微留意
    print("MindSpore:", ms_res.asnumpy())
    print("PyTorch:", torch_res.detach().numpy())

class SinhNetMindspore(nn.Cell):
    """(2a,2b,2c) 使用 sinh 组网"""
    def __init__(self):
        super(SinhNetMindspore, self).__init__()
        self.act = sinh

    def construct(self, x):
        return self.act(x)

def test_sinh_nn_inference_compare_with_torch():
    """
    (2b) 在网络中调用 sinh 比对前向推理
    """
    print_env_info()
    net_ms = SinhNetMindspore()
    data = np.random.randn(4, 3).astype(np.float32)
    ms_out = net_ms(Tensor(data)).asnumpy()
    torch_out = torch.sinh(torch.tensor(data, dtype=torch.float32)).detach().numpy()
    assert np.allclose(ms_out, torch_out, atol=1e-3)

def test_sinh_function_grad():
    """
    (2c) 测试 sinh 对输入的梯度
    """
    print_env_info()
    x_ms = Tensor(np.random.randn(4, 3).astype(np.float32))
    x_ms.requires_grad = True
    grad_op = ops.GradOperation(get_all=True)(sinh)
    grad_ms = grad_op(x_ms)[0].asnumpy()

    x_torch = torch.tensor(x_ms.asnumpy(), dtype=torch.float32, requires_grad=True)
    y_torch = torch.sinh(x_torch)
    y_torch.sum().backward()
    grad_torch = x_torch.grad.detach().numpy()
    assert np.allclose(grad_ms, grad_torch, atol=1e-3)
