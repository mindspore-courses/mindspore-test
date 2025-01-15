# test_sign.py

import pytest
import numpy as np
import mindspore
from mindspore import Tensor, nn, ops
from mindspore import dtype as mstype
from mindspore.mint import sign
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

def test_sign_error_input():
    """
    (1d) 测试随机混乱输入，如字符串, tuple, list, dict
    """
    print_env_info()
    bad_inputs = [
        "this is a string",
        (1, 0),
        [1, 2, 3],
        {"a": 1},
        None
    ]
    for val in bad_inputs:
        try:
            _ = sign(val)
        except Exception as e:
            print(f"Sign error with input={val} =>", e)


##############################
# 2. 计算测试
##############################
def test_sign_calculation_fixed_dtype():
    """
    (1b) 固定 dtype(float32) + 固定输入，对比 MindSpore 与 PyTorch sign
    """
    print_env_info()
    arr = np.array([-2.0, 0.0, 3.0, 5.5], dtype=np.float32)
    ms_res = sign(Tensor(arr, mstype.float32))
    torch_res = torch.sign(torch.tensor(arr, dtype=torch.float32))
    assert np.allclose(ms_res.asnumpy(), torch_res.detach().numpy(), atol=1e-3)

def test_sign_calculation_random_dtype():
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
        arr = (np.random.randn(5, 5)*10).astype(mindspore.dtype_to_nptype(ms_dt))
        ms_res = sign(Tensor(arr, ms_dt))
        torch_res = torch.sign(torch.tensor(arr, dtype=torch_dt))
        assert np.allclose(ms_res.asnumpy(), torch_res.detach().numpy(), atol=1e-3)

def test_sign_calculation_fixed_shape_diff_param():
    """
    (1c) 固定输入 + 无意义的额外参数
    """
    print_env_info()
    arr = np.array([[-1, 0], [1, 2]], dtype=np.float32)
    try:
        _ = sign(Tensor(arr), "extra_param")
    except Exception as e:
        print("Sign extra param error:", e)

def test_sign_calculation_empty():
    print_env_info()
    arr = np.array([], dtype=np.float32)
    ms_res = sign(Tensor(arr))
    torch_res = torch.sign(torch.tensor(arr))
    assert ms_res.shape == (0,)
    assert torch_res.shape == (0,)

def test_sign_calculation_extreme_values():
    """
    极端值(正负极大数)
    """
    print_env_info()
    arr = np.array([-1e10, 0, 1e10], dtype=np.float32)
    ms_res = sign(Tensor(arr))
    torch_res = torch.sign(torch.tensor(arr))
    print("MindSpore:", ms_res.asnumpy())
    print("PyTorch:", torch_res.detach().numpy())

class SignNetMindspore(nn.Cell):
    """
    (2a, 2b, 2c) 在网络中使用 sign
    """
    def __init__(self):
        super(SignNetMindspore, self).__init__()
        self.act = sign

    def construct(self, x):
        return self.act(x)

class SignModule(torch.nn.Module):
    def forward(self, x):
        return torch.sign(x)

def test_sign_nn_inference_compare_with_torch():
    """
    (2b) 在网络中调用 sign, 对比前向结果
    """
    print_env_info()
    net_ms = SignNetMindspore()
    data = np.array([[-3, 0], [4, 5]], dtype=np.float32)
    ms_out = net_ms(Tensor(data)).asnumpy()

    net_torch = SignModule()
    torch_out = net_torch(torch.tensor(data)).detach().numpy()
    assert np.allclose(ms_out, torch_out, atol=1e-3)

def test_sign_function_grad():
    """
    (2c) 测试 sign 对输入的梯度，通常为0或不支持
    """
    print_env_info()
    x_arr = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    x_ms = Tensor(x_arr)
    x_ms.requires_grad = True

    grad_op = ops.GradOperation(get_all=True)(sign)
    try:
        grad_ms = grad_op(x_ms)[0].asnumpy()
        print("MindSpore sign grad:", grad_ms)
    except Exception as e:
        print("MindSpore sign grad error:", e)

    x_torch = torch.tensor(x_arr, dtype=torch.float32, requires_grad=True)
    y_torch = torch.sign(x_torch)
    try:
        y_torch.sum().backward()
        print("PyTorch sign grad:", x_torch.grad.detach().numpy())
    except Exception as e:
        print("PyTorch sign grad error:", e)
