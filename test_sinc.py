# test_sinc.py

import pytest
import numpy as np
import mindspore
from mindspore import Tensor, nn, ops
from mindspore import dtype as mstype
from mindspore.mint import sinc
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

def test_sinc_error_input():
    """
    (1d) 测试随机混乱输入：字符串、None、整数等非张量
    """
    print_env_info()
    bad_inputs = ["this is a string", None, 123]
    for item in bad_inputs:
        try:
            _ = sinc(item)
        except Exception as e:
            print(f"sinc error with input={item}:", e)

def test_sinc_calculation_fixed_dtype():
    """
    (1b) 使用固定 dtype(float32)+固定输入，对比 MindSpore 与 PyTorch
    """
    print_env_info()
    arr = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    ms_res = sinc(Tensor(arr, mstype.float32))
    torch_res = torch.sinc(torch.tensor(arr, dtype=torch.float32))
    assert np.allclose(ms_res.asnumpy(), torch_res.detach().numpy(), atol=1e-3)

def test_sinc_calculation_random_dtype():
    """
    (1a) 随机输入，不同 dtype: float16, float32, float64
    """
    print_env_info()
    dmap = {
        mstype.float16: torch.float16,
        mstype.float32: torch.float32,
        mstype.float64: torch.float64
    }
    for ms_dt, torch_dt in dmap.items():
        arr = np.random.randn(5, 5).astype(mindspore.dtype_to_nptype(ms_dt))
        ms_out = sinc(Tensor(arr, ms_dt))
        torch_out = torch.sinc(torch.tensor(arr, dtype=torch_dt))
        assert np.allclose(ms_out.asnumpy(), torch_out.detach().numpy(), atol=1e-3)

def test_sinc_calculation_fixed_shape_diff_param():
    """
    (1c) 固定 shape 输入，传入额外无意义参数，检查是否报错
    """
    print_env_info()
    arr = np.array([0, 1], dtype=np.float32)
    try:
        _ = sinc(Tensor(arr), "extra_param")
    except Exception as e:
        print("MindSpore sinc extra param error:", e)

def test_sinc_calculation_broadcast():
    """
    扩展：测试 broadcasting，(4,1) vs (1,4)
    """
    print_env_info()
    a = np.random.randn(4,1).astype(np.float32)
    b = np.random.randn(1,4).astype(np.float32)
    ms_out_a = sinc(Tensor(a))
    ms_out_b = sinc(Tensor(b))
    torch_a = torch.sinc(torch.tensor(a))
    torch_b = torch.sinc(torch.tensor(b))
    # 这里只是演示 broadcasting 可以用在外层操作，若真需要 broadcasting x,y，可结合其他函数测试

def test_sinc_calculation_empty():
    """
    扩展：空张量 (0, )
    """
    print_env_info()
    arr = np.array([], dtype=np.float32)
    ms_out = sinc(Tensor(arr))
    torch_out = torch.sinc(torch.tensor(arr))
    # 对空张量可能都是空输出，也可做断言
    assert ms_out.shape == (0,)
    assert torch_out.shape == (0,)

def test_sinc_calculation_shape_mismatch():
    """
    扩展：形状不匹配时（如传入 2D + 3D）并想进行算子操作时可能报错
    """
    print_env_info()
    arr2d = np.random.randn(2, 3).astype(np.float32)
    arr3d = np.random.randn(2, 3, 4).astype(np.float32)
    # sinc 本身是一元函数，这里仅演示不匹配形状无关 => 可能不报错
    # 但若你要测试多输入函数，可以在此演示 shape mismatch

class SincNetMindspore(nn.Cell):
    """(2a,2b,2c) 在网络中使用 sinc"""
    def __init__(self):
        super(SincNetMindspore, self).__init__()
        self.act = sinc

    def construct(self, x):
        return self.act(x)

def test_sinc_nn_inference_compare_with_torch():
    """
    (2b) 在网络中调用 sinc，比较前向结果
    """
    print_env_info()
    net_ms = SincNetMindspore()
    data = np.random.randn(4, 3).astype(np.float32)
    ms_out = net_ms(Tensor(data)).asnumpy()
    torch_out = torch.sinc(torch.tensor(data, dtype=torch.float32)).detach().numpy()
    assert np.allclose(ms_out, torch_out, atol=1e-3)

def test_sinc_function_grad():
    """
    (2c) 测试 sinc 对输入的梯度
    """
    print_env_info()
    x_ms = Tensor(np.random.randn(4, 3).astype(np.float32))
    x_ms.requires_grad = True

    grad_op = ops.GradOperation(get_all=True)(sinc)
    grad_ms = grad_op(x_ms)[0].asnumpy()

    x_torch = torch.tensor(x_ms.asnumpy(), dtype=torch.float32, requires_grad=True)
    y_torch = torch.sinc(x_torch)
    y_torch.sum().backward()
    grad_torch = x_torch.grad.detach().numpy()

    assert np.allclose(grad_ms, grad_torch, atol=1e-3)
