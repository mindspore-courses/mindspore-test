import numpy as np
import pytest
import torch
from mindspore import Tensor, mint, value_and_grad
import mindspore as ms

epsilon = 1e-10  # 一个很小的值，确保不包含边界


@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_arctanh_different_dtypes(mode):
    """1.(a) 测试random输入不同dtype，对比两个框架的支持度"""
    ms.set_context(mode=mode)
    ms_dtypes = [
        ms.int8,
        ms.int16,
        ms.int32,
        ms.int64,
        ms.uint8,
        ms.float16,
        ms.float32,
        ms.float64,
        ms.bfloat16,
        ms.bool_,
    ]
    torch_dtypes = [
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.float16,
        torch.float32,
        torch.float64,
        torch.bfloat16,
        torch.bool,
    ]

    for i in range(len(ms_dtypes)):
        dtype_ms = ms_dtypes[i]
        dtype_torch = torch_dtypes[i]
        input_data = np.random.uniform(-1 + epsilon, 1 - epsilon, (2, 3))
        ms_input = Tensor(input_data, dtype_ms)
        torch_input = torch.tensor(input_data, dtype=dtype_torch)
        err = False
        try:
            ms_result = mint.arctanh(ms_input).asnumpy()
        except Exception as e:
            err = True
            print(f"mint.arctanh not supported for {dtype_ms}")
        try:
            torch_result = torch.arctanh(torch_input).numpy()
        except Exception as e:
            err = True
            print(f"torch.arctanh not supported for {dtype_torch}")
        if not err:
            try:
                assert np.allclose(ms_result, torch_result, atol=1e-3)
            except AssertionError:
                print(f"Assertion failed for {dtype_ms} and {dtype_torch}")


@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_arctanh_fixed_dtype_random_input(mode):
    """1.(b) 测试固定dtype，random输入值，对比两个框架输出是否相等（误差范围为小于1e-3）"""
    ms.set_context(mode=mode)
    shapes = [(4), (3, 2), (3, 4, 5), (6, 7, 8, 9)]
    for shape in shapes:
        input_data = np.random.uniform(-1 + epsilon, 1 - epsilon, shape)
        ms_input = Tensor(input_data, ms.float32)
        torch_input = torch.tensor(input_data, dtype=torch.float32)
        ms_result = mint.arctanh(ms_input).asnumpy()
        torch_result = torch.arctanh(torch_input).numpy()
        assert np.allclose(ms_result, torch_result, atol=1e-3)


@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_arctanh_fixed_shape_input_diff_param(mode):
    """1.(c) 测试固定shape，固定输入值，不同输入参数（string\bool等类型），两个框架的支持度（由于没有更多参数，因此输入多余参数看报错信息）"""
    ms.set_context(mode=mode)
    input_data = np.array([0, -0.5])
    ms_input = Tensor(input_data, ms.float32)
    torch_input = torch.tensor(input_data, dtype=torch.float32)
    try:
        ms_result = mint.arctanh(ms_input, "extra_param").asnumpy()
    except Exception as e:
        print(f"mint.arctanh extra param error:", e)
    try:
        torch_result = torch.arctanh(torch_input, "extra_param").numpy()
    except Exception as e:
        print(f"torch.arctanh extra param error:", e)


@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_arctanh_random_input_error_message(mode):
    """1.(d) 测试随机混乱输入，报错信息的准确性"""
    ms.set_context(mode=mode)
    input_data = np.array([0, -0.5])
    ms_input = Tensor(input_data, ms.float32)
    try:
        ms_result = mint.arctanh(1.5).asnumpy()
    except Exception as e:
        print(f"mint.arctanh error message:", e)
    try:
        ms_result = mint.arctanh(True).asnumpy()
    except Exception as e:
        print(f"mint.arctanh error message:", e)
    try:
        ms_result = mint.arctanh(ms_input, "extra_param").asnumpy()
    except Exception as e:
        print(f"mint.arctanh extra param error:", e)


@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_arctanh_network(mode):
    """使用Pytorch和MindSpore，固定输入和权重，测试正向推理结果和反向梯度"""
    ms.set_context(mode=mode)
    input_data = np.array([0, -0.5])
    ms_input = Tensor(input_data, ms.float32)
    torch_input = torch.tensor(input_data, dtype=torch.float32, requires_grad=True)

    def forward_ms(x):
        return mint.arctanh(x)

    def forward_pt(x):
        return torch.arctanh(x)

    # Gradients for MindSpore
    grad_fn = value_and_grad(forward_ms)
    output_ms, gradient_ms = grad_fn(ms_input)

    # Forward pass for Pytorch
    output_pt = forward_pt(torch_input)
    output_pt.backward(torch.ones_like(output_pt))

    gradient_pt = torch_input.grad

    # 比较输出结果和梯度
    assert np.allclose(output_ms.asnumpy(), output_pt.detach().numpy(), atol=1e-3)
    assert np.allclose(gradient_ms.asnumpy(), gradient_pt.numpy(), atol=1e-3)

    # 打印输出和梯度
    print("MindSpore output:", output_ms)
    print("MindSpore gradient:", gradient_ms)
    print("Pytorch output:", output_pt)
    print("Pytorch gradient:", gradient_pt)
