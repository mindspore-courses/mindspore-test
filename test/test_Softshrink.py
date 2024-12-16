import pytest
import numpy as np
import torch
import mindspore as ms
from mindspore import mint, Tensor

input_data = [[1, 6, 2, 4], [7, 3, 8, 2], [2, 9, 11, 5]]
dtype_ms_list = [ms.int8, ms.int16, ms.int32, ms.int64, ms.uint8, ms.uint16, ms.uint32, ms.uint64, ms.float16,
                 ms.float32, ms.float64, ms.bfloat16, ms.bool_]
dtype_torch_list = [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.uint16, torch.uint32,
                    torch.uint64, torch.float16, torch.float32, torch.float64, torch.bfloat16, torch.bool]


def sum_softshrink_output(tensor, lambd=0.5):
    '''利用ms.ops.grad获取mindspore计算的梯度'''
    softshrink = mint.nn.Softshrink(lambd=lambd)
    return softshrink(tensor).sum()


@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_softshrink_different_dtypes(mode):
    """测试不同数据类型下，MindSpore和PyTorch的Softshrink支持度, 如果都支持，那计算结果的差异"""
    ms.set_context(mode=mode)

    for dtype_ms, dtype_torch in zip(dtype_ms_list, dtype_torch_list):
        ms_tensor = Tensor(input_data, dtype=dtype_ms)
        torch_tensor = torch.tensor(input_data, dtype=dtype_torch)

        err = False
        try:
            ms_result = mint.nn.Softshrink(lambd=0.5)(ms_tensor).asnumpy()
        except Exception as e:
            err = True
            print(f"MindSpore Softshrink not supported for {dtype_ms}")

        try:
            torch_result = torch.nn.functional.softshrink(torch_tensor, lambd=0.5).numpy()
        except Exception as e:
            err = True
            print(f"PyTorch Softshrink not supported for {dtype_torch}")

        if not err:
            assert np.allclose(ms_result, torch_result, atol=1e-3)


@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_softshrink_random_input_fixed_dtype(mode):
    """测试固定数据类型下，不同的随机输入维度下的Softshrink结果一致性"""
    ms.set_context(mode=mode)

    shapes = [[5], [5, 2], [5, 4, 3], [4, 6, 7, 8]]
    for shape in shapes:
        ms_tensor = Tensor(np.random.randn(*shape), dtype=ms.float32)
        torch_tensor = torch.tensor(ms_tensor.asnumpy(), dtype=torch.float32)

        ms_result = mint.nn.Softshrink(lambd=0.5)(ms_tensor).asnumpy()
        torch_result = torch.nn.functional.softshrink(torch_tensor, lambd=0.5).numpy()

        assert np.allclose(ms_result, torch_result, atol=1e-3)


@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_softshrink_different_params(mode):
    """测试MindSpore和PyTorch中Softshrink的不同lambd参数支持度"""
    ms.set_context(mode=mode)

    lambds = [0.1, 0.5, 1.0, 2.0]
    for lambd in lambds:
        # MindSpore Softshrink
        ms_tensor = Tensor(input_data, dtype=ms.float32)
        ms_result = mint.nn.Softshrink(lambd=lambd)(ms_tensor).asnumpy()

        # PyTorch Softshrink
        torch_tensor = torch.tensor(input_data, dtype=torch.float32)
        torch_result = torch.nn.functional.softshrink(torch_tensor, lambd=lambd).numpy()

        # 断言两个框架的结果一致
        assert np.allclose(ms_result, torch_result, atol=1e-3), f"Mismatch in results for lambd {lambd}"


@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_softshrink_wrong_input(mode):
    """测试无效输入时，MindSpore和PyTorch的错误处理"""
    ms.set_context(mode=mode)

    ms_tensor = Tensor(input_data, dtype=ms.float32)
    torch_tensor = torch.tensor(input_data, dtype=torch.float32)

    try:
        ms_result = mint.nn.Softshrink(lambd=0.5)(ms_tensor).asnumpy()  # 有效
    except Exception as e:
        print(f"MindSpore Softshrink error: {e}")

    try:
        torch_result = torch.nn.functional.softshrink(torch_tensor, lambd=0.5).numpy()  # 有效
    except Exception as e:
        print(f"PyTorch Softshrink error: {e}")

    try:
        ms_result = mint.nn.Softshrink(lambd=-10)(ms_tensor).asnumpy()  
    except Exception as e:
        print(f"MindSpore Softshrink invalid lambd error: {e}")

    try:
        torch_result = torch.nn.functional.softshrink(torch_tensor, lambd=-10).numpy()  
    except Exception as e:
        print(f"PyTorch Softshrink invalid lambd error: {e}")


@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_softshrink_mint_forward_back(mode):
    """使用Pytorch和MindSpore, 固定输入和权重, 测试正向推理结果和反向梯度"""
    ms.set_context(mode=mode)

    ms_tensor = Tensor(input_data, ms.float32)
    softshrink_ms = mint.nn.Softshrink(lambd=0.5)
    torch_tensor = torch.tensor(input_data, dtype=torch.float32, requires_grad=True)
    softshrink_torch = torch.nn.functional.softshrink(torch_tensor, lambd=0.5)

    output_ms = softshrink_ms(ms_tensor)
    output_torch = softshrink_torch

    gradient_function = ms.ops.grad(sum_softshrink_output)
    grad_ms = gradient_function(ms_tensor)

    output_torch.sum().backward()
    grad_torch = torch_tensor.grad

    assert np.allclose(output_ms.asnumpy(), output_torch.detach().numpy(),
                       atol=1e-3), "Forward outputs differ more than allowed tolerance"
    assert np.allclose(grad_ms.asnumpy(), grad_torch.numpy(), atol=1e-3), "Gradients differ more than allowed tolerance"
