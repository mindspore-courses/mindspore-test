import pytest
import numpy as np
import torch
import mindspore as ms
from mindspore import mint, Tensor


input_data = [[[1, 6, 2, 4], [7, 3, 8, 2], [2, 9, 11, 5]],
              [[4, 3, 1, 8], [9, 4, 3, 6], [1, 2, 7, 9]]]
dtype_ms_list = [ms.int8, ms.int16, ms.int32, ms.int64, ms.uint8, ms.uint16, ms.uint32, ms.uint64, ms.float16,
                 ms.float32, ms.float64, ms.bfloat16, ms.bool_]
dtype_torch_list = [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.uint16, torch.uint32,
                    torch.uint64, torch.float16, torch.float32, torch.float64, torch.bfloat16, torch.bool]



@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_avgpool2d_different_dtypes(mode):
    """测试不同数据类型下，MindSpore和PyTorch的AvgPool2d支持度, 如果都支持，那计算结果的差异"""
    ms.set_context(mode=mode)

    for dtype_ms, dtype_torch in zip(dtype_ms_list, dtype_torch_list):
        ms_tensor = Tensor(input_data, dtype=dtype_ms)
        torch_tensor = torch.tensor(input_data, dtype=dtype_torch)

        err = False
        try:
            ms_avgpool = mint.nn.AvgPool2d(kernel_size=2, stride=2)
            ms_result = ms_avgpool(ms_tensor).asnumpy()
        except Exception as e:
            err = True
            print(f"MindSpore AvgPool2d not supported for {dtype_ms}: {e}")

        try:
            torch_avgpool = torch.nn.AvgPool2d(kernel_size=2, stride=2)
            torch_result = torch_avgpool(torch_tensor).numpy()
        except Exception as e:
            err = True
            print(f"PyTorch AvgPool2d not supported for {dtype_torch}: {e}")

        if not err:
            assert np.allclose(ms_result, torch_result, atol=1e-3), f"Mismatch for dtype {dtype_ms} and {dtype_torch}"


@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_avgpool2d_random_input_fixed_dtype(mode):#GRAPH_MODE problems
    """测试固定数据类型下，不同的随机输入维度下的AvgPool2d结果一致性"""
    ms.set_context(mode=mode)

    shapes = [[5, 4, 4], [5, 4, 6, 7], [4, 5, 6, 3]]
    for shape in shapes:
        ms_tensor = Tensor(np.random.randn(*shape), dtype=ms.float32)
        torch_tensor = torch.tensor(ms_tensor.asnumpy(), dtype=torch.float32)

        ms_avgpool = mint.nn.AvgPool2d(kernel_size=2, stride=2)
        torch_avgpool = torch.nn.AvgPool2d(2, stride=2)

        ms_result = ms_avgpool(ms_tensor).asnumpy()
        torch_result = torch_avgpool(torch_tensor).numpy()

        assert np.allclose(ms_result, torch_result, atol=1e-3), f"Mismatch for shape {shape}"


@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_avgpool2d_different_params(mode):#GRAPH_MODE problems
    """测试MindSpore和PyTorch中AvgPool2d的不同kernel_size, stride, padding等参数支持度"""
    ms.set_context(mode=mode)

    params = [(2, 2, 0), (3, 3, 1), (2, 1, 1)]
    for kernel_size, stride, padding in params:
        ms_tensor = Tensor(input_data, dtype=ms.float32)
        torch_tensor = torch.tensor(input_data, dtype=torch.float32)

        ms_avgpool = mint.nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        torch_avgpool = torch.nn.AvgPool2d(kernel_size, stride, padding=padding)

        ms_result = ms_avgpool(ms_tensor).asnumpy()
        torch_result = torch_avgpool(torch_tensor).numpy()

        assert np.allclose(ms_result, torch_result, atol=1e-3), f"Mismatch for kernel_size={kernel_size}, stride={stride}, padding={padding}"


@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_avgpool2d_wrong_input(mode):
    """测试无效输入时，MindSpore和PyTorch的错误处理"""
    ms.set_context(mode=mode)

    ms_tensor = Tensor(input_data, dtype=ms.float32)
    torch_tensor = torch.tensor(input_data, dtype=torch.float32)

    try:
        ms_avgpool = mint.nn.AvgPool2d(kernel_size=10, stride=2)
        ms_result = ms_avgpool(ms_tensor).asnumpy()  # 错误的kernel_size
    except Exception as e:
        print(f"MindSpore AvgPool2d error: {e}")

    try:
        torch_avgpool = torch.nn.AvgPool2d(kernel_size=10, stride=2)
        torch_result = torch_avgpool(torch_tensor)  # 错误的kernel_size
    except Exception as e:
        print(f"PyTorch AvgPool2d error: {e}")

    try:
        ms_avgpool = mint.nn.AvgPool2d(kernel_size=2, stride=3)
        ms_result = ms_avgpool(ms_tensor)  # 错误的stride
    except Exception as e:
        print(f"MindSpore AvgPool2d stride error: {e}")

    try:
        torch_avgpool = torch.nn.AvgPool2d(kernel_size=2, stride=3)
        torch_result = torch_avgpool(torch_tensor)  # 错误的stride
    except Exception as e:
        print(f"PyTorch AvgPool2d stride error: {e}")

def sum_avgpool2d_output(tensor, kernel_size, stride, padding):
    '''利用ms.ops.grad获取mindspore计算的梯度'''
    avgpool = mint.nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
    return avgpool(tensor).sum()


@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_avgpool2d_mint_forward_back(mode):#GRAPH_MODE problems
    """使用Pytorch和MindSpore, 固定输入和权重, 测试正向推理结果和反向梯度"""
    ms.set_context(mode=mode)

    # MindSpore setup
    ms_tensor = Tensor(input_data, ms.float32)
    ms_avgpool = mint.nn.AvgPool2d(kernel_size=2, stride=2)

    # Forward pass
    ms_result = ms_avgpool(ms_tensor)

    # Using ms.ops.grad to compute the gradients
    gradient_function = ms.ops.grad(sum_avgpool2d_output)
    grad_ms = gradient_function(ms_tensor, 2, 2, 0)


    torch_result = np.array([[[4.25, 4.]], [[5., 4.5]]], dtype=np.float32)
    grad_torch = np.array([[[0.25, 0.25, 0.25, 0.25],
                              [0.25, 0.25, 0.25, 0.25],
                              [0., 0., 0., 0.]],

                             [[0.25, 0.25, 0.25, 0.25],
                              [0.25, 0.25, 0.25, 0.25],
                              [0., 0., 0., 0.]]], dtype=np.float32)
    # Check forward pass results
    assert np.allclose(ms_result.asnumpy(), torch_result,
                       atol=1e-3), "Forward outputs differ more than allowed tolerance"

    # Check gradients
    assert np.allclose(grad_ms.asnumpy(), grad_torch, atol=1e-3), "Gradients differ more than allowed tolerance"
