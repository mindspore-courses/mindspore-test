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


def sum_linear_output(tensor, weight, bias=None):
    '''利用ms.ops.grad获取mindspore计算的梯度'''
    linear = mint.nn.Linear(in_features=tensor.shape[1], out_features=weight.shape[0])
    return linear(tensor).sum()


@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_linear_different_dtypes(mode):
    """测试不同数据类型下，MindSpore和PyTorch的Linear支持度, 如果都支持，那计算结果的差异"""
    ms.set_context(mode=mode)

    for dtype_ms, dtype_torch in zip(dtype_ms_list, dtype_torch_list):
        ms_tensor = Tensor(input_data, dtype=dtype_ms)
        torch_tensor = torch.tensor(input_data, dtype=dtype_torch)

        err = False
        try:
            ms_linear = mint.nn.Linear(in_features=4, out_features=3)  # 4输入，3输出
            ms_result = ms_linear(ms_tensor).asnumpy()
        except Exception as e:
            err = True
            print(f"MindSpore Linear not supported for {dtype_ms}: {e}")

        try:
            torch_linear = torch.nn.Linear(4, 3)
            torch_result = torch_linear(torch_tensor).detach().numpy()
        except Exception as e:
            err = True
            print(f"PyTorch Linear not supported for {dtype_torch}: {e}")
        #不在这里测试，固定w和b在test_linear_mint_forward_back中测试
        #if not err:
        #    assert np.allclose(ms_result, torch_result, atol=1e-3), f"Mismatch for dtype {dtype_ms} and {dtype_torch}"


@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_linear_mint_forward_back(mode):
    """使用Pytorch和MindSpore, 固定输入和权重, 测试正向推理结果和反向梯度"""
    ms.set_context(mode=mode)

    # MindSpore setup
    input_data = np.array([[1, 6, 2, 4], [7, 3, 8, 2], [2, 9, 11, 5]], dtype=np.float32)  # shape: (3, 4)

    # 固定权重和偏置
    weight_init = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]],
                           dtype=np.float32)  # 权重: (3, 4)
    bias_init = np.array([0.1, 0.2, 0.3], dtype=np.float32)  

    ms_tensor = Tensor(input_data, ms.float32)
    ms_linear = mint.nn.Linear(in_features=4, out_features=3,
                               weight_init=Tensor(weight_init), bias_init=Tensor(bias_init))
    ms_result = ms_linear(ms_tensor)
    print(ms_result)

    gradient_function = ms.ops.grad(ms_linear)
    grad_ms = gradient_function(ms_tensor)

    torch_result_target = np.array([[3.6, 8.900001, 14.2],
                                    [4.6000004, 12.700001, 20.800001],
                                    [7.4, 18.300001, 29.2]], dtype=np.float32)

    grad_torch_target = np.array([[1.5, 1.8, 2.1, 2.4],
                                  [1.5, 1.8, 2.1, 2.4],
                                  [1.5, 1.8, 2.1, 2.4]], dtype=np.float32)

    # Check forward pass results
    assert np.allclose(ms_result.asnumpy(), torch_result_target,
                       atol=1e-3), "Forward outputs differ more than allowed tolerance"

    # Check gradients
    assert np.allclose(grad_ms.asnumpy(), grad_torch_target, atol=1e-3), "Gradients differ more than allowed tolerance"


@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_linear_random_input_fixed_dtype(mode):
    """测试固定数据类型下，不同的随机输入维度下的Linear结果一致性"""
    ms.set_context(mode=mode)

    shapes = [[5, 4], [5, 4, 2], [5, 4, 3, 2]]

    for shape in shapes:
        ms_tensor = Tensor(np.random.randn(*shape), dtype=ms.float32)
        torch_tensor = torch.tensor(ms_tensor.asnumpy(), dtype=torch.float32)
        weight = np.random.randn(shape[-1], 3).astype(np.float32)
        bias = np.random.randn(3).astype(np.float32) 
        ms_linear = mint.nn.Linear(in_features=shape[-1], out_features=3, weight_init=Tensor(weight),
                                   bias_init=Tensor(bias))
        torch_linear = torch.nn.Linear(shape[-1], 3)

        with torch.no_grad():
            torch_linear.weight.copy_(torch.tensor(weight))
            torch_linear.bias.copy_(torch.tensor(bias))


        ms_result = ms_linear(ms_tensor).asnumpy()
        torch_result = torch_linear(torch_tensor).detach().numpy()
        assert np.allclose(ms_result, torch_result, atol=1e-3), f"Mismatch for shape {shape}"


@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_linear_different_params(mode):
    """测试MindSpore和PyTorch中Linear层的不同in_features和out_features参数支持度"""
    ms.set_context(mode=mode)

    params = [(4, 3), (5, 2), (6, 10)]
    for in_features, out_features in params:
        ms_tensor = Tensor(input_data, dtype=ms.float32)
        torch_tensor = torch.tensor(input_data, dtype=torch.float32)

        ms_linear = mint.nn.Linear(in_features=in_features, out_features=out_features)
        torch_linear = torch.nn.Linear(in_features, out_features)

        ms_result = ms_linear(ms_tensor).asnumpy()
        torch_result = torch_linear(torch_tensor).detach().numpy()


        assert np.allclose(ms_result, torch_result,
                           atol=1e-3), f"Mismatch for in_features={in_features} and out_features={out_features}"


@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_linear_wrong_input(mode):
    """测试无效输入时，MindSpore和PyTorch的错误处理"""
    ms.set_context(mode=mode)

    ms_tensor = Tensor(input_data, dtype=ms.float32)
    torch_tensor = torch.tensor(input_data, dtype=torch.float32)

    try:
        ms_linear = mint.nn.Linear(in_features=4, out_features=3)
        ms_result = ms_linear(ms_tensor).asnumpy() 
    except Exception as e:
        print(f"MindSpore Linear error: {e}")

    try:
        weight_init = np.random.randn(3, 4).astype(np.float32)  
        bias_init = np.random.randn(3).astype(np.float32) 
        ms_linear = mint.nn.Linear(in_features=4, out_features=3, weight_init=weight_init, bias_init=bias_init)
        ms_result = ms_linear(ms_tensor).asnumpy() 
        print(f"MindSpore Linear with correct weight_init and bias_init output: {ms_result}")
    except Exception as e:
        print(f"MindSpore Linear weight_init/bias_init error: {e}")

    try:
        weight_init = np.random.randn(4, 3).astype(np.float32)
        bias_init = np.random.randn(4).astype(np.float32) 
        ms_linear = mint.nn.Linear(in_features=4, out_features=3, weight_init=Tensor(weight_init),
                                   bias_init=Tensor(bias_init))
        ms_result = ms_linear(ms_tensor)  
    except Exception as e:
        print(f"MindSpore Linear incorrect weight_init/bias_init shape error: {e}")

    try:
        incorrect_input_data = [[1, 6], [7, 3], [2, 9]] 
        incorrect_ms_tensor = Tensor(incorrect_input_data, dtype=ms.float32)
        ms_linear = mint.nn.Linear(in_features=4, out_features=3)
        ms_result = ms_linear(incorrect_ms_tensor)  
    except Exception as e:
        print(f"MindSpore Linear incorrect input shape error: {e}")

    try:
        torch_linear = torch.nn.Linear(in_features=4, out_features=3)
        torch_result = torch_linear(torch_tensor) 
    except Exception as e:
        print(f"PyTorch Linear error: {e}")

    try:
        weight_init = np.random.randn(3, 4).astype(np.float32)  
        bias_init = np.random.randn(3).astype(np.float32) 
        torch_linear = torch.nn.Linear(in_features=4, out_features=3)
        torch_linear.weight.data = torch.tensor(weight_init)
        torch_linear.bias.data = torch.tensor(bias_init) 
        torch_result = torch_linear(torch_tensor)  
        print(f"PyTorch Linear with correct weight_init and bias_init output: {torch_result}")
    except Exception as e:
        print(f"PyTorch Linear weight_init/bias_init error: {e}")

    try:
        weight_init = np.random.randn(4, 3).astype(np.float32) 
        bias_init = np.random.randn(4).astype(np.float32) 
        torch_linear = torch.nn.Linear(in_features=4, out_features=3)
        torch_linear.weight.data = torch.tensor(weight_init) 
        torch_linear.bias.data = torch.tensor(bias_init) 
        torch_result = torch_linear(torch_tensor) 
    except Exception as e:
        print(f"PyTorch Linear incorrect weight_init/bias_init shape error: {e}")

    try:
        incorrect_input_data = [[1, 6], [7, 3], [2, 9]]
        incorrect_torch_tensor = torch.tensor(incorrect_input_data, dtype=torch.float32)
        torch_linear = torch.nn.Linear(in_features=4, out_features=3)
        torch_result = torch_linear(incorrect_torch_tensor) 
    except Exception as e:
        print(f"PyTorch Linear incorrect input shape error: {e}")





