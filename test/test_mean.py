import pytest
import numpy as np
import mindspore as ms
from mindspore import mint, Tensor, value_and_grad
import torch


dtype_ms = ms.float32
dtype_torch = torch.float32
input_data = [[1,6,2,4],[7,3,8,2],[2, 9, 11, 5]]
ms_tensor = Tensor(input_data, dtype_ms)
torch_tensor = torch.tensor(input_data, dtype=dtype_torch)


def is_same(input_data=[[1,6,2,4],[7,3,8,2],[2, 9, 11, 5]], shape=None, dtype_ms=ms.float32, dtype_torch=torch.float32, dim=-1, keepdim=False):
    if shape != None:
        input_data = np.random.randn(*shape)

    ms_tensor = Tensor(input_data, dtype_ms)
    torch_tensor = torch.tensor(input_data, dtype=dtype_torch)

    ms_result = mint.mean(ms_tensor, dim=dim, keepdim=keepdim).asnumpy()
    torch_result = torch.mean(torch_tensor, dim=dim, keepdim=keepdim).numpy()

    return np.allclose(ms_result, torch_result)


@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_mean_different_dtypes(mode):
    """测试random输入不同dtype，对比两个框架的支持度"""
    ms.set_context(mode=mode)
    ms_dtypes = [ms.float16, ms.float32, ms.float64]
    torch_dtypes = [torch.float16, torch.float32, torch.float64]

    for i in range(len(ms_dtypes)):
        dtype_ms = ms_dtypes[i]
        dtype_torch = torch_dtypes[i]
        ms_tensor = Tensor(input_data, dtype_ms)
        torch_tensor = torch.tensor(input_data, dtype=dtype_torch)

        err = False
        try:
            ms_result = mint.mean(ms_tensor).asnumpy() 
        except Exception as e:
            er = True
            print(f"mint.mean not supported for {dtype_ms}")
            # print(e)

        try:
            torch_result = torch.mean(torch_tensor).numpy() 
        except Exception as e:
            err = True
            print(f"torch.mean not supported for {dtype_torch}")
            # print(e)
        if not err:
            assert np.allclose(ms_result, torch_result)


@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_mean_random_input_fixed_dtype(mode):
    """测试固定数据类型下的随机输入值，对比输出误差"""
    ms.set_context(mode=mode)

    shapes = [[5], [5, 2], [5, 4, 3], [4, 6, 7, 8]]
    for i in range(len(shapes)):
        shape = shapes[i]
        result = is_same(shape=shape)
        assert result


@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_mean_different_para(mode):
    """测试固定shape，固定输入值，不同输入参数（string\bool等类型），两个框架的支持度"""
    ms.set_context(mode=mode)

    dims = [None, 0, 1]
    keepdims = [True, False]
    paras = [(dim, keepdim) for dim in dims for keepdim in keepdims]

    for dim, keepdim in paras:
        result = is_same(dim=dim, keepdim=keepdim)
        assert result


@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_mean_wrong_input(mode):
    """测试随机混乱输入，报错信息的准确性"""
    ms.set_context(mode=mode)

    try:
        ms_result = mint.mean(ms_tensor, dim=2, keepdim=True).asnumpy()
    except Exception as e:
        print(f"dim 超出范围报错信息：\n{e}")
    

@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_mean_forward_back(mode):
    """使用Pytorch和MindSpore, 固定输入和权重, 测试正向推理结果和反向梯度"""
    ms.set_context(mode=mode)

    torch_tensor = torch.tensor(input_data, dtype=dtype_torch, requires_grad=True)

    def forward_pt(x):
        return torch.mean(x)
    
    def forward_ms(x):
        return mint.mean(x)
    
    grad_fn = value_and_grad(forward_ms)
    output_ms, gradient_ms = grad_fn(ms_tensor)
    output_pt = forward_pt(torch_tensor)
    output_pt.backward()
    gradient_pt = torch_tensor.grad
    assert np.allclose(output_ms.asnumpy(), output_pt.detach().numpy(), atol=1e-3)
    assert np.allclose(gradient_ms.asnumpy(), gradient_pt.numpy(), atol=1e-3)
        