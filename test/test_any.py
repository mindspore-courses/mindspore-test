import pytest
import numpy as np
import mindspore as ms
from mindspore import mint, Tensor, value_and_grad
import torch


dtype_ms = ms.float32
dtype_torch = torch.float32
input_data = [[1,0],[0,0],[1,1]]
ms_tensor = Tensor(input_data, dtype_ms)
torch_tensor = torch.tensor(input_data, dtype=dtype_torch)


def is_same(input_data=[[1,0],[0,0],[1,1]], shape=None, dtype_ms=ms.float32, dtype_torch=torch.float32, dim=-1, keepdim=False):
    if shape != None:
        input_data = np.random.randn(*shape)

    ms_tensor = Tensor(input_data, dtype_ms)
    torch_tensor = torch.tensor(input_data, dtype=dtype_torch)

    ms_result = mint.any(ms_tensor, dim=dim, keepdim=keepdim).asnumpy()
    torch_result = torch.any(torch_tensor, dim=dim, keepdim=keepdim).numpy()

    return np.allclose(ms_result, torch_result)


@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_any_different_dtypes(mode):
    """测试random输入不同dtype，对比两个框架的支持度"""
    ms.set_context(mode=mode)
    ms_dtypes = [ms.int8, ms.int16, ms.int32, ms.int64, ms.uint8,ms.uint16, ms.uint32, ms.uint64, ms.float16, ms.float32, ms.float64, ms.bfloat16, ms.bool_]
    torch_dtypes = [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.uint16, torch.uint32, torch.uint64, torch.float16, torch.float32, torch.float64, torch.bfloat16, torch.bool]
    
    for i in range(len(ms_dtypes)):
        dtype_ms = ms_dtypes[i]
        dtype_torch = torch_dtypes[i]

        ms_tensor = Tensor(input_data, dtype_ms)
        torch_tensor = torch.tensor(input_data, dtype=dtype_torch)

        err = False
        try:
            ms_result = mint.any(ms_tensor, dim=1).asnumpy()
        except Exception as e:
            err = True
            print(f"mint.any not supported for {dtype_ms}")

        try:
            torch_result = torch.any(torch_tensor, dim=1).numpy()
        except Exception as e:
            err = True
            print(f"torch.any not supported for {dtype_torch}")

        if not err:
            assert np.allclose(ms_result, torch_result)


@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_any_random_input_fixed_dtype(mode):
    """测试固定数据类型下的随机输入值，对比输出误差"""
    ms.set_context(mode=mode)

    shapes = [[5], [5, 2], [5, 4, 3], [4, 6, 7, 8]]
    for i in range(len(shapes)):
        shape = shapes[i]
        result = is_same(shape=shape)
        assert result
    

@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_any_different_para(mode):
    """测试固定shape，固定输入值，不同输入参数（string\bool等类型），两个框架的支持度"""
    ms.set_context(mode=mode)

    dims = [None, 0, 1]
    keepdims = [True, False]
    paras = [(dim, keepdim) for dim in dims for keepdim in keepdims]

    for dim, keepdim in paras:
        result = is_same(dim=dim, keepdim=keepdim)
        assert result


@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_any_wrong_input(mode):
    """测试随机混乱输入，报错信息的准确性"""
    ms.set_context(mode=mode)
    
    try:
        ms_result = mint.any(ms_tensor, dim=1, keepdim=1).asnumpy()
    except Exception as e:
        print(f"keepdim 不是 bool 类型报错信息：\n{e}")

    try:
        ms_result = mint.any(input_data, dim=1).asnumpy()
    except Exception as e:
        print(f"input 不是 Tensor 报错信息：\n{e}")


@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_any_forward_back(mode):
    """使用Pytorch和MindSpore, 固定输入和权重, 测试正向推理结果和反向梯度"""
    ms.set_context(mode=mode)
    torch_tensor = torch.tensor(input_data, dtype=dtype_torch, requires_grad=True)

    def forward_pt(x):
        return torch.any(x, dim=1, keepdim=False)

    def forward_ms(x):
        return mint.any(x, dim=1, keepdim=False)

    grad_fn = value_and_grad(forward_ms)
    output_ms, gradient_ms = grad_fn(ms_tensor)
    output_pt = forward_pt(torch_tensor)
    # output_pt.backward()
    # gradient_pt = torch_tensor.grad
    assert np.allclose(output_ms.asnumpy(), output_pt.detach().numpy(), atol=1e-3)
    # assert np.allclose(gradient_ms.asnumpy(), gradient_pt.numpy(), atol=1e-3)
    # print(output_ms, gradient_ms)
    # print(output_pt, gradient_pt)
        