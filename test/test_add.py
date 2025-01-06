import pytest
import numpy as np
import mindspore as ms
from mindspore import mint, Tensor, value_and_grad
import torch


def prepare_data(x=[[1, 6, 2, 4],[7, 3, 8, 2],[2, 9, 11, 5]], y=[[3, 8, 1, 9],[4, 2, 4, 3],[8, 7, 23, 15]], dtype_ms=ms.float32, dtype_torch=torch.float32, requires_grad=False):
    
    ms_x = Tensor(x, dtype=dtype_ms)
    ms_y = Tensor(y, dtype=dtype_ms)
    torch_x = torch.tensor(x, dtype=dtype_torch, requires_grad=requires_grad)
    torch_y = torch.tensor(y, dtype=dtype_torch, requires_grad=requires_grad)
    return ms_x, ms_y, torch_x, torch_y


def is_same(input_x=[[1,6,2,4],[7,3,8,2],[2, 9, 11, 5]], input_y=[[3, 8, 1, 9],[4, 2, 4, 3],[8, 7, 23, 15]], alpha=1, shape=None, dtype_ms=ms.float32, dtype_torch=torch.float32):
    if shape != None:
        input_x = np.random.randn(*shape)
        input_y = np.random.randn(*shape)
    
    ms_x, ms_y, torch_x, torch_y = prepare_data(input_x, input_y, dtype_ms=dtype_ms, dtype_torch=dtype_torch)

    ms_result = mint.add(ms_x, ms_y, alpha=alpha)
    torch_result = torch.add(torch_x, torch_y, alpha=alpha)
    return np.allclose(ms_result.asnumpy(), torch_result.numpy())


@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_add_different_dtypes(mode):
    """测试random输入不同dtype，对比两个框架的支持度"""
    ms.set_context(mode=mode)
    ms_dtypes = [ms.int8, ms.int16, ms.int32, ms.int64, ms.uint8, ms.uint16, ms.uint32, ms.uint64, ms.float16, ms.float32, ms.float64, ms.bfloat16, ms.bool_]
    torch_dtypes = [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.uint16, torch.uint32, torch.uint64, torch.float16, torch.float32, torch.float64, torch.bfloat16, torch.bool]

    for i in range(len(ms_dtypes)):
        dtype_ms = ms_dtypes[i]
        dtype_torch = torch_dtypes[i]

        ms_x, ms_y, torch_x, torch_y = prepare_data(dtype_ms=dtype_ms, dtype_torch=dtype_torch)

        err = False
        try:
            ms_result = mint.add(ms_x, ms_y).asnumpy()
        except Exception as e:
            err = True
            print(f"mint.add not supported for {dtype_ms}")
            # print(e)

        try:
            torch_result = torch.add(torch_x, torch_y).numpy()
        except Exception as e:
            err = True
            print(f"torch.add not supported for {dtype_torch}")
            # print(e)

        if not err:
            assert np.allclose(ms_result, torch_result, atol=1e-3)


@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_add_random_input_fixed_dtype(mode):
    """测试固定数据类型下的随机输入值，对比输出误差"""
    ms.set_context(mode=mode)

    shapes = [[5], [5, 2], [5, 4, 3], [4, 6, 7, 8]]
    for i in range(len(shapes)):
        shape = shapes[i]
        result = is_same(shape=shape)
        assert result
        
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_add_different_para(mode):
    """测试固定shape，固定输入值，不同输入参数（string\bool等类型），两个框架的支持度"""
    ms.set_context(mode=mode)

    alphas = [0.5, 1, -2, 2.5, True]
    dtype_mss = [ms.float32, ms.int32, ms.int32, ms.float16, ms.bool_]
    dtype_torchs=[torch.float32, torch.int32, torch.int32, torch.float16, torch.bool]
    for dtype_ms, dtype_torch, alpha in zip(dtype_mss, dtype_torchs, alphas):
        result = is_same(dtype_ms=dtype_ms, dtype_torch=dtype_torch, alpha=alpha)
        assert result
            

@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_add_wrong_input(mode):
    """测试随机混乱输入，报错信息的准确性"""
    ms.set_context(mode=mode)
    
    input_x=[[1,6,2,4],[7,3,8,2],[2, 9, 11, 5]]
    input_y=[[3, 8, 1, 9],[4, 2, 4, 3],[8, 7, 23, 15]]

    try:
        ms_result = mint.add(input_x, input_y).asnumpy()
    except Exception as e:
        print(f"input不是 Tensor、number.Number、bool 报错信息：\n{e}")
        
    try:
        ms_result = mint.add(input_x, input_y, alpha=1.5).asnumpy()
    except Exception as e:
        print(f"alpha 是 float 类型，但是 input 不是 float 类型 报错信息：\n{e}")
        
    try:
        ms_result = mint.add(input_x, input_y, alpha=True).asnumpy()
    except Exception as e:
        print(f"alpha 是 bool 类型，但是 input 不是 bool 类型 报错信息：\n{e}")


@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_add_forward_back(mode):
    """使用Pytorch和MindSpore, 固定输入和权重, 测试正向推理结果和反向梯度"""
    ms.set_context(mode=mode)
    ms_x, ms_y, torch_x, torch_y = prepare_data(requires_grad=True)

    def forward_pt(x, y):
        return torch.add(x, y).sum()
    
    def forward_ms(x, y):
        return mint.add(x, y).sum()
    
    grad_fn = value_and_grad(forward_ms)
    output_ms, gradient_ms = grad_fn(ms_x, ms_y)
    output_pt = forward_pt(torch_x, torch_y)
    output_pt.backward()
    gradient_pt = torch_x.grad
    assert np.allclose(output_ms.asnumpy(), output_pt.detach().numpy(), atol=1e-3)
    assert np.allclose(gradient_ms.asnumpy(), gradient_pt.numpy(), atol=1e-3)