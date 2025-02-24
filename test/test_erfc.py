import pytest
import numpy as np
import mindspore as ms
from mindspore import mint, Tensor, value_and_grad
import torch


dtype_ms = ms.float32
dtype_torch = torch.float32
input_data = [[1, 6, 2, 4], [7, 3, 8, 2], [2, 9, 11, 5]]
ms_tensor = Tensor(input_data, dtype_ms)
torch_tensor = torch.tensor(input_data, dtype = dtype_torch)


def is_same(input_data=[[1, 6, 2, 4], [7, 3, 8, 2], [2, 9, 11, 5]], shape=None, dtype_ms=ms.float32, dtype_torch=torch.float32):
    if shape is not None:
        input_data = np.random.randn(*shape)

    ms_tensor = Tensor(input_data, dtype_ms)
    torch_tensor = torch.tensor(input_data, dtype = dtype_torch)

    ms_result = mint.erfc(ms_tensor).asnumpy()
    torch_result = torch.erfc(torch_tensor).numpy()
    if  np.allclose(ms_result, torch_result):
        return True
    else:
        print(f"shape: {shape} got wrong result")
        print(f"input_data: {input_data}")
        print(f"ms_result: {ms_result}")
        print(f"torch_result: {torch_result}")    
        return False


@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_erfc_different_dtypes(mode):
    """测试random输入不同dtype，对比两个框架的支持度"""
    ms.set_context(mode=mode)
    ms_dtypes = [ms.float16, ms.float32, ms.float64, ms.int64, ms.bool_, ms.bfloat16]
    torch_dtypes = [torch.float16, torch.float32, torch.float64, torch.int64, torch.bool, torch.bfloat16]

    for i in range(len(ms_dtypes)):
        dtype_ms = ms_dtypes[i]
        dtype_torch = torch_dtypes[i]
        ms_tensor = Tensor(input_data, dtype_ms)
        torch_tensor = torch.tensor(input_data, dtype=dtype_torch)

        err = False
        try:
            ms_result = mint.erfc(ms_tensor).asnumpy()
        except Exception as e:
            err = True
            print(f"mint.erfc not supported for {dtype_ms}")

        try:
            torch_result = torch.erfc(torch_tensor).numpy()
        except Exception as e:
            err = True
            print(f"torch.erfc not supported for {dtype_torch}")

        if not err:
            if not np.allclose(ms_result, torch_result):
                print(f"mint.erfc is supported for {dtype_ms} but not working properly")


@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_erfc_random_input_fixed_dtype(mode):
    """测试固定数据类型下的随机输入值，对比输出误差"""
    ms.set_context(mode=mode)

    shapes = [[5], [5, 2], [5, 4, 3], [4, 6, 7, 8]]
    for shape in shapes:
        result = is_same(shape=shape)
        assert result
    

@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_erfc_different_para(mode):
    """测试固定shape，固定输入值，不同输入参数（string\bool等类型），两个框架的支持度"""
    """这个无其他参数，所以不需要测试"""
    pass
    ms.set_context(mode=mode)

    shapes = [[5], [5, 2], [5, 4, 3], [4, 6, 7, 8]]
    for shape in shapes:
        result = is_same(shape=shape)
        assert result


@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_erfc_wrong_input(mode):
    """测试随机混乱输入，报错信息的准确性"""
    ms.set_context(mode=mode)

    try:
        ms_result = mint.erfc('a').asnumpy()
    except TypeError as e:
        print(f"测试通过！TypeError -input 不是 Tensor 报错信息：\n{e}")

    try:
        ms_result = ms.mint.erfc(Tensor(input_data, ms.int32)).asnumpy()
    except TypeError as e:
        print(f"测试通过！Ascend: 如果 input 的数据类型不是 float16、float32、float64、int64、bool、bfloat16 报错信息：\n{e}")


@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_erfc_forward_back(mode):
    """使用Pytorch和MindSpore, 固定输入和权重, 测试正向推理结果和反向梯度"""
    ms.set_context(mode=mode)

    torch_tensor = torch.tensor(input_data, dtype=dtype_torch, requires_grad=True)

    def forward_pt(x):
        return torch.erfc(x)
    
    def forward_ms(x):
        return mint.erfc(x)
    
    grad_fn = value_and_grad(forward_ms)
    output_ms, gradient_ms = grad_fn(ms_tensor)
    output_pt = forward_pt(torch_tensor)
    output_pt.backward(torch.ones_like(output_pt))
    gradient_pt = torch_tensor.grad
    assert np.allclose(output_ms.asnumpy(), output_pt.detach().numpy(), atol=1e-3)
    assert np.allclose(gradient_ms.asnumpy(), gradient_pt.numpy(), atol=1e-3)
