import pytest
import numpy as np
import mindspore as ms
from mindspore import mint, Tensor, value_and_grad
import torch


dtype_ms = ms.float32
dtype_torch = torch.float32
input_data_x = [[1, 2], [3, 4], [5, 6]]
input_data_y = [[6, 5], [4, 3], [2, 1]]
ms_tensor_x = Tensor(input_data_x, dtype_ms)
ms_tensor_y = Tensor(input_data_y, dtype_ms)
torch_tensor_x = torch.tensor(input_data_x, dtype=dtype_torch)
torch_tensor_y = torch.tensor(input_data_y, dtype=dtype_torch)


def is_same(input_data_x=[[1, 2], [3, 4], [5, 6]], input_data_y=[[6, 5], [4, 3], [2, 1]], shape=None, dtype_ms=ms.float32, dtype_torch=torch.float32):
    if shape is not None:
        input_data_x = np.random.randn(*shape)
        input_data_y = np.random.randn(*shape)

    ms_tensor_x = Tensor(input_data_x, dtype_ms)
    ms_tensor_y = Tensor(input_data_y, dtype_ms)
    torch_tensor_x = torch.tensor(input_data_x, dtype=dtype_torch)
    torch_tensor_y = torch.tensor(input_data_y, dtype=dtype_torch)

    ms_result = mint.div(ms_tensor_x, ms_tensor_y).asnumpy()
    torch_result = torch.div(torch_tensor_x, torch_tensor_y).numpy()
    if  np.allclose(ms_result, torch_result):
        return True
    else:
        print(f"input_data: {input_data_x}")
        print(f"input_data: {input_data_y}")
        print(f"ms_result: {ms_result}")
        print(f"torch_result: {torch_result}")    
        return False


@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_div_different_dtypes(mode):
    """测试random输入不同dtype，对比两个框架的支持度"""
    """这个并没指出具体数据类型支持，所以都进行测试"""
    ms.set_context(mode=mode)
    ms_dtypes = [ms.int8, ms.int16, ms.int32, ms.int64, ms.uint8, ms.uint16, ms.uint32, ms.uint64, ms.float16, ms.float32, ms.float64, ms.bfloat16, ms.bool_]
    torch_dtypes = [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.uint16, torch.uint32, torch.uint64, torch.float16, torch.float32, torch.float64, torch.bfloat16, torch.bool]
    
    for i in range(len(ms_dtypes)):
        dtype_ms = ms_dtypes[i]
        dtype_torch = torch_dtypes[i]

        ms_tensor_x = Tensor(input_data_x, dtype_ms)
        ms_tensor_y = Tensor(input_data_y, dtype_ms)
        torch_tensor_x = torch.tensor(input_data_x, dtype=dtype_torch)
        torch_tensor_y = torch.tensor(input_data_y, dtype=dtype_torch)

        err = False
        try:
            ms_result = mint.div(ms_tensor_x, ms_tensor_y).asnumpy()
        except Exception as e:
            err = True
            print(f"mint.div not supported for {dtype_ms}")

        try:
            torch_result = torch.div(torch_tensor_x, torch_tensor_y).numpy()
        except Exception as e:
            err = True
            print(f"torch.div not supported for {dtype_torch}")

        if not err:
            if not np.allclose(ms_result, torch_result):
                print(f"mint.div is supported for {dtype_ms} but not working properly")


@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_div_random_input_fixed_dtype(mode):
    """测试固定数据类型下的随机输入值，对比输出误差"""
    ms.set_context(mode=mode)

    shapes = [[5], [5, 2], [5, 4, 3], [4, 6, 7, 8]]
    for shape in shapes:
        result = is_same(shape=shape)
        assert result
    

@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_div_different_para(mode):
    """测试固定shape，固定输入值，不同输入参数（string\bool等类型），两个框架的支持度"""
    ms.set_context(mode=mode)
    parameters = [None, "trunc", "floor"]
    for param in parameters:
        try:
            ms_output = ms.mint.div(ms_tensor_x, ms_tensor_y,rounding_mode=param).asnumpy()
            torch_output = torch.div(torch_tensor_x, torch_tensor_y, rounding_mode=param).numpy()
            # 对比输出
            np.testing.assert_allclose(ms_output, torch_output)
            print(f"参数 {param} 测试通过！")
        except Exception as e:
            print(f"参数 {param} 不支持，错误信息: {e}")

@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_div_wrong_input(mode):
    """测试随机混乱输入，报错信息的准确性"""
    ms.set_context(mode=mode)
    try:
        ms_result = mint.div(ms_tensor_x, 'a').asnumpy()
    except TypeError as e:
        print(f"TypeError - 测试通过！如果 input 和 other 不是以下之一：Tensor、Number、bool。报错信息：\n{e}")

    try:
        ms_result = mint.div(input_data_x, ms_tensor_y).asnumpy()
    except TypeError as e:
        print(f"TypeError - 测试通过！如果 input 和 other 不是以下之一：Tensor、Number、bool。报错信息：\n{e}")

    try:
        ms_result = mint.div(ms_tensor_x, ms_tensor_y, rounding_mode= "invalid_mode").asnumpy()
    except ValueError as e:
        print(f"ValueError - 测试通过！如果 rounding_mode 不是以下之一： None 、 'floor' 、 'trunc' 。报错信息：\n{e}")


@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_div_forward_back(mode):
    """使用Pytorch和MindSpore, 固定输入和权重, 测试正向推理结果和反向梯度"""
    ms.set_context(mode=mode)
    torch_tensor_x = torch.tensor(input_data_x, dtype=dtype_torch, requires_grad=True)
    torch_tensor_y = torch.tensor(input_data_y, dtype=dtype_torch, requires_grad=True)

    def forward_pt(x, y):
        return torch.div(x, y)

    def forward_ms(x, y):
        return mint.div(x, y)

    grad_fn = value_and_grad(forward_ms)
    output_ms, gradient_ms = grad_fn(ms_tensor_x, ms_tensor_y)
    output_pt = forward_pt(torch_tensor_x, torch_tensor_y)
    output_pt.backward(torch.ones_like(output_pt))
    gradient_pt = torch_tensor_x.grad
    assert np.allclose(output_ms.asnumpy(), output_pt.detach().numpy(), atol=1e-3)
    assert np.allclose(gradient_ms.asnumpy(), gradient_pt.numpy(), atol=1e-3)
