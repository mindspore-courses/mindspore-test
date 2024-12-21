import pytest
import numpy as np
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import torch


"""
    测试mindspore.mint.split接口
    mindspore.mint.split: 将张量切块
"""
@pytest.mark.parametrize("dtype", [ 
    np.bool_,
    np.int_,
    np.intc,
    np.intp,
    np.int8,
    np.int16,
    np.int32, 
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32, 
    np.uint64,
    np.float_,
    np.float16,
    np.float32, 
    np.float64,
    np.complex_,
    np.complex64,
    np.complex128
])
def test_split_random_input_dtype(dtype):
    """
    测试random输入不同dtype，对比MindSpore和Pytorch的支持度
    """
    mindspore.set_context(mode=mindspore.PYNATIVE_MODE, pynative_synchronize=True)
    flag1 = True
    flag2 = True
    shape = (4,4)
    split_size_or_sections = [2, 2]
    dim = 0
    try:
        # MindSpore
        input_ms = mindspore.Tensor(np.random.random(size=shape).astype(dtype))
        result_ms = mindspore.mint.split(input_ms, split_size_or_sections, dim)

        assert isinstance(result_ms, tuple) and all(isinstance(x, mindspore.Tensor) for x in result_ms)
    except Exception as e:
        print(f"MindSpore出现报错: {e}")
        flag1 = False

    try:
        # Pytorch
        input_pt = torch.from_numpy(np.random.random(size=shape).astype(dtype))
        result_pt = torch.split(input_pt, split_size_or_sections, dim)
        assert isinstance(result_pt, tuple) and all(isinstance(x, torch.Tensor) for x in result_pt)

    except Exception as e:
        print(f"Pytorch出现报错: {e}")
        flag2 = False
    if not flag1 and flag2:
        pytest.fail("mindspore不支持："+str(dtype))
    if flag1 and not flag2:
        pytest.fail("pytorch不支持："+str(dtype))
    if not flag1 and not flag2:
        pytest.fail("both mindspore 和 pytorch不支持："+str(dtype))
        


@pytest.mark.parametrize("input_value", [np.array([[1, 2], [3, 4]]).astype(np.float32),
                                        np.array([[5, 6], [7, 8]]).astype(np.float32),
                                        np.array([[9, 10], [11, 12]]).astype(np.float32),
                                        np.array([[13, 14], [15, 16]]).astype(np.float32)])
def test_split_fixed_dtype_random_value(input_value):
    """
    测试固定dtype，random输入值，对比两个框架输出（误差范围小于1e-3）
    """
    dtype = input_value.dtype
    split_size_or_sections = [1, 1]
    dim = 0
    # MindSpore部分
    input_ms = mindspore.Tensor(input_value)
    result_ms = mindspore.mint.split(input_ms, split_size_or_sections, dim)

    # Pytorch部分
    input_pt = torch.from_numpy(input_value)
    result_pt = torch.split(input_pt, split_size_or_sections, dim)

    for r_ms, r_pt in zip(result_ms, result_pt):
        assert np.allclose(r_ms.asnumpy(), r_pt.numpy(), atol=1e-3)


@pytest.mark.parametrize("input_param", [([1,1], 0), (2, 0)])
def test_split_fixed_shape_fixed_value_different_params(input_param):
    """
    测试固定shape，固定输入值，不同输入参数类型，两个框架的支持度
    split_size_or_sections参数的不同类型
    """
    split_size_or_sections, dim = input_param
    input_value = np.array([[1, 2], [3, 4]]).astype(np.float32)
    flag = True
    try:
        # MindSpore部分
        input_ms = mindspore.Tensor(input_value)

        result_ms = mindspore.mint.split(input_ms, split_size_or_sections, dim)

        assert isinstance(result_ms, tuple)
    except Exception as e:
        print(f"MindSpore出现报错: {e}")
        flag = False
    try:
        # Pytorch部分
        input_pt = torch.from_numpy(input_value)
  
        result_pt = torch.split(input_pt, split_size_or_sections, dim)


        assert isinstance(result_pt, tuple)
    except Exception as e:
        print(f"Pytorch出现报错: {e}")
        flag = False
    assert flag


@pytest.mark.parametrize("random_messy_input", [
    ([[1, 2], [3, 4]], 2, 1,TypeError), 
    (mindspore.tensor([[1, 2], [3, 4]]), 2, 1.0, TypeError),
    (mindspore.tensor([[1, 2], [3, 4]]), 2, 3, ValueError),
    (mindspore.tensor([[1, 2], [3, 4]]), [1.0, 1.0], 0, TypeError),
    (mindspore.tensor([[1, 2], [3, 4]]), mindspore.tensor([1,1]), 1,TypeError),
    (mindspore.tensor([[1, 2], [3, 4]]), [1,1,1], 0, ValueError)
])
def test_split_random_messy_input_error_info(random_messy_input):
    """
    测试随机混乱输入，报错信息的准确性
        TypeError - tensor 不是Tensor。
        TypeError - dim 不是int类型。
        ValueError - dim 不在[-tensor.ndim, tensor.ndim)范围中。
        TypeError - split_size_or_sections 中的每个元素不是int类型。
        TypeError - split_size_or_sections 不是int，tuple(int)或list(int)。
        ValueError - split_size_or_sections 的和不等于tensor.shape[dim]。
    """
    flag = False
    input = random_messy_input[0]
    split_size_or_sections = random_messy_input[1]
    dim = random_messy_input[2]
    try:
        result_ms = mindspore.mint.split(input, split_size_or_sections, dim)
    except Exception as e_ms:
        assert isinstance(e_ms, random_messy_input[3])
        flag = True
    if not flag:
        pytest.fail("在预期应捕获异常的情况下，未捕获到任何异常，测试不通过")




# 以下测试使用该接口的构造函数/神经网络的准确性




def test_split_in_neural_network():
    """
    测试一个包含split操作的网络示例
    """
    input_value = np.random.rand(4, 4).astype(np.float32)

    # mindspore
    class SimpleNet_ms(nn.Cell):
        def __init__(self):
            super(SimpleNet_ms, self).__init__()

        def construct(self, x):
            split_result = mindspore.mint.split(x, 2, 0)
            return split_result[0]

    
    input_ms = mindspore.Tensor(input_value)
    net_ms = SimpleNet_ms()
    result_ms = net_ms(input_ms)
    
    # pytorch
    class SimpleNet_pt(torch.nn.Module):
        def __init__(self):
            super(SimpleNet_pt, self).__init__()

        def forward(self, x):
            split_result = torch.split(x, 2, 0)
            return split_result[0]

    input_pt = torch.from_numpy(input_value)
    net_pt = SimpleNet_pt()
    result_pt = net_pt(input_pt)

    assert np.allclose(result_ms.asnumpy(), result_pt.numpy(), atol=1e-3)


def test_split_backward():
    """
    测试函数反向 
    """
    # 定义简单函数
    def function_ms(x):
        return mindspore.mint.split(x,2,0)[0].sum()
    def function_pt(x):
        return torch.split(x,2,0)[0].sum()

    input_value = np.random.rand(4, 4).astype(np.float32)

    input_ms = mindspore.Tensor(input_value, mindspore.float32)
    grad_ms = mindspore.grad(function_ms)(input_ms)

    input_pt = torch.from_numpy(input_value).float().requires_grad_(True)
    result_pt = function_pt(input_pt)
    result_pt.backward()
    grad_pt = input_pt.grad

    assert np.allclose(grad_ms.asnumpy(), grad_pt.numpy(), atol=1e-3)


