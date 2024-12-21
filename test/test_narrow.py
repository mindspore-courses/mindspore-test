import pytest
import numpy as np
import mindspore
import torch

"""
    测试mindspore.mint.narrow接口
    mindspore.mint.narrow(input, dim, start, length)
        沿着指定的轴，指定起始位置获取指定长度的Tensor。
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
def test_narrow_random_input_dtype(dtype):
    """
    测试random输入不同dtype，对比MindSpore和Pytorch的支持度
    """
    mindspore.set_context(mode=mindspore.PYNATIVE_MODE, pynative_synchronize=True)
    flag1 = True
    flag2 = True
    shape = (4, 4)
    dim = 0
    start = 1
    length = 2
    try:
        # MindSpore
        input_ms = mindspore.Tensor(np.random.random(size=shape).astype(dtype))
        result_ms = mindspore.mint.narrow(input_ms, dim, start, length)
        print(result_ms)
        assert isinstance(result_ms, mindspore.Tensor)
    except Exception as e:
        print(f"MindSpore出现报错: {e}")
        flag1 = False

    try:
        # Pytorch
        input_pt = torch.from_numpy(np.random.random(size=shape).astype(dtype))
        result_pt = torch.narrow(input_pt, dim, start, length)
        assert isinstance(result_pt, torch.Tensor)

    except Exception as e:
        print(f"Pytorch出现报错: {e}")
        flag2 = False
    if not flag1 and flag2:
        pytest.fail("mindspore不支持："+str(dtype))
    if flag1 and not flag2:
        pytest.fail("pytorch不支持："+str(dtype))
    if not flag1 and not flag2:
        pytest.fail("both mindspore 和 pytorch不支持："+str(dtype))



@pytest.mark.parametrize("input_shape", [
    (2,2),
    (4,4),
    (32,32),
    (64,64,64)
])
def test_narrow_fixed_dtype_random_value(input_shape):
    """
    测试固定dtype，random输入值，对比两个框架输出（误差范围小于1e-3）
    """
    try:
        input = np.random.random(size=input_shape).astype(np.float32)
        dim = 0
        start = 1
        length = 1
        # MindSpore部分
        input_ms = mindspore.Tensor(input)
        result_ms1 = mindspore.mint.narrow(input_ms, dim, start, length)
        a = result_ms1.asnumpy()
        # Pytorch部分
        input_pt = torch.from_numpy(input)
        result_pt = torch.narrow(input_pt, dim, start, length)

        assert np.allclose(a, result_pt.numpy(), atol=1e-3)
    except Exception as e:
        print(f"出现报错: {e}")
        pytest.fail(e)


@pytest.mark.parametrize("input_param", [(0, 1, 1)])
def test_narrow_fixed_shape_fixed_value_different_params(input_param):
    """
    测试固定shape，固定输入值，不同输入参数类型，两个框架的支持度
    """
    input_value = np.array([[1, 2], [3, 4]]).astype(np.float32)
    dim, start, length = input_param
    flag = True
    try:
        # MindSpore部分
        input_ms = mindspore.Tensor(input_value)
        result_ms = mindspore.mint.narrow(input_ms, dim, start, length)

        assert isinstance(result_ms, mindspore.Tensor)
    except Exception as e:
        print(f"MindSpore出现报错: {e}")
        flag = False
    try:
        # Pytorch部分
        input_pt = torch.from_numpy(input_value)
        result_pt = torch.narrow(input_pt, dim, start, length)

        assert isinstance(result_pt, torch.Tensor)
    except Exception as e:
        print(f"Pytorch出现报错: {e}")
        flag = False
    assert flag


@pytest.mark.parametrize("random_messy_input", [
    ([[1, 2], [3, 4]], 0, 0, 1, TypeError), 
    (mindspore.tensor([[1, 2], [3, 4]]), 2, 0, 1, ValueError),
    (mindspore.tensor([[1, 2], [3, 4]]), -3, 0, 1, ValueError), 
    (mindspore.tensor([[1, 2], [3, 4]]), 0, 2, 1, ValueError), 
    (mindspore.tensor([[1, 2], [3, 4]]), 0, -3, 1, ValueError), 
    (mindspore.tensor([[1, 2], [3, 4]]), 0, 0, 3, ValueError), 
    (mindspore.tensor([[1, 2], [3, 4]]), 0, 0, -1, ValueError)
])
def test_narrow_random_messy_input_error_info(random_messy_input):
    """
    测试随机混乱输入，报错信息的准确性
    """
    flag = False
    input = random_messy_input[0]
    dim = random_messy_input[1]
    start = random_messy_input[2]
    length = random_messy_input[3]
    try:
        result_ms = mindspore.mint.narrow(input, dim, start, length)
        print(result_ms)
    except Exception as e_ms:
        flag = True
        assert isinstance(e_ms, random_messy_input[4])
        
    if not flag:
        pytest.fail("在预期应捕获异常的情况下，未捕获到任何异常，测试不通过")




# 测试使用该接口的构造函数的准确性


def test_narrow_in_neural_network():
    """
    测试一个包含narrow操作的网络示例
    """
    input_value = np.random.random(size=(4,4)).astype(np.float32)

    # mindspore
    class SimpleNet_ms(mindspore.nn.Cell):
        def __init__(self):
            super(SimpleNet_ms, self).__init__()

        def construct(self, x):
            narrow_result = mindspore.mint.narrow(x, 0, 1, 2)
            return narrow_result

    input_ms = mindspore.Tensor(input_value)
    net_ms = SimpleNet_ms()
    result_ms = net_ms(input_ms)

    # pytorch
    class SimpleNet_pt(torch.nn.Module):
        def __init__(self):
            super(SimpleNet_pt, self).__init__()

        def forward(self, x):
            narrow_result = torch.narrow(x, 0, 1, 2)
            return narrow_result

    input_pt = torch.from_numpy(input_value)
    net_pt = SimpleNet_pt()
    result_pt = net_pt(input_pt)

    assert np.allclose(result_ms.asnumpy(), result_pt.numpy(), atol=1e-3)


def test_narrow_backward():
    """
    测试函数反向 
    """
    # 定义简单函数
    def function_ms(x):
        return mindspore.mint.narrow(x, 0, 1, 2).sum()
    def function_pt(x):
        return torch.narrow(x, 0, 1, 2).sum()

    input_value = np.random.rand(4, 4).astype(np.float32)

    input_ms = mindspore.Tensor(input_value, mindspore.float32)
    grad_ms = mindspore.grad(function_ms)(input_ms)

    input_pt = torch.from_numpy(input_value).float().requires_grad_(True)
    result_pt = function_pt(input_pt)
    result_pt.backward()
    grad_pt = input_pt.grad

    assert np.allclose(grad_ms.asnumpy(), grad_pt.numpy(), atol=1e-3)




