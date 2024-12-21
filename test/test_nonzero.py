import mindspore.nn
import pytest
import numpy as np
import mindspore
import torch


"""
    测试mindspore.mint.nonzero接口
    mindspore.mint.nonzero(input, as_tuple=False)
        返回所有非零元素下标位置。
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
def test_nonzero_random_input_dtype(dtype):
    """
    测试random输入不同dtype，对比MindSpore和Pytorch的支持度
    """
    mindspore.set_context(mode=mindspore.PYNATIVE_MODE, pynative_synchronize=True)
    flag1 = True
    flag2 = True
    
    shape = (4, 4)
    try:
        # MindSpore
        input_ms = mindspore.Tensor(np.random.random(size=shape).astype(dtype))
        result_ms = mindspore.mint.nonzero(input_ms)

        assert isinstance(result_ms, mindspore.Tensor) and result_ms.ndim >= 2
    except Exception as e:
        print(f"MindSpore出现报错: {e}")
        flag1 = False

    try:
        # Pytorch
        input_pt = torch.from_numpy(np.random.random(size=shape).astype(dtype))
        result_pt = torch.nonzero(input_pt)
        assert isinstance(result_pt, torch.Tensor) and result_pt.ndim >= 2

    except Exception as e:
        print(f"Pytorch出现报错: {e}")
        flag2 = False

    if not flag1 and flag2:
        pytest.fail("mindspore不支持："+str(dtype))
    if flag1 and not flag2:
        pytest.fail("pytorch不支持："+str(dtype))
    if not flag1 and not flag2:
        pytest.fail("both mindspore 和 pytorch不支持："+str(dtype))


@pytest.mark.parametrize("input_value", [np.array([[1, 0], [0, 1]]).astype(np.float32),
                                        np.array([[0, 1], [1, 0]]).astype(np.float32),
                                        np.array([[1, 1], [0, 0]]).astype(np.float32),
                                        np.array([[0, 0], [1, 1]]).astype(np.float32)])
def test_nonzero_fixed_dtype_random_value(input_value):
    """
    测试固定dtype，random输入值，对比两个框架输出（误差范围小于1e-3）
    """
    dtype = input_value.dtype
    # MindSpore部分
    input_ms = mindspore.Tensor(input_value)
    result_ms = mindspore.mint.nonzero(input_ms)
    print(result_ms.dtype)
    # Pytorch部分
    input_pt = torch.from_numpy(input_value)
    result_pt = torch.nonzero(input_pt)

    assert np.allclose(result_ms.asnumpy(), result_pt.numpy(), atol=1e-3)


@pytest.mark.parametrize("input_param", [
    False, 
    True
])
def test_nonzero_fixed_shape_fixed_value_different_params(input_param):
    """
    测试固定shape，固定输入值，不同输入参数，两个框架的支持度
    """
    mindspore.set_context(mode=mindspore.PYNATIVE_MODE, pynative_synchronize=True)
    input_value = np.array([1,0,0,1]).astype(np.float32)
    as_tuple = input_param
    flag = True
    try:
        # MindSpore部分
        input_ms = mindspore.Tensor(input_value)
        result_ms = mindspore.mint.nonzero(input_ms, as_tuple=input_param)

        if not as_tuple:
            assert isinstance(result_ms, mindspore.Tensor)
        else:
            assert isinstance(result_ms, tuple)
    except Exception as e:
        print(f"MindSpore出现报错: {e}")
        flag = False
    try:
        # Pytorch部分
        input_pt = torch.from_numpy(input_value)
        result_pt = torch.nonzero(input_pt,as_tuple=input_param)
        
        if not as_tuple:
            assert isinstance(result_pt, torch.Tensor)
        else:
            assert isinstance(result_pt, tuple)
    except Exception as e:
        print(f"Pytorch出现报错: {e}")
        flag = False
    assert flag


@pytest.mark.parametrize("random_messy_input", [
    (np.array([[1, 2], [3, 4]]).astype(np.float32), False, TypeError),
    (mindspore.tensor([[1, 2], [3, 4]]), "str", TypeError),
    (mindspore.tensor(0), False, ValueError)
])
def test_nonzero_random_messy_input_error_info(random_messy_input):
    """
    测试随机混乱输入，报错信息的准确性
    TypeError - 如果 input 不是Tensor。
    TypeError - 如果 as_tuple 不是bool。
    ValueError - 如果 input 的维度为0。
    """
    mindspore.set_context(mode=mindspore.PYNATIVE_MODE, pynative_synchronize=True)
    flag = False
    input = random_messy_input[0]
    as_tuple = random_messy_input[1]
    try:
        result_ms = mindspore.mint.nonzero(input, as_tuple=as_tuple)
        print(result_ms)
    except Exception as e_ms:
        assert isinstance(e_ms, random_messy_input[2])
        flag = True
    if not flag:
        pytest.fail("在预期应捕获异常的情况下，未捕获到任何异常，测试不通过")



# 以下测试使用该接口的函数/神经网络的准确性


def test_nonzero_in_neural_network():
    """
    测试包含nonzero操作的网络示例
    """
    input_value = np.random.random(size=(2, 10)).astype(np.float32)

    class Net_ms(mindspore.nn.Cell):
        def __init__(self):
            super(Net_ms, self).__init__()
        def construct(self, x):
            nonzero_indices = mindspore.mint.nonzero(x)
            mask = mindspore.ops.ZerosLike()(x)
            for index in nonzero_indices:
                mask[index[0], index[1]] = 1
            out = x * mask
            return out.sum()
        
    class Net_pt(torch.nn.Module):
        def __init__(self):
            super(Net_pt, self).__init__()
        def forward(self, x):
            nonzero_indices = torch.nonzero(x)
            mask = torch.zeros_like(x)
            mask[nonzero_indices[:, 0], nonzero_indices[:, 1]] = 1
            out = x * mask
            return out.sum()

    input_ms = mindspore.Tensor(input_value)
    net_ms = Net_ms()
    result_ms = net_ms(input_ms)


    input_pt = torch.from_numpy(input_value)
    net_pt = Net_pt()
    result_pt = net_pt(input_pt)

    assert np.allclose(result_ms.asnumpy(), result_pt.detach().numpy(), atol=1e-3)


def test_nonzero_backward():
    """
    测试函数反向 
    """
    # 定义简单函数
    class Net_ms(mindspore.nn.Cell):
        def __init__(self):
            super(Net_ms, self).__init__()
            self.fc = mindspore.nn.Linear(10, 1)

        def construct(self, x):
            nonzero_indices = mindspore.mint.nonzero(x)
            mask = mindspore.ops.ZerosLike()(x)
            for index in nonzero_indices:
                mask[index[0], index[1]] = 1
            out = x * mask
            return out.sum()
        
    class Net_pt(torch.nn.Module):
        def __init__(self):
            super(Net_pt, self).__init__()
        def forward(self, x):
            nonzero_indices = torch.nonzero(x)
            mask = torch.zeros_like(x)
            mask[nonzero_indices[:, 0], nonzero_indices[:, 1]] = 1
            out = x * mask
            return out.sum()


    input_value = np.random.random(size=(2, 10)).astype(np.float32)

    input_ms = mindspore.Tensor(input_value, mindspore.float32)
    grad_ms = mindspore.grad(Net_ms())(input_ms)

    input_pt = torch.from_numpy(input_value)
    input_pt.requires_grad = True
    result_pt = Net_pt()(input_pt)
    result_pt.backward()
    grad_pt = input_pt.grad

    assert np.allclose(grad_ms.asnumpy(), grad_pt.numpy(), atol=1e-3)



