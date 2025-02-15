import torch
import mindspore
import numpy as np
import pytest

'''
    测试：
    mindspore.mint.prod(input, dim=None, keepdim=False, *, dtype=None)
    默认情况下，使用指定维度的所有元素的乘积代替该维度的其他元素，以移除该维度。也可仅缩小该维度大小至1。 keepdim 控制输出和输入的维度是否相同。
'''

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
def test_prod_random_input_dtype(dtype):
    """
    测试random输入不同dtype，对比MindSpore和Pytorch的支持度
    """
    flag1 = True
    flag2 = True
    shape = (4, 4)
    try:
        # MindSpore
        input_ms = mindspore.Tensor(np.random.random(size=shape).astype(dtype))
        result_ms = mindspore.mint.prod(input_ms)
        print(result_ms)
        assert isinstance(result_ms, mindspore.Tensor) and result_ms.ndim == 0
    except Exception as e:
        print(f"MindSpore出现报错: {e}")
        flag1 = False

    try:
        # Pytorch
        input_pt = torch.from_numpy(np.random.random(size=shape).astype(dtype))
        result_pt = torch.prod(input_pt)
        assert isinstance(result_pt, torch.Tensor) and result_pt.ndim == 0

    except Exception as e:
        print(f"Pytorch出现报错: {e}")
        flag2 = False

    if not flag1 and flag2:
        pytest.fail("mindspore不支持："+str(dtype))
    if flag1 and not flag2:
        pytest.fail("pytorch不支持："+str(dtype))
    if not flag1 and not flag2:
        pytest.fail("both mindspore 和 pytorch不支持："+str(dtype))



@pytest.mark.parametrize("input", [
    {"shape":[4,4], "dim":None, "keepdim":False},
    {"shape":[4,4], "dim":0, "keepdim":False},
    {"shape":[4,4], "dim":1, "keepdim":False},
    {"shape":[4,4,4,8], "dim":None, "keepdim":False},
    {"shape":[4,4], "dim":1, "keepdim":True}
])
def test_prod_fixed_dtype_random_value(input):
    """
    测试固定dtype，random输入值，对比两个框架输出（误差范围小于1e-3）
    """
    input_value = np.random.random(size=input["shape"])

    # MindSpore部分
    input_ms = mindspore.Tensor(input_value)
    result_ms = mindspore.mint.prod(input_ms,dim=input["dim"],keepdim=input["keepdim"])

    # Pytorch部分
    input_pt = torch.from_numpy(input_value)
    if input["dim"] is not None:
        result_pt = torch.prod(input_pt,dim=input["dim"],keepdim=input["keepdim"])
    else:
        result_pt = torch.prod(input_pt)


    assert np.allclose(result_ms.asnumpy(), result_pt.numpy(), atol=1e-3)

@pytest.mark.parametrize("input_param", [
    0
])
def test_prod_fixed_shape_fixed_value_different_params(input_param):
    """
    测试固定shape，固定输入值，不同输入参数，两个框架的支持度
    """
    input_value = np.random.random(size=[4,4])
    flag = True
    try:
        # MindSpore部分
        input_ms = mindspore.Tensor(input_value)
        result_ms = mindspore.mint.prod(input_ms, dim=input_param)

    except Exception as e:
        print(f"MindSpore出现报错: {e}")
        flag = False
    try:
        # Pytorch部分
        input_pt = torch.from_numpy(input_value).refine_names('a', 'b')
        result_pt = torch.prod(input_pt,dim=input_param)   
       
    except Exception as e:
        print(f"Pytorch出现报错: {e}")
        flag = False
    assert flag


@pytest.mark.parametrize("random_messy_input", [
    (np.array([[1, 2], [3, 4]]), 0, False, TypeError),
    (mindspore.tensor([[1, 2], [3, 4]]), "str", False, TypeError),
    (mindspore.tensor([[1, 2], [3, 4]]), 0, 1, TypeError),
    (mindspore.tensor([[1, 2], [3, 4]]), -3,False, ValueError),
    (mindspore.tensor([[1, 2], [3, 4]]), 2,False, ValueError)
])
def test_prod_random_messy_input_error_info(random_messy_input):
    """
    测试随机混乱输入，报错信息的准确性
    TypeError - input 不是Tensor。
    TypeError - dim 不是int。
    TypeError - keepdim 不是bool类型。
    ValueError - dim 超出范围。
    """
    flag = False
    input = random_messy_input[0]
    dim = random_messy_input[1]
    keepdim = random_messy_input[2]
    try:
        result_ms = mindspore.mint.prod(input, dim=dim, keepdim=keepdim)
        print(result_ms)
    except Exception as e_ms:
        assert isinstance(e_ms, random_messy_input[-1])
        flag = True
    if not flag:
        pytest.fail("在预期应捕获异常的情况下，未捕获到任何异常，测试不通过")





def test_prod_in_neural_network():
    """
    测试包含prod操作的网络示例
    """
    input_value = np.random.random(size=(32)).astype(np.float32)

    class ProductPooling_pt(torch.nn.Module):
        def __init__(self):
            super(ProductPooling_pt, self).__init__()

        def forward(self, x):
            y = x.reshape((8,4))
            pooled = torch.prod(y, dim=-1)
            return pooled.sum(dim=-1)
        
    class ProductPooling_ms(mindspore.nn.Cell):
        def __init__(self):
            super(ProductPooling_ms, self).__init__()

        def construct(self, x):
            y = x.reshape((8,4))
            pooled = mindspore.mint.prod(y, dim=-1)
            return pooled.sum(axis=-1)
        
    input_ms = mindspore.Tensor(input_value)
    net_ms = ProductPooling_ms()
    result_ms = net_ms(input_ms)


    input_pt = torch.from_numpy(input_value)
    net_pt = ProductPooling_pt()
    result_pt = net_pt(input_pt)

    assert np.allclose(result_ms.asnumpy(), result_pt.detach().numpy(), atol=1e-3)


def test_prod_backward():
    """
    测试函数反向 
    """
    class ProductPooling_pt(torch.nn.Module):
        def __init__(self):
            super(ProductPooling_pt, self).__init__()

        def forward(self, x):
            y = x.reshape((8,4))
            pooled = torch.prod(y, dim=-1)
            return pooled.sum(dim=-1)
        
    class ProductPooling_ms(mindspore.nn.Cell):
        def __init__(self):
            super(ProductPooling_ms, self).__init__()

        def construct(self, x):
            y = x.reshape((8,4))
            pooled = mindspore.mint.prod(y, dim=-1)
            return pooled.sum(axis=-1)


    input_value = np.random.random(size=(32)).astype(np.float32)

    input_ms = mindspore.Tensor(input_value, mindspore.float32)
    grad_ms = mindspore.grad(ProductPooling_ms())(input_ms)

    input_pt = torch.from_numpy(input_value)
    input_pt.requires_grad = True
    result_pt = ProductPooling_pt()(input_pt)
    result_pt.backward()
    grad_pt = input_pt.grad

    assert np.allclose(grad_ms.asnumpy(), grad_pt.numpy(), atol=1e-3)




