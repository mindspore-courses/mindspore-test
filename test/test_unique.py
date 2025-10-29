import torch
import mindspore
import numpy as np
import pytest

'''
    测试：
    mindspore.mint.unique(input, sorted=True, return_inverse=False, return_counts=False, dim=None)[源代码]
    对输入Tensor中元素去重。
    在 return_inverse=True 时，会返回一个索引Tensor，包含输入Tensor中的元素在输出Tensor中的索引； 
    在 return_counts=True 时，会返回一个Tensor，表示输出元素在输入中的个数。

    input (Tensor) - 输入Tensor。
    sorted (bool) - 输出是否需要进行升序排序。默认值： True 。
    return_inverse (bool) - 是否输出 input 在 output 上对应的index。默认值： False 。
    return_counts (bool) - 是否输出 output 中元素的数量。默认值： False 。
    dim (int) - 做去重操作的维度，当设置为 None 的时候，对展开的输入做去重操作, 否则，将给定维度的Tensor视为一个元素去做去重操作。默认值：None 。
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
def test_unique_random_input_dtype(dtype):
    """
    测试random输入不同dtype，对比MindSpore和Pytorch的支持度
    """
    flag1 = True
    flag2 = True
    shape = (4, 4)
    try:
        # MindSpore
        input_ms = mindspore.Tensor(np.random.random(size=shape).astype(dtype))
        result_ms = mindspore.mint.unique(input_ms)
        print(result_ms)
        assert isinstance(result_ms, mindspore.Tensor)
    except Exception as e:
        print(f"MindSpore出现报错: {e}")
        flag1 = False

    try:
        # Pytorch
        input_pt = torch.from_numpy(np.random.random(size=shape).astype(dtype))
        result_pt = torch.unique(input_pt)
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



@pytest.mark.parametrize("input", [
    {"shape":[4,4], "sorted":False, "return_inverse":False, "return_counts":False, "dim":None},
    {"shape":[4,4], "sorted":True, "return_inverse":False, "return_counts":False, "dim":None},
    {"shape":[4,4], "sorted":False, "return_inverse":True, "return_counts":False, "dim":None},
    {"shape":[4,4], "sorted":False, "return_inverse":False, "return_counts":True, "dim":None},
    {"shape":[4,4], "sorted":False, "return_inverse":False, "return_counts":False, "dim":0}
])
def test_unique_fixed_dtype_random_value(input):
    """
    测试固定dtype，random输入值，对比两个框架输出（误差范围小于1e-3）
    """
    input_value = np.random.random(size=input["shape"])
    sorted = input["sorted"]
    return_inverse = input["return_inverse"]
    return_counts = input["return_counts"]
    dim = input["dim"]
    # MindSpore部分
    input_ms = mindspore.Tensor(input_value)
    result_ms = mindspore.mint.unique(input_ms,dim=dim,sorted=sorted,return_counts=return_counts,return_inverse=return_inverse)
    print(result_ms)
    # Pytorch部分
    input_pt = torch.from_numpy(input_value)
    result_pt = torch.unique(input_pt,dim=dim,sorted=sorted,return_counts=return_counts,return_inverse=return_inverse)
    print(result_pt)

    if return_inverse or return_counts:
        for result_ms_i,result_pt_i in zip(result_ms,result_pt):
            assert np.allclose(result_ms_i.asnumpy(), result_pt_i.numpy(), atol=1e-3)
    else:
        assert np.allclose(result_ms.asnumpy(), result_pt.numpy(), atol=1e-3)


@pytest.mark.parametrize("input_param", [

])
def test_unique_fixed_shape_fixed_value_different_params(input_param):
    """
    测试固定shape，固定输入值，不同输入参数，两个框架的支持度
    """
    pass


@pytest.mark.parametrize("random_messy_input", [
    {"input":np.random.random(), "other":np.random.random(), "error":TypeError},
    {"input":np.random.random(size=[4,4]), "other":np.random.random(size=[4]), "error":TypeError}
])
def test_unique_random_messy_input_error_info(random_messy_input):
    """
    TypeError - input 不是Tensor。
    """
    flag = False
    input = random_messy_input["input"]
    try:
        result_ms = mindspore.mint.unique(input)
        print(result_ms)
    except Exception as e_ms:
        assert isinstance(e_ms, random_messy_input["error"])
        flag = True
    if not flag:
        pytest.fail("在预期应捕获异常的情况下，未捕获到任何异常，测试不通过")





def test_unique_in_neural_network():
    """
    测试包含unique操作的网络示例
    """
    input_value = np.random.random(size=(32)).astype(np.float32)

    class SimpleNet_pt(torch.nn.Module):
        def __init__(self):
            super(SimpleNet_pt, self).__init__()

        def forward(self, x):
            unique_output = torch.unique(x)
            unique_elements = unique_output[0]
            sum_output = torch.sum(unique_elements)
            return sum_output
        
    class SimpleNet_ms(mindspore.nn.Cell):
        def __init__(self):
            super(SimpleNet_ms, self).__init__()

        def construct(self, x):
            unique_output = mindspore.mint.unique(x)
            unique_elements = unique_output[0]
            sum_output = unique_elements.sum()
            return sum_output
        
    input_ms = mindspore.Tensor(input_value)
    net_ms = SimpleNet_ms()
    result_ms = net_ms(input_ms)


    input_pt = torch.from_numpy(input_value)
    net_pt = SimpleNet_pt()
    result_pt = net_pt(input_pt)

    assert np.allclose(result_ms.asnumpy(), result_pt.detach().numpy(), atol=1e-3)


def test_unique_backward():
    """
    测试函数反向 
    """
    pass



