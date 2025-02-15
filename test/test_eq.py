import torch
import mindspore
import numpy as np
import pytest

'''
    测试：
    mindspore.mint.eq(input, other)
    逐元素比较两个输入Tensor是否相等。
    第二个输入可以是一个shape可以广播成第一个输入的Number或Tensor， 反之亦然。

    input (Union[Tensor, Number]) - 第一个输入可以是数值型，也可以是数据类型为数值型的Tensor。
    other (Union[Tensor, Number]) - 当第一个输入是Tensor时，第二个输入是数值型或数据类型为数值型的Tensor，数据类型与第一个输入相同。
    当第一个输入是数值型时，第二个输入应为Tensor。
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
def test_eq_random_input_dtype(dtype):
    """
    测试random输入不同dtype，对比MindSpore和Pytorch的支持度
    """
    flag1 = True
    flag2 = True
    shape = (4, 4)
    try:
        # MindSpore
        input_ms = mindspore.Tensor(np.random.random(size=shape).astype(dtype))
        result_ms = mindspore.mint.eq(input_ms,input_ms)
        print(result_ms)
        assert isinstance(result_ms, mindspore.Tensor)
    except Exception as e:
        print(f"MindSpore出现报错: {e}")
        flag1 = False

    try:
        # Pytorch
        input_pt = torch.from_numpy(np.random.random(size=shape).astype(dtype))
        result_pt = torch.eq(input_pt,input_pt)
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
    {"input":np.random.random(size=(4,4)), "other":np.random.random(size=(4,4))},
    {"input":np.random.random(size=(4,4)), "other":np.random.random(size=(1))},
    {"input":np.random.random(size=(1)), "other":np.random.random(size=(4,4))},
    {"input":np.random.random(size=(4,4)), "other":np.random.random(size=(4,4,4))},
    {"input":np.random.random(size=(4,4,4)), "other":np.random.random(size=(4,4))},
])
def test_eq_fixed_dtype_random_value(input):
    """
    测试固定dtype，random输入值，对比两个框架输出（误差范围小于1e-3）
    """


    # MindSpore部分
    input_ms = mindspore.Tensor(input["input"])
    other_ms = mindspore.Tensor(input["other"])
    result_ms = mindspore.mint.eq(input_ms,other_ms)

    # Pytorch部分
    input_pt = torch.from_numpy(input["input"])
    other_ms = torch.from_numpy(input["other"])
    result_pt = torch.eq(input_pt,other_ms)

    assert np.allclose(result_ms.asnumpy(), result_pt.numpy(), atol=1e-3)



@pytest.mark.parametrize("input_param", [
    
])
def test_eq_fixed_shape_fixed_value_different_params(input_param):
    """
    测试固定shape，固定输入值，不同输入参数，两个框架的支持度
    """
    pass


@pytest.mark.parametrize("random_messy_input", [
    {"input":np.random.random(size=[4,4]), "other":np.random.random(size=[4]), "error":TypeError},
    {"input":mindspore.Tensor(np.random.random(size=[4,4])), "other":mindspore.Tensor(np.random.random(size=(3))), "error":ValueError}
])
def test_eq_random_messy_input_error_info(random_messy_input):
    """
    测试随机混乱输入，报错信息的准确性
    TypeError - input 和 other 都不是Tensor。
    """
    flag = False
    input_ms = random_messy_input["input"]
    other_ms = random_messy_input["other"]
    try:
        result_ms = mindspore.mint.eq(input_ms,other_ms)
        print(result_ms)
    except Exception as e_ms:
        assert isinstance(e_ms, random_messy_input["error"])
        flag = True
    if not flag:
        pytest.fail("在预期应捕获异常的情况下，未捕获到任何异常，测试不通过")





def test_eq_in_neural_network():
    """
    测试包含eq操作的简单网络示例
    """
    input_value = np.random.random(size=(4))

    class SimpleNet_pt(torch.nn.Module):
        def __init__(self):
            super(SimpleNet_pt, self).__init__()
        def forward(self, x, y):
            r = torch.eq(x,y)
            return r.sum()
        
    class SimpleNet_ms(mindspore.nn.Cell):
        def __init__(self):
            super(SimpleNet_ms, self).__init__()
        def construct(self, x, y):
            r = mindspore.mint.eq(x,y)
            return r.sum()
        
    input_ms = mindspore.Tensor(input_value)
    net_ms = SimpleNet_ms()
    result_ms = net_ms(input_ms,input_ms)


    input_pt = torch.from_numpy(input_value)
    net_pt = SimpleNet_pt()
    result_pt = net_pt(input_pt,input_pt)

    assert np.allclose(result_ms.asnumpy(), result_pt.detach().numpy(), atol=1e-3)


def test_eq_backward():
    """
    测试函数反向 
    """
    pass




