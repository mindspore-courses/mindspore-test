import pytest
import numpy as np
import mindspore
import torch


"""
mindspore.mint.tile(input, dims)
    复制 dims 次 input 来创建新的Tensor。输出Tensor的第i维度有 input.shape[i] * dims[i] 个元素，并且 input 的值沿第i维度被复制 dims[i] 次。

"""
#用于比较MindSpore和PyTorch结果
def compare_results_ms_pt(result_ms, result_pt, atol=1e-3):
    try:
        assert np.allclose(result_ms.asnumpy(), result_pt.detach().numpy(), atol=atol)
    except AssertionError:
        pytest.fail("MindSpore和PyTorch结果超出误差范围，不相等")



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
def test_tile_random_dtype(dtype):
    """
    测试在随机输入不同数据类型时，MindSpore和PyTorch的tile接口的支持度。
    参数：
    - dtype：输入张量的数据类型，遍历多种常见类型进行测试。
    """
    mindspore.set_context(mode=mindspore.PYNATIVE_MODE, pynative_synchronize=True)
    flag1 = True
    flag2 = True
    input_shape = (2, 3) 
    input_data = np.random.rand(*input_shape).astype(dtype)
    multiples = (2, 2)
    try:

        # MindSpore部分
        input_ms = mindspore.Tensor(input_data)  
        result_ms = mindspore.mint.tile(input_ms, multiples)
        assert isinstance(result_ms, mindspore.Tensor)
    except Exception as e:
        print(f"MindSpore出现报错: {e}")
        flag1 = False
    try:
        # PyTorch部分
        input_pt = torch.from_numpy(input_data)
        result_pt = torch.tile(input_pt, multiples)

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
    (3, 4), (2, 2, 2)
])
def test_tile_fixed_dtype_random_value(input_shape):
    """
    测试在固定数据类型，随机输入值的情况下，MindSpore和PyTorch的tile接口输出是否相等（在误差范围内）。
    不同形状进行测试。
    """
    dtype = np.float32
    input_data = np.random.rand(*input_shape).astype(dtype)
    multiples = (2, 3)  
    
    # MindSpore部分
    input_ms = mindspore.Tensor(input_data)
    result_ms = mindspore.mint.tile(input_ms, multiples)

    # PyTorch部分
    input_pt = torch.from_numpy(input_data)
    result_pt = torch.tile(input_pt, multiples)

    compare_results_ms_pt(result_ms, result_pt)



# @pytest.mark.parametrize("input_value", [
#     "test_string", True, False
# ])
# def test_tile_fixed_shape_fixed_value(input_value):

#     input_shape = ()  # 零维情况示例，对应输入单个值
#     try:
#         # MindSpore部分
#         input_ms = mindspore.Tensor(input_value) if isinstance(input_value, (int, float, bool, str)) else input_value
#         multiples = (2,)  # 设定重复倍数示例
#         result_ms = mindspore.mint.tile(input_ms, multiples)

#         # PyTorch部分
#         input_pt = torch.tensor(input_value) if isinstance(input_value, (int, float, bool, str)) else input_value
#         result_pt = torch.tile(input_pt, multiples)

#         assert isinstance(result_ms, mindspore.Tensor) and isinstance(result_pt, torch.Tensor)
#     except Exception as e:
#         pytest.fail(f"输入值 {input_value} 时执行tile操作出现异常，异常信息: {str(e)}")



@pytest.mark.parametrize("random_messy_input", [
    (np.array([[1, 2], [3, 4]]).astype(np.float32), 2, TypeError),  
    (np.array([1, 2]), (4.0,), TypeError),  
    (np.array([1, 2]), (-1,), ValueError) 
])
def test_tile_random_messy_input_error_info(random_messy_input):
    """
    测试在随机混乱输入情况下，tile接口报错信息的准确性。
    TypeError - dims 不是tuple或者其元素并非全部是int。
    ValueError - dims 的元素并非全部大于或等于0
    """
    flag = False
    input_data = random_messy_input[0]
    multiples = random_messy_input[1]
    try:
        # MindSpore部分
        input_ms = mindspore.Tensor(input_data)
        result_ms = mindspore.mint.tile(input_ms, multiples)
        print(result_ms)
    except Exception as e_ms:
        assert isinstance(e_ms,random_messy_input[2])
        flag = True
    if not flag:
        pytest.fail("在预期应捕获异常的情况下，未捕获到任何异常，测试不通过")
        

                          
                          
# 以下测试使用该接口的构造函数/神经网络的准确性


def test_tile_in_neural_network():
    """
    测试一个包含tile操作的网络示例
    """
    input_value = np.random.random(size=(4, 4)).astype(np.float32)

    # mindspore
    class SimpleNet_ms(mindspore.nn.Cell):
        def __init__(self):
            super(SimpleNet_ms, self).__init__()

        def construct(self, x):
            split_result = mindspore.mint.tile(x, (2,2))
            return split_result.sum()

    
    input_ms = mindspore.Tensor(input_value)
    net_ms = SimpleNet_ms()
    result_ms = net_ms(input_ms)
    
    # pytorch
    class SimpleNet_pt(torch.nn.Module):
        def __init__(self):
            super(SimpleNet_pt, self).__init__()

        def forward(self, x):
            split_result = torch.tile(x, (2,2))
            return split_result.sum()

    input_pt = torch.from_numpy(input_value)
    net_pt = SimpleNet_pt()
    result_pt = net_pt(input_pt)

    compare_results_ms_pt(result_ms, result_pt)


def test_tile_backward():
    """
    测试函数反向 
    """
    # 定义简单函数
    def function_ms(x):
        return mindspore.mint.tile(x,(2,1)).sum()
    def function_pt(x):
        return torch.tile(x,(2,1)).sum()

    input_value = np.random.rand(4, 4).astype(np.float32)

    input_ms = mindspore.Tensor(input_value, mindspore.float32)
    grad_ms = mindspore.grad(function_ms)(input_ms)

    input_pt = torch.from_numpy(input_value).float().requires_grad_(True)
    result_pt = function_pt(input_pt)
    result_pt.backward()
    grad_pt = input_pt.grad

    compare_results_ms_pt(grad_ms, grad_pt)

