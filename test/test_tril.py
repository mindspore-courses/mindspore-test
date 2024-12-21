import pytest
import numpy as np
import mindspore
import torch


""""
mindspore.mint.tril(input, diagonal=0)
    返回输入Tensor input 的下三角形部分(包含对角线和下面的元素)，并将其他元素设置为0。
"""

# 辅助函数
def compare_results_ms_pt(result_ms, result_pt, atol=1e-3):
    """
    比较MindSpore和PyTorch的计算结果
    """
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
def test_tril_random_dtype(dtype):
    """
    测试在随机输入不同数据类型时，MindSpore和PyTorch的tril接口的支持度。
    """
    mindspore.set_context(mode=mindspore.PYNATIVE_MODE, pynative_synchronize=True)
    flag1 = True
    flag2 = True
    input_shape = (3, 3) 
    input_data = np.random.rand(*input_shape).astype(dtype)
    k = 0  # 对角线偏移
    try:
        # MindSpore部分
        input_ms = mindspore.Tensor(input_data)
        result_ms = mindspore.mint.tril(input_ms, k)
        assert isinstance(result_ms, mindspore.Tensor)

    except Exception as e:
        print(f"MindSpore出现报错: {e}")
        flag1 = False

    try:
        # PyTorch部分
        input_pt = torch.from_numpy(input_data)
        result_pt = torch.tril(input_pt, k)
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
    (4, 4), (2, 5)
])
def test_tril_fixed_dtype_random_value(input_shape):
    """
    测试在固定数据类型，随机输入值的情况下，MindSpore和PyTorch的tril接口输出是否相等。
    """
    dtype = np.float32
    input_data = np.random.rand(*input_shape).astype(dtype)
    k = -1  # 设定对角线偏移  
    
    # MindSpore部分
    input_ms = mindspore.Tensor(input_data)
    result_ms = mindspore.mint.tril(input_ms, k)

    # PyTorch部分
    input_pt = torch.from_numpy(input_data)
    result_pt = torch.tril(input_pt, k)

    compare_results_ms_pt(result_ms, result_pt)




@pytest.mark.parametrize("random_messy_input", [
    ([[1,2],[3,4]], 0, TypeError),
    (mindspore.Tensor([[1,2],[3,4]]), 0.0, TypeError),
    (mindspore.Tensor([1,2,3,4]), 0, ValueError)
])
def test_tril_random_messy_input_error_info(random_messy_input):
    """
    测试在随机混乱输入情况下，MindSpore的tril接口报错信息的准确性。
    TypeError - 如果 input 不是Tensor。
    TypeError - 如果 diagonal 不是int类型。
    TypeError - 如果 input 的数据类型既不是数值型也不是bool。
    ValueError - 如果 input 的秩小于2。
    """
    flag = False
    input_ms = random_messy_input[0]
    k = random_messy_input[1]
    try:
        
        result_ms = mindspore.mint.tril(input_ms, k)
        print(result_ms)
        
    except Exception as e_ms:
        assert isinstance(e_ms, random_messy_input[-1])
        flag = True
    if not flag:
        pytest.fail("在预期应捕获异常的情况下，未捕获到任何异常，测试不通过")


        
        
        
# 以下测试使用该接口的构造函数/神经网络的准确性




def test_tril_in_neural_network():
    """
    测试一个包含split操作的网络示例
    """
    input_value = np.random.rand(4, 4).astype(np.float32)

    # mindspore
    class SimpleNet_ms(mindspore.nn.Cell):
        def __init__(self):
            super(SimpleNet_ms, self).__init__()

        def construct(self, x):
            split_result = mindspore.mint.tril(x,0)
            return split_result.sum()

    
    input_ms = mindspore.Tensor(input_value)
    net_ms = SimpleNet_ms()
    result_ms = net_ms(input_ms)
    
    # pytorch
    class SimpleNet_pt(torch.nn.Module):
        def __init__(self):
            super(SimpleNet_pt, self).__init__()

        def forward(self, x):
            split_result = torch.tril(x, 0)
            return split_result.sum()

    input_pt = torch.from_numpy(input_value)
    net_pt = SimpleNet_pt()
    result_pt = net_pt(input_pt)

    assert np.allclose(result_ms.asnumpy(), result_pt.numpy(), atol=1e-3)


def test_tril_backward():
    """
    测试函数反向 
    """
    # 定义简单函数
    def function_ms(x):
        return mindspore.mint.tril(x,0).sum()
    def function_pt(x):
        return torch.tril(x,0).sum()

    input_value = np.random.rand(4, 4).astype(np.float32)

    input_ms = mindspore.Tensor(input_value, mindspore.float32)
    grad_ms = mindspore.grad(function_ms)(input_ms)

    input_pt = torch.from_numpy(input_value).float().requires_grad_(True)
    result_pt = function_pt(input_pt)
    result_pt.backward()
    grad_pt = input_pt.grad

    assert np.allclose(grad_ms.asnumpy(), grad_pt.numpy(), atol=1e-3)
