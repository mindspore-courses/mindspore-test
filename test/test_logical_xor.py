import torch
import mindspore as ms
import numpy as np
from mindspore import value_and_grad
import pytest
from mindspore import ops

ms_int_lst = [ms.int8, ms.int16, ms.int32, ms.int64]
ms_uint_lst = [ms.uint8, ms.uint16, ms.uint32, ms.uint64]
ms_float_lst = [ms.float16, ms.float32, ms.float64, ms.bfloat16]
ms_complex_lst = [ms.complex64, ms.complex128]


pt_int_lst = [torch.int8, torch.int16, torch.int32, torch.int64]
pt_uint_lst = [torch.uint8, torch.uint16, torch.uint32, torch.uint64]
pt_float_lst = [torch.float16, torch.float32, torch.float64, torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e4m3fnuz, torch.float8_e5m2fnuz]
pt_complex_lst = [torch.complex32, torch.complex64, torch.complex128]

@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_dtype(mode):
    """
    (1a-1) 遍历全部类型的张量，判断是否可用
    """
    ms.set_context(mode=mode, device_target='Ascend')
    
    # mindspore框架
    for dtype in ms_int_lst:
        input_ms1 = ms.tensor([[1, -2, 3, -4], [5, -6, 7, -8]], dtype=dtype)
        input_ms2 = ms.tensor([[1, 0, -1, 3], [2, 4, -3, 1]], dtype=dtype)
        try:
            output_ms = ms.mint.logical_xor(input_ms1, input_ms2)
            print(dtype, output_ms)
        except Exception as e:
            print(f"mint.logical_xor does not support dtype:{dtype}")


    for dtype in ms_uint_lst:
        input_ms1 = ms.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=dtype)
        input_ms2 = ms.tensor([[1, 2, 1, 4], [2, 4, 2, 1]], dtype=dtype)
        try:
            output_ms = ms.mint.logical_xor(input_ms1, input_ms2)
            print(dtype, output_ms)
        except Exception as e:
            print(f"mint.logical_xor does not support dtype:{dtype}")


    for dtype in ms_float_lst:
        input_ms1 = ms.tensor([[1.1, 1.2, 1.3, 1.4], [1.5, 1.6, 1.7, 1.8]], dtype=dtype)
        input_ms2 = ms.tensor([[2.3, 2.1, 4.3, 2.4], [4.5, 2.6, 2.4, 1.6]], dtype=dtype)
        try:
            output_ms = ms.mint.logical_xor(input_ms1, input_ms2)
            print(dtype, output_ms)
        except Exception as e:
            print(f"mint.logical_xor does not support dtype:{dtype}")


    for dtype in ms_complex_lst:
        input_ms1 = ms.tensor([[1 + 1j, 1 - 1j, 1 + 0.5j], [1 + 2j, 1 + 3j, 1 + 4j]], dtype=dtype)
        input_ms2 = ms.tensor([[2 + 1j, 1 - 3j, 2 + 0.5j], [1 + 6j, 3 + 6j, 2 + 4j]], dtype=dtype)
        try:
            output_ms = ms.mint.logical_xor(input_ms1, input_ms2)
            print(dtype, output_ms)
        except Exception as e:
            print(f"mint.logical_xor does not support dtype:{dtype}")


    input_ms1 = ms.tensor([[True, True, False, False], [True, False, True, False]], dtype=ms.bool_)
    input_ms2 = ms.tensor([[False, True, False, True], [True, True, True, False]], dtype=ms.bool_)
    try:
        output_ms = ms.mint.logical_xor(input_ms1, input_ms2)
        print(ms.bool_, output_ms)
    except Exception as e:
        print(f"mint.logical_xor does not support dtype:{ms.bool_}")

    # torch 框架

    for dtype in pt_int_lst:
        input_pt1 = torch.tensor([[1, -2, 3, -4], [5, -6, 7, -8]], dtype=dtype)
        input_pt2 = torch.tensor([[1, -1, 24, -5], [8, 2, 6, -3]], dtype=dtype)
        try:
            output_pt = torch.logical_xor(input_pt1, input_pt2)
            print(dtype, output_pt)
        except Exception as e:
            print(f"torch.logical_xor does not support dtype:{dtype}")


    for dtype in pt_uint_lst:
        input_pt1 = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=dtype)
        input_pt2 = torch.tensor([[2, 3, 1, 4], [4, 6, 8, 24]], dtype=dtype)
        try:
            output_pt = torch.logical_xor(input_pt1, input_pt2)
            print(dtype, output_pt)
        except Exception as e:
            print(f"torch.logical_xor does not support dtype:{dtype}")


    for dtype in pt_float_lst:
        input_pt1 = torch.tensor([[1.1, 1.2, 1.3, 1.4], [1.5, 1.6, 1.7, 1.8]], dtype=dtype)
        input_pt2 = torch.tensor([[1.3, 2.4, 2.7, 1.5], [1.3, 1.7, 2.3, 2.9]], dtype=dtype)
        try:
            output_pt = torch.logical_xor(input_pt1, input_pt2)
            print(dtype, output_pt)
        except Exception as e:
            print(f"torch.logical_xor does not support dtype:{dtype}")


    for dtype in pt_complex_lst:
        input_pt1 = torch.tensor([[1 + 1j, 1 - 1j, 1 + 0.5j], [1 + 2j, 1 + 3j, 1 + 4j]], dtype=dtype)
        input_pt2 = torch.tensor([[2 + 1j, 1 - 3j, 2 + 0.5j], [1 + 6j, 3 + 6j, 2 + 4j]], dtype=dtype)
        try:
            output_pt = torch.logical_xor(input_pt1, input_pt2)
            print(dtype, output_pt)
        except Exception as e:
            print(f"torch.logical_xor does not support dtype:{dtype}")

    input_pt1 = torch.tensor([[True, True, False, False], [True, False, True, False]], dtype=torch.bool)
    input_pt2 = torch.tensor([[False, True, False, True], [True, True, True, False]], dtype=torch.bool)
    try:
        output_pt = torch.logical_xor(input_pt1, input_pt2)
        print(torch.bool, output_pt)
    except Exception as e:
        print(f"torch.logical_xor does not support dtype:{torch.bool}")

@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('ms_type,pt_type', [[ms.int8, torch.int8], [ms.int16, torch.int16], [ms.int32, torch.int32], [ms.int64, torch.int64], [ms.uint8, torch.uint8],
                                            [ms.float16, torch.float16], [ms.float32, torch.float32], [ms.float64, torch.float64], [ms.bfloat16, torch.bfloat16],
                                            [ms.complex64, torch.complex64], [ms.complex128, torch.complex128], [ms.bool_, torch.bool]])
def test_dtype_diff(mode, ms_type, pt_type):
    """
    (1a-2) 对于mindspore和pytorch均支持的数据类型进行遍历，测试输出差异
    """
    ms.set_context(mode=mode, device_target='Ascend')
    print(ms_type, pt_type)
    
    if (ms_type in ms_int_lst) and (pt_type in pt_int_lst):
        input_ms1 = ms.tensor([[1, -2, 3, -4], [5, -6, 7, -8]], dtype=ms_type)
        input_ms2 = ms.tensor([[1, 0, -1, 3], [2, 4, -3, 1]], dtype=ms_type)
        output_ms = ms.mint.logical_xor(input_ms1, input_ms2)
        
        input_pt1 = torch.tensor([[1, -2, 3, -4], [5, -6, 7, -8]], dtype=pt_type)
        input_pt2 = torch.tensor([[1, 0, -1, 3], [2, 4, -3, 1]], dtype=pt_type)
        output_pt = torch.logical_xor(input_pt1, input_pt2)
        # print("output:", output_ms, output_pt)
        assert np.allclose(output_ms.numpy(), output_pt.numpy(), rtol=1e-3)
        
    if (ms_type in ms_uint_lst) and (pt_type in pt_uint_lst):
        input_ms1 = ms.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=ms_type)
        input_ms2 = ms.tensor([[1, 2, 1, 4], [2, 4, 2, 1]], dtype=ms_type)
        output_ms = ms.mint.logical_xor(input_ms1, input_ms2)
        
        input_pt1 = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=pt_type)
        input_pt2 = torch.tensor([[1, 2, 1, 4], [2, 4, 2, 1]], dtype=pt_type)
        output_pt = torch.logical_xor(input_pt1, input_pt2)
        # print("output:", output_ms, output_pt)
        assert np.allclose(output_ms.numpy(), output_pt.numpy(), rtol=1e-3)
    
    if (ms_type in ms_float_lst) and (pt_type in pt_float_lst):
        input_ms1 = ms.tensor([[1.1, 1.2, 1.3, 1.4], [1.5, 1.6, 1.7, 1.8]], dtype=ms_type)
        input_ms2 = ms.tensor([[2.3, 2.1, 4.3, 2.4], [4.5, 2.6, 2.4, 1.6]], dtype=ms_type)
        output_ms = ms.mint.logical_xor(input_ms1, input_ms2)
        
        input_pt1 = torch.tensor([[1.1, 1.2, 1.3, 1.4], [1.5, 1.6, 1.7, 1.8]], dtype=pt_type)
        input_pt2 = torch.tensor([[2.3, 2.1, 4.3, 2.4], [4.5, 2.6, 2.4, 1.6]], dtype=pt_type)
        output_pt = torch.logical_xor(input_pt1, input_pt2)
        # print("output:", output_ms, output_pt)
        
        if ms_type == ms.bfloat16 and pt_type == torch.bfloat16:
            # 由于numpy不支持bfloat16类型，因此需要进行张量类型转换为float32后再比较差异
            cast_op = ops.Cast()
            output_ms = cast_op(output_ms, ms.float32)
            output_pt = output_pt.to(dtype=torch.float32)
            
        assert np.allclose(output_ms.numpy(), output_pt.numpy(), rtol=1e-3)
    
    if (ms_type in ms_complex_lst) and (pt_type in pt_complex_lst):
        input_ms1 = ms.tensor([[1 + 1j, 1 - 1j, 1 + 0.5j], [1 + 2j, 1 + 3j, 1 + 4j]], dtype=ms_type)
        input_ms2 = ms.tensor([[2 + 1j, 1 - 3j, 2 + 0.5j], [1 + 6j, 3 + 6j, 2 + 4j]], dtype=ms_type)
        output_ms = ms.mint.logical_xor(input_ms1, input_ms2)
        
        input_pt1 = torch.tensor([[1 + 1j, 1 - 1j, 1 + 0.5j], [1 + 2j, 1 + 3j, 1 + 4j]], dtype=pt_type)
        input_pt2 = torch.tensor([[2 + 1j, 1 - 3j, 2 + 0.5j], [1 + 6j, 3 + 6j, 2 + 4j]], dtype=pt_type)
        output_pt = torch.logical_xor(input_pt1, input_pt2)
        # print("output:", output_ms, output_pt)
        assert np.allclose(output_ms.numpy(), output_pt.numpy(), rtol=1e-3)
    
    if (ms_type == ms.bool_) and (pt_type == torch.bool):
        input_ms1 = ms.tensor([[True, True, False, False], [True, False, True, False]], dtype=ms.bool_)
        input_ms2 = ms.tensor([[False, True, False, True], [True, True, True, False]], dtype=ms.bool_)
        output_ms = ms.mint.logical_xor(input_ms1, input_ms2)
        
        input_pt1 = torch.tensor([[True, True, False, False], [True, False, True, False]], dtype=torch.bool)
        input_pt2 = torch.tensor([[False, True, False, True], [True, True, True, False]], dtype=torch.bool)
        output_pt = torch.logical_xor(input_pt1, input_pt2)
        # print("output:", output_ms, output_pt)
        assert np.allclose(output_ms.numpy(), output_pt.numpy(), rtol=1e-3)

@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_random_tensor(mode):
    """
    (1b) 固定dtype=mindspore.float64, 使用随机输入值，测试两个框架的输出是否相等
    """
    ms.set_context(mode=mode, device_target='Ascend')
    input_pt1 = torch.randn(size=(3, 4), dtype=torch.float64) * 5
    input_pt2 = torch.randn(size=(3, 4), dtype=torch.float64) * 5
    
    input_ms1 = ms.tensor(input_pt1.numpy(), dtype=ms.float64)
    input_ms2 = ms.tensor(input_pt2.numpy(), dtype=ms.float64)
    
    output_pt = torch.logical_xor(input_pt1, input_pt2)
    output_ms = ms.mint.logical_xor(input_ms1, input_ms2)
    assert np.allclose(output_ms.numpy(), output_pt.numpy(), rtol=1e-3)

@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_chaotic_input(mode):
    """
    (1d) 测试混乱输入
    """
    ms.set_context(mode=mode, device_target='Ascend')
    
    # 测试输入为列表的情况
    reported_flag = False
    try:
        output_ms = ms.mint.logical_xor([False, True, False], [True, True, False])
        print(output_ms)
    except Exception as e:
        e = str(e)
        reported_flag = True
        assert 'List' in e and 'Tensor' in e
    
    if reported_flag == False:
        assert "No error message was reported when the input is list" == 1
        
    # 测试输入为元组的情况
    reported_flag = False
    try:
        output_ms = ms.mint.logical_xor((False, True, False), (True, True, False))
        print(output_ms)
    except Exception as e:
        e = str(e)
        reported_flag = True
        assert 'Tuple' in e and 'Tensor' in e
    
    if reported_flag == False:
        assert "No error message was reported when the input is tuple" == 1
    
    # 测试输入为numpy数组的情况
    reported_flag = False
    try:
        output_ms = ms.mint.logical_xor(np.array([False, True, False]), np.array([True, True, False]))
        print(output_ms)
    except Exception as e:
        e = str(e)
        reported_flag = True
        assert 'array' in e and 'Tensor' in e
    
    if reported_flag == False:
        assert "No error message was reported when the input is numpy array" == 1
    
        
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_broadcast(mode):
    """
    测试广播机制是否正常
    """
    ms.set_context(mode=mode, device_target='Ascend')
    
    input_ms1 = ms.tensor([[1], [2], [0]], dtype=ms.int64)
    input_ms2 = ms.tensor([[0, 1, 2]], dtype=ms.int64)
    output_ms = ms.mint.logical_xor(input_ms1, input_ms2)
    
    input_pt1 = torch.tensor([[1], [2], [0]], dtype=torch.int64)
    input_pt2 = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    output_pt = torch.logical_xor(input_pt1, input_pt2)
    
    print(output_ms, output_pt)
    assert np.allclose(output_ms.numpy(), output_pt.numpy(), rtol=1e-3)
    

@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_forward_backward(mode):
    """
    (2) 测试两个框架的正向推理结果和反向梯度精度差异
    """
    ms.set_context(mode=mode, device_target='Ascend')
    
    data_lst1 = [[1.1, 1.2, 1.3, 1.4], [1.5, 1.6, 1.7, 1.8]]
    data_lst2 = [[1.1, 0, 1.3, 0], [0, 1.6, 0, 1.8]]
    
    input_ms1 = ms.tensor(data_lst1, dtype=ms.float64)
    input_ms2 = ms.tensor(data_lst2, dtype=ms.float64)
    
    input_pt1 = torch.tensor(data_lst1, dtype=torch.float64, requires_grad=True)
    input_pt2 = torch.tensor(data_lst2, dtype=torch.float64, requires_grad=True)
    
    def func_pt(x, y):
        output1 = x * torch.logical_xor(x, y)
        return output1.sum()
    
    def func_ms(x, y):
        output1 = x * ms.mint.logical_xor(x, y)
        return output1.sum()
    
    output_pt = func_pt(input_pt1, input_pt2)
    output_pt.backward()
    gradient_pt1 = input_pt1.grad
    gradient_pt2 = input_pt2.grad
    
    grad_func = value_and_grad(func_ms, grad_position=(0,1))
    output_ms, gradient_ms = grad_func(input_ms1, input_ms2)
    gradient_ms1 = gradient_ms[0]
    gradient_ms2 = gradient_ms[1]
    
    # print("output:", output_pt, output_ms)
    # print("gradient", gradient_pt1, gradient_ms)
    assert np.allclose(output_ms.numpy(), output_pt.detach().numpy(), rtol=1e-3)
    assert np.allclose(gradient_pt1.numpy(), gradient_ms1.numpy(), rtol=1e-3)
    assert np.allclose(gradient_pt2.numpy(), gradient_ms2.numpy(), rtol=1e-3)
      
pytest.main(['-vs', 'test_logical_xor.py', '--html', './report/report.html'])