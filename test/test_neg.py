import mindspore as ms
import torch
import numpy as np
import pytest
from mindspore import ops
from mindspore import value_and_grad

ms_int_lst = [ms.int8, ms.int16, ms.int32, ms.int64]
ms_uint_lst = [ms.uint8, ms.uint16, ms.uint32, ms.uint64]
ms_float_lst = [ms.float16, ms.float32, ms.float64, ms.bfloat16]
ms_complex_lst = [ms.complex64, ms.complex128]

pt_int_lst = [torch.int8, torch.int16, torch.int32, torch.int64]
pt_uint_lst = [torch.uint8, torch.uint16, torch.uint32, torch.uint64]
pt_float_lst = [torch.float32, torch.float64, torch.float16, torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e4m3fnuz, torch.float8_e5m2fnuz]
pt_complex_lst = [torch.complex32, torch.complex64, torch.complex128]

@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_dtype(mode):
    """
    (1a-1) 遍历全部类型的张量，判断是否可用
    """
    ms.set_context(mode=mode, device_target='Ascend')
    
    # mindspore框架
    print("Mindspore框架")
    for dtype in ms_int_lst:
        input_ms = ms.tensor([[1, -2, 3, 1], [2, -3, 1, 4]], dtype=dtype)
        try:
            output_ms = ms.mint.neg(input_ms)
            print(dtype, output_ms)
        except Exception as e:
            print(f"mint.neg function does not support dtype: {dtype}")
    
    for dtype in ms_uint_lst:
        input_ms = ms.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=dtype)
        try:
            output_ms = ms.mint.neg(input_ms)
            print(dtype, output_ms)
        except Exception as e:
            print(f"mint.neg function does not support dtype: {dtype}")
    
    for dtype in ms_float_lst:
        input_ms = ms.tensor([[1.1, 1.2, 1.3, 1.4], [1.5, 1.6, 1.7, 1.8]], dtype=dtype)
        try:
            output_ms = ms.mint.neg(input_ms)
            print(dtype, output_ms)
        except Exception as e:
            print(f"mint.neg function does not support dtype: {dtype}")
        
    for dtype in ms_complex_lst:
        input_ms = ms.tensor([[1 + 1j, 1 + 2j, 1 + 3j], [1 + 4j, 1 + 5j, 1 + 6j]], dtype=dtype)
        try:
            output_ms = ms.mint.neg(input_ms)
            print(dtype, output_ms)
        except Exception as e:
            print(f"mint.neg function does not support dtype: {dtype}")

    input_ms = ms.tensor([[True, False, False], [False, True, True]], dtype=ms.bool_)
    try:
        output_ms = ms.mint.neg(input_ms)
        print(ms.bool_, output_ms)
    except Exception as e:
        print(f"mint.neg function does not support dtype: {ms.bool_}")
        
    # pytorch框架
    print("Pytorch框架")
    for dtype in pt_int_lst:
        input_pt = torch.tensor([[1, -2, 3, 1], [2, -3, 1, 4]], dtype=dtype)
        try:
            output_pt = torch.neg(input_pt)
            print(dtype, output_pt)
        except Exception as e:
            print(f"torch.neg function does not support dtype: {dtype}")
    
    for dtype in pt_uint_lst:
        input_pt = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=dtype)
        try:
            output_pt = torch.neg(input_pt)
            print(dtype, output_pt)
        except Exception as e:
            print(f"torch.neg function does not support dtype: {dtype}")
    
    for dtype in pt_float_lst:
        input_pt = torch.tensor([[1.1, 1.2, 1.3, 1.4], [1.5, 1.6, 1.7, 1.8]], dtype=dtype)
        try:
            output_pt = torch.neg(input_pt)
            print(dtype, output_pt)
        except Exception as e:
            print(f"torch.neg function does not support dtype: {dtype}")
    
    for dtype in pt_complex_lst:
        input_pt = torch.tensor([[1 + 1j, 1 + 2j, 1 + 3j], [1 + 4j, 1 + 5j, 1 + 6j]], dtype=dtype)
        try:
            output_pt = torch.neg(input_pt)
            print(dtype, output_pt)
        except Exception as e:
            print(f"torch.neg function does not support dtype: {dtype}")
            
    input_pt = torch.tensor([[True, False, False], [False, True, True]], dtype=torch.bool)
    try:
        output_pt = torch.neg(input_pt)
        print(torch.bool, output_pt)
    except Exception as e:
            print(f"torch.neg function does not support dtype: {torch.bool}")


@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('ms_type,pt_type', [[ms.int8, torch.int8], [ms.int32, torch.int32], [ms.int64, torch.int64], 
                                             [ms.float16, torch.float16], [ms.float32, torch.float32], [ms.float64, torch.float64], [ms.bfloat16, torch.bfloat16],
                                            [ms.complex64, torch.complex64], [ms.complex128, torch.complex128]])
def test_dtype_diff(mode, ms_type, pt_type):
    """
    (1a-2) 对于mindspore和pytorch均支持的数据类型进行遍历，测试输出差异
    """
    ms.set_context(mode=mode, device_target='Ascend')
    print(ms_type, pt_type)
    if (ms_type in ms_int_lst) and (pt_type in pt_int_lst):
        input_ms = ms.tensor([[1, -2, 3, 1], [2, -3, 1, 4]], dtype=ms_type)
        output_ms = ms.mint.neg(input_ms)
        
        input_pt = torch.tensor([[1, -2, 3, 1], [2, -3, 1, 4]], dtype=pt_type)
        output_pt = torch.neg(input_pt)
        assert np.allclose(output_ms.numpy(), output_pt.numpy(), rtol=1e-3)
    
    if (ms_type in ms_uint_lst) and (pt_type in pt_uint_lst):
        input_ms = ms.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=ms_type)
        output_ms = ms.mint.neg(input_ms)
        
        input_pt = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=pt_type)
        output_pt = torch.neg(input_pt)
        assert np.allclose(output_ms.numpy(), output_pt.numpy(), rtol=1e-3)
    
    if (ms_type in ms_float_lst) and (pt_type in pt_float_lst):
        input_ms = ms.tensor([[1.1, 1.2, 1.3, 1.4], [1.5, 1.6, 1.7, 1.8]], dtype=ms_type)
        output_ms = ms.mint.neg(input_ms)
        
        input_pt = torch.tensor([[1.1, 1.2, 1.3, 1.4], [1.5, 1.6, 1.7, 1.8]], dtype=pt_type)
        output_pt = torch.neg(input_pt)
        
        if ms_type == ms.bfloat16 and pt_type == torch.bfloat16:
            # 由于numpy不支持bfloat16类型，因此需要进行张量类型转换为float32后再比较差异
            cast_op = ops.Cast()
            output_ms = cast_op(output_ms, ms.float32)
            output_pt = output_pt.to(dtype=torch.float32)
            
        assert np.allclose(output_ms.numpy(), output_pt.numpy(), rtol=1e-3)
    
    if (ms_type in ms_complex_lst) and (pt_type in pt_complex_lst):
        input_ms = ms.tensor([[1 + 1j, 1 + 2j, 1 + 3j], [1 + 4j, 1 + 5j, 1 + 6j]], dtype=ms_type)
        output_ms = ms.mint.neg(input_ms)
        
        input_pt = torch.tensor([[1 + 1j, 1 + 2j, 1 + 3j], [1 + 4j, 1 + 5j, 1 + 6j]], dtype=pt_type)
        output_pt = torch.neg(input_pt)
        assert np.allclose(output_ms.numpy(), output_pt.numpy(), rtol=1e-3)
        
    if (ms_type == ms.bool_) and (pt_type == torch.bool):
        input_ms = ms.tensor([[True, False, False], [False, True, True]], dtype=ms.bool_)
        output_ms = ms.mint.neg(input_ms)
        
        input_pt = torch.tensor([[True, False, False], [False, True, True]], dtype=torch.bool)
        output_pt = torch.neg(input_pt)
        assert np.allclose(output_ms.numpy(), output_pt.numpy(), rtol=1e-3)
        
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_random_tensor(mode):
    """
    (1b) 固定dtype=mindspore.float64, 使用随机输入值，测试两个框架的输出是否相等
    """
    ms.set_context(mode=mode, device_target='Ascend')
    input_pt = torch.randn(size=(3, 4), dtype=torch.float64) * 5
    input_ms = ms.tensor(input_pt.numpy(), dtype=ms.float64)
    
    output_pt = torch.neg(input_pt)
    output_ms = ms.mint.neg(input_ms) 
    assert np.allclose(output_ms.numpy(), output_pt.numpy(), rtol=1e-3)
    
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_chaotic_input(mode):
    """
    (1d) 测试混乱输入
    """
    ms.set_context(mode=mode, device_target='Ascend')
    # 测试输入numpy数组
    reported_flag = False
    try:
        output_ms = ms.mint.neg(np.array([[1.1, 1.2, 1.3], [1.4, 1.5, 1.6]], dtype=np.float64))
    except Exception as e:
        reported_flag = True
        e = str(e)
        assert ("numpy.ndarray" in e) and ("Tensor" in e)
    
    if reported_flag == False:
        assert "No error message was reported when the input is numpy array" == 1
    
    # 测试输入列表
    reported_flag = False
    try:
        output_ms = ms.mint.neg([[1.1, 1.2, 1.3], [1.4, 1.5, 1.6]])
    except Exception as e:
        reported_flag = True
        e = str(e)
        assert ("List" in e) and ("Tensor" in e)
    
    if reported_flag == False:
        assert "No error message was reported when the input is list" == 1
        
    # 测试输入元组
    reported_flag = False
    try:
        output_ms = ms.mint.neg((1, 2, 3))
    except Exception as e:
        reported_flag = True
        e = str(e)
        assert ("Tuple" in e) and ("Tensor" in e)
    
    if reported_flag == False:
        assert "No error message was reported when the input is tuple" == 1

@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_forward_backward(mode):
    """
    (2) 测试两个框架的正向推理结果和反向梯度精度差异
    """
    ms.set_context(mode=mode, device_target='Ascend')
    input_pt = torch.tensor([[1.1, 1.2, 1.3, 1.4], [1.5, 1.6, 1.7, 1.8]], dtype=torch.float64, requires_grad=True)
    input_ms = ms.tensor([[1.1, 1.2, 1.3, 1.4], [1.5, 1.6, 1.7, 1.8]], dtype=ms.float64)
    
    def func_pt(x):
        y = torch.neg(x) * torch.neg(x) + torch.neg(x)
        return y.sum()
    
    def func_ms(x):
        y = ms.mint.neg(x) * ms.mint.neg(x) + ms.mint.neg(x)
        return y.sum()
    
    output_pt = func_pt(input_pt)
    output_pt.backward()
    gradient_pt = input_pt.grad
    
    grad_func = value_and_grad(func_ms)
    output_ms, gradient_ms = grad_func(input_ms)
    
    # print("output:", output_pt, output_ms)
    # print("gradient:", gradient_pt, gradient_ms)
    
    assert np.allclose(output_pt.detach().numpy(), output_ms.numpy(), rtol=1e-3)
    assert np.allclose(gradient_pt.numpy(), gradient_ms.numpy(), rtol=1e-3)

pytest.main(['-vs', 'test_neg.py', '--html', './report/report.html'])