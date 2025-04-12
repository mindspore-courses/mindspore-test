import torch
import mindspore as ms
import numpy as np
from mindspore import value_and_grad
import pytest

ms_int_lst = [ms.int8, ms.int16, ms.int32, ms.int64]
ms_uint_lst = [ms.uint8, ms.uint16, ms.uint32, ms.uint64]
ms_float_lst = [ms.float16, ms.float32, ms.float64, ms.bfloat16]
ms_complex_lst = [ms.complex64, ms.complex128]

pt_int_lst = [torch.int8, torch.int16, torch.int32, torch.int64]
pt_uint_lst = [torch.uint8, torch.uint16, torch.uint32, torch.uint64]
pt_float_lst = [torch.float32, torch.float64, torch.float16, torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e4m3fnuz, torch.float8_e5m2fnuz]
pt_complex_lst = [torch.complex32, torch.complex64, torch.complex128]


def test_random_tensor():
    """
    (1b) 固定dtype=mindspore.float32, 使用随机输入值，测试两个框架的输出是否相等
    """
    
    input_pt = torch.randn((3, 6), dtype=torch.float32)
    input_ms = ms.tensor(input_pt.numpy(), dtype=ms.float32)
    
    output_pt = torch.special.erfc(input_pt)
    output_ms = ms.mint.special.erfc(input_ms)
    
    print(output_pt, output_ms)
    assert np.allclose(output_pt.numpy(), output_ms.numpy(), rtol=1e-3)
    
def test_dtype():
    """
    (1a) 遍历全部类型的张量，判断是否可用
    """
    # mindspore 框架
    for dtype in ms_int_lst:
        input_ms = ms.tensor([[1, -2, 3], [5, -6, 7], [2, 4, 5]], dtype=dtype)
        try:
            output_ms = ms.mint.special.erfc(input_ms)
            print(dtype, output_ms)
        except Exception as e:
            print(f"The function does not support dtype:{dtype}")
    
    for dtype in ms_uint_lst:
        input_ms = ms.tensor([[1, 2, 3], [5, 6, 7], [2, 4, 5]], dtype=dtype)
        try:
            output_ms = ms.mint.special.erfc(input_ms)
            print(dtype, output_ms)
        except Exception as e:
            print(f"The function does not support dtype:{dtype}")
    
    for dtype in ms_float_lst:
        input_ms = ms.tensor([[1.3, 2.1, 3.7], [2.1, 13.2, 4.2], [4.2, 4.3, 14.3]], dtype=dtype)
        try:
            output_ms = ms.mint.special.erfc(input_ms)
            print(dtype, output_ms)
        except Exception as e:
            print(f"The function does not support dtype:{dtype}")
     
   
    for dtype in ms_complex_lst:
        input_ms = ms.tensor([[1 + 1j, 1 + 21j, 1+3j], [1 + 4.2j, 1 + 5j, 1 + 6.1j], [1 + 7j, 1 + 8.1j, 1 + 9.9j]], dtype=dtype)
        try:
            output_ms = ms.mint.special.erfc(input_ms)
            print(dtype, output_ms)
        except Exception as e:
            print(f"The function does not support dtype:{dtype}")
    
    input_ms = ms.tensor([[True, True, False], [False, True, False], [True, False, True]], dtype=ms.bool_)
    try:
        output_ms = ms.mint.special.erfc(input_ms)
        print(ms.bool_, output_ms)
    except Exception as e:
        print(f"The function does not support dtype:{ms.bool_}")
    
    # torch 框架
    for dtype in pt_int_lst:
        input_pt = torch.tensor([[1, -2, 3], [5, -6, 7], [2, 4, 5]], dtype=dtype)
        try:
            output_pt = torch.special.erfc(input_pt)
            print(dtype, output_pt) 
        except Exception as e:
            print(f"The function does not support dtype:{dtype}")
      
    for dtype in pt_uint_lst:
        input_pt = torch.tensor([[1, 2, 3], [5, 6, 7], [2, 4, 5]], dtype=dtype)
        try:
            output_pt = torch.special.erfc(input_pt)
            print(dtype, output_pt) 
        except Exception as e:
            print(f"The function does not support dtype:{dtype}")
        
    for dtype in pt_float_lst:
        input_pt = torch.tensor([[1.3, 2.1, 3.7], [2.1, 13.2, 4.2], [4.2, 4.3, 14.3]], dtype=dtype)
        try:
            output_pt = torch.special.erfc(input_pt)
            print(dtype, output_pt)
        except Exception as e:
            print(f"The function does not support dtype: {dtype}")
        
    for dtype in pt_complex_lst:
        input_pt = torch.tensor([[1 + 1j, 1 + 21j, 1 + 3j], [1 + 4.2j, 1 + 5j, 1 + 6.1j], [1 + 7j, 1 + 8.1j, 1 + 9.9j]], dtype=dtype)
        try:
            output_pt = torch.special.erfc(input_pt)
            print(dtype, output_pt) 
        except Exception as e:
            print(f"The function does not support dtype:{dtype}")
    
    input_pt = torch.tensor([[True, True, False], [False, True, False], [True, False, True]], dtype=torch.bool)
    try:
        output_pt = torch.special.erfc(input_pt)
        print(torch.bool, output_pt) 
    except Exception as e:
        print(f"The function does not support dtype:{torch.bool}")

@pytest.mark.parametrize('ms_type,pt_type', [[ms.int64, torch.int64], [ms.float16, torch.float16], [ms.float32, torch.float32], [ms.float64, torch.float64], [ms.bool_, torch.bool]])
def test_dtype_diff(ms_type, pt_type):
    if (ms_type in ms_int_lst) and (pt_type in pt_int_lst):
        input_ms = ms.tensor([[1, -2, 3, -4], [5, -6, 7, -8]], dtype=ms_type)
        output_ms = ms.mint.special.erfc(input_ms)
        
        input_pt = torch.tensor([[1, -2, 3, -4], [5, -6, 7, -8]], dtype=pt_type)
        output_pt = torch.special.erfc(input_pt)
        assert np.allclose(output_pt.numpy(), output_ms.numpy(), rtol=1e-3)
    
    if (ms_type in ms_float_lst) and (pt_type in pt_float_lst):
        input_ms = ms.tensor([[1.1, 1.2, 1.3, 1.4], [1.5, 1.6, 1.7, 1.8]], dtype=ms_type)
        output_ms = ms.mint.special.erfc(input_ms)
        
        input_pt = torch.tensor([[1.1, 1.2, 1.3, 1.4], [1.5, 1.6, 1.7, 1.8]], dtype=pt_type)
        output_pt = torch.special.erfc(input_pt)
        
        if ms_type == ms.bfloat16 and pt_type == torch.bfloat16:
            # 由于numpy不支持bfloat16类型，因此需要进行张量类型转换为float32后再比较差异
            cast_op = ops.Cast()
            output_ms = cast_op(output_ms, ms.float32)
            output_pt = output_pt.to(dtype=torch.float32)
            
        assert np.allclose(output_pt.numpy(), output_ms.numpy(), rtol=1e-3)
    
    if ms_type == ms.bool_ and pt_type == torch.bool:
        input_ms = ms.tensor([[True, True, False, False], [True, False, True, False]], dtype=ms.bool_)
        output_ms = ms.mint.special.erfc(input_ms)
        
        input_pt = torch.tensor([[True, True, False, False], [True, False, True, False]], dtype=torch.bool)
        output_pt = torch.special.erfc(input_pt)
        assert np.allclose(output_pt.numpy(), output_ms.numpy(), rtol=1e-3)
    
def test_random_input():
    """
    (1d) 测试混乱输入
    """
    # 测试输入不为张量的情况
    reported_flag = False
    try:
        output_ms = ms.mint.special.erfc([1.2, 1.3, 1.4, 1.5])
        print(output_ms)
    except Exception as e:
        reported_flag = True
        e = str(e)
        assert "List" in e and "Tensor" in e
    
    if reported_flag == False:
        assert "No error message was reported when the input is List" == 1
        
     # 测试输入为空的情况
    reported_flag = False
    try:
        output_ms = ms.mint.special.erfc(None)
        print(output_ms)
    except Exception as e:
        reported_flag = True
        e = str(e)
        assert "None" in e and "Tensor" in e
    
    if reported_flag == False:
        assert "No error message was reported when the input is None" == 1

def test_forward_backward():
    """
    (2) 测试两个框架的正向推理结果和反向梯度精度差异
    """
    data_lst = [[1.3, 2.1, 3.7], [2.1, 13.2, 4.2], [4.2, 4.3, 14.3]]
    input_pt = torch.tensor(data_lst, dtype=torch.float32, requires_grad=True)
    input_ms = ms.tensor(data_lst, dtype=ms.float32)
    
    def ms_func(ms_tensor):
        result_ms = ms.mint.special.erfc(ms_tensor)
        return result_ms.sum()
    
    def pt_func(pt_tensor):
        result_pt = torch.special.erfc(pt_tensor)
        return result_pt.sum()
    
    grad_func = value_and_grad(ms_func)
    output_ms, gradient_ms = grad_func(input_ms)
    output_pt = pt_func(input_pt)
    output_pt.backward()
    gradient_pt = input_pt.grad
    
    print(gradient_ms, gradient_pt)
    print(output_ms, output_pt)
    
    assert np.allclose(output_ms.numpy(), output_pt.detach().numpy(), rtol=1e-3)
    assert np.allclose(gradient_ms.numpy(), gradient_pt.numpy(), rtol=1e-3)
    
    
ms.set_context(device_target='Ascend')
pytest.main(['-vs', 'test_erfc.py', '--html', './report/report.html'])