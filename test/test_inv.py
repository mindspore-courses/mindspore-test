import torch
import mindspore as ms
import numpy as np
from mindspore import value_and_grad
import pytest

def test_random_tensor():
    """
    (1b) 固定dtype=mindspore.float32, 使用随机输入值，测试两个框架的输出是否相等
    """
    input_pt = torch.randint(-100, 100, (4, 4), dtype=torch.float32)
    input_ms = ms.tensor(input_pt.numpy(), dtype=ms.float32)
    
    output_pt = torch.linalg.inv(input_pt)
    output_ms = ms.mint.linalg.inv(input_ms)
    
    assert np.allclose(output_pt.numpy(), output_ms.numpy(), rtol=1e-3)
    
def test_dtype():
    """
    (1a-1) 遍历全部类型的张量，判断是否可用
    """
    # mindspore框架
    ms_int_lst = [ms.int8, ms.int16, ms.int32, ms.int64]
    for dtype in ms_int_lst:
        input_ms = ms.tensor([[1, -2, 3], [5, -6, 7], [2, 4, 5]], dtype=dtype)
        try:
            output_ms = ms.mint.linalg.inv(input_ms)
            print(dtype, output_ms)
        except Exception as e:
            print(f"mindspore.mint.linalg.inv does not support dtype:{dtype}")
    
    ms_uint_lst = [ms.uint8, ms.uint16, ms.uint32, ms.uint64]
    for dtype in ms_uint_lst:
        input_ms = ms.tensor([[1, 2, 3], [5, 6, 7], [2, 4, 5]], dtype=dtype)
        try:
            output_ms = ms.mint.linalg.inv(input_ms)
            print(dtype, output_ms)
        except Exception as e:
            print(f"mindspore.mint.linalg.inv does not support dtype:{dtype}")
    
    ms_float_lst = [ms.float16, ms.float32, ms.float64, ms.bfloat16]
    for dtype in ms_float_lst:
        input_ms = ms.tensor([[1.3, 2.1, 3.7], [2.1, 13.2, 4.2], [4.2, 4.3, 14.3]], dtype=dtype)
        try:
            output_ms = ms.mint.linalg.inv(input_ms)
            print(dtype, output_ms)
        except Exception as e:
            print(f"mindspore.mint.linalg.inv does not support dtype:{dtype}")
     
    ms_complex_lst = [ms.complex64, ms.complex128]
    for dtype in ms_complex_lst:
        input_ms = ms.tensor([[1 + 1j, 1 + 21j, 1 + 3j], [1 + 4.2j, 1 + 5j, 1 + 6.1j], [1 + 7j, 1 + 8.1j, 1 + 9.9j]], dtype=dtype)
        try:
            output_ms = ms.mint.linalg.inv(input_ms)
            print(dtype, output_ms)
        except Exception as e:
            print(f"mindspore.mint.linalg.inv does not support dtype:{dtype}")
    
    input_ms = ms.tensor([[True, True, False], [False, True, False], [True, False, True]], dtype=ms.bool_)
    try:
        output_ms = ms.mint.linalg.inv(input_ms)
        print(ms.bool_, output_ms)
    except Exception as e:
        print(f"mindspore.mint.linalg.inv does not support dtype:{ms.bool_}")
    
    # mindspore框架中该函数仅支持mindspore.float32类型的张量
    
    # torch 框架
    pt_int_lst = [torch.int8, torch.int16, torch.int32, torch.int64]
    for dtype in pt_int_lst:
        input_pt = torch.tensor([[1, -2, 3], [5, -6, 7], [2, 4, 5]], dtype=dtype)
        try:
            output_pt = torch.linalg.inv(input_pt)
            print(dtype, output_pt) 
        except Exception as e:
            print(f"torch.linalg.inv does not support dtype:{dtype}")
      
    pt_uint_lst = [torch.uint8, torch.uint16, torch.uint32, torch.uint64]
    for dtype in pt_uint_lst:
        input_pt = torch.tensor([[1, 2, 3], [5, 6, 7], [2, 4, 5]], dtype=dtype)
        try:
            output_pt = torch.linalg.inv(input_pt)
            print(dtype, output_pt) 
        except Exception as e:
            print(f"torch.linalg.inv does not support dtype:{dtype}")
        
    pt_float_lst = [torch.float32, torch.float64, torch.float16, torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e4m3fnuz, torch.float8_e5m2fnuz]
    for dtype in pt_float_lst:
        input_pt = torch.tensor([[1.3, 2.1, 3.7], [2.1, 13.2, 4.2], [4.2, 4.3, 14.3]], dtype=dtype)
        try:
            output_pt = torch.linalg.inv(input_pt)
            print(dtype, output_pt)
        except Exception as e:
            print(f"torch.linalg.inv does not support dtype: {dtype}")
        
    
    pt_complex_lst = [torch.complex32, torch.complex64, torch.complex128]
    for dtype in pt_complex_lst:
        input_pt = torch.tensor([[1 + 1j, 1 + 21j, 1 + 3j], [1 + 4.2j, 1 + 5j, 1 + 6.1j], [1 + 7j, 1 + 8.1j, 1 + 9.9j]], dtype=dtype)
        try:
            output_pt = torch.linalg.inv(input_pt)
            print(dtype, output_pt) 
        except Exception as e:
            print(f"torch.linalg.inv does not support dtype:{dtype}")
    
    input_pt = torch.tensor([[True, True, False], [False, True, False], [True, False, True]], dtype=torch.bool)
    try:
        output_pt = torch.linalg.inv(input_pt)
        print(torch.bool, output_pt) 
    except Exception as e:
        print(f"torch.linalg.inv does not support dtype:{torch.bool}")
    
    # pytorch框架中对应函数仅支持torch.float32, torch.float64, torch.complex64, torch.complex128 类型的张量
    return

def test_dtype_diff():
    """
    (1a)-2 测试输入mindspore和pytorch对应接口均支持的张量类型时两个框架的输出差异，对于这一接口二者均支持的张量类型只有float32
    """
    input_ms = ms.tensor([[1.3, 2.1, 3.7], [2.1, 13.2, 4.2], [4.2, 4.3, 14.3]], dtype=ms.float32)
    output_ms = ms.mint.linalg.inv(input_ms)
    
    input_pt = torch.tensor([[1.3, 2.1, 3.7], [2.1, 13.2, 4.2], [4.2, 4.3, 14.3]], dtype=torch.float32)
    output_pt = torch.linalg.inv(input_pt)
    assert np.allclose(output_ms.numpy(), output_pt.numpy(), rtol=1e-3)

def test_random_input():
    """
    (1d) 测试混乱输入
    """
    # 测试张量最后两个维度不相等的情况
    input_ms = ms.ops.randn((2, 5), dtype=ms.float32)
    reported_flag = False
    try:
        output_ms = ms.mint.linalg.inv(input_ms)
        print(output_ms)
    except Exception as e:
        reported_flag = True
        e = str(e)
        assert ("dimension" in e) and ("same" in e)
    
    if reported_flag == False:
        assert "No error message was reported when the tensor's last two dimensions are not same." == 1
    
    # 测试其他类型的输入
    input_ms = "Tensor"
    reported_flag = False
    try:
        output_ms = ms.mint.linalg.inv(input_ms)
        print(output_ms)
    except Exception as e:
        reported_flag = True
        e = str(e)
        assert "string" in e and "Tensor" in e
    
    if reported_flag == False:
        assert "No error message was reported when the input is string." == 1
        
    # 测试1维张量
    reported_flag = False
    input_ms = ms.ops.randn((100), dtype=ms.float32)
    try:
        output_ms = ms.mint.linalg.inv(input_ms)
        print(output_ms)
    except Exception as e:
        reported_flag = True
        e = str(e)
        assert "at least" in e and "2" in e
        
    if reported_flag == False:
        assert "No error message was reported when the tensor's dimension is less then 2." == 1
    
    
    # 测试6维以上的张量
    reported_flag = False
    input_ms = ms.ops.randn((2, 2, 2, 2, 2, 2, 2), dtype=ms.float32)
    try:
        output_ms = ms.mint.linalg.inv(input_ms)
        print(output_ms)
    except Exception as e:
        reported_flag = True
        e = str(e)
        assert "larger than" in e and "6" in e
        
    if reported_flag == False:
        assert "No error message was reported when the tensor's dimension is larger than 6." == 1
    
    # 测试输入为空的情况
    reported_flag = False
    try:
        output_ms = ms.mint.linalg.inv(None)
        print(output_ms)
    except Exception as e:
        reported_flag = True
        e = str(e)
        assert "None" in e and "Tensor" in e
    
    if reported_flag == False:
        assert "No error message was reported when the input is None" == 1
    
    # 测试输入矩阵奇异矩阵
    input_ms = ms.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=ms.float32)
    reported_flag = False
    try:
        output_ms = ms.mint.linalg.inv(input_ms)
    except Exception as e:
        reported_flag = True
        e = str(e)
        assert "singular" in e
    
    if reported_flag == False:
        assert "No error message was reported when the input is singular matrix." == 1
    
    # 测试问题：输入非奇异矩阵，函数不报错。
        
def test_forward_backward():
    data_lst = [[1.3, 2.1, 3.7], [2.1, 13.2, 4.2], [4.2, 4.3, 14.3]]
    input_pt = torch.tensor(data_lst, dtype=torch.float32, requires_grad=True)
    input_ms = ms.tensor(data_lst, dtype=ms.float32)
    
    def ms_func(ms_tensor):
        result_ms = ms.mint.linalg.inv(ms_tensor)
        return result_ms.sum()
    
    def pt_func(pt_tensor):
        result_pt = torch.linalg.inv(pt_tensor)
        return result_pt.sum()
    
    grad_func = value_and_grad(ms_func)
    output_ms, gradient_ms = grad_func(input_ms)
    output_pt = pt_func(input_pt)
    output_pt.backward()
    gradient_pt = input_pt.grad
    
    assert np.allclose(output_ms.numpy(), output_pt.detach().numpy(), rtol=1e-3)
    assert np.allclose(gradient_ms.numpy(), gradient_pt.numpy(), rtol=1e-3)
    
ms.set_context(device_target='Ascend')
pytest.main(['-vs', 'test_inv.py', '--html', './report/report.html'])

