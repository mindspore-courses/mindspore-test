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
pt_float_lst = [torch.float32, torch.float64, torch.float16, torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e4m3fnuz, torch.float8_e5m2fnuz]
pt_complex_lst = [torch.complex32, torch.complex64, torch.complex128]



def test_ramdom_tensor():
    """
    (1b) 固定dtype=mindspore.int64, 使用随机输入值，测试两个框架的输出是否相等
    """
    input_pt = torch.randint(-100, 100, (4, 4), dtype=torch.int64)
    input_ms = ms.tensor(input_pt.numpy(), dtype=ms.int64)
    output_pt = torch.nn.functional.pad(input_pt, (2, 2), 'constant', 9)
    output_ms = ms.mint.nn.functional.pad(input_ms, [2, 2], 'constant', 9)

    assert np.allclose(output_pt.numpy(), output_ms.numpy(), rtol=1e-3)
    
@pytest.mark.parametrize('mode', ['constant', 'reflect', 'replicate', 'circular'])
def test_modes(mode):
    """
    (1c) 测试不同的模式
    """
    data_lst = [[1.1, 1.2, 1.3, 1.4], [1.5, 1.6, 1.7, 1.8]]
    input_pt = torch.tensor(data_lst, dtype=torch.float32)
    input_ms = ms.tensor(data_lst, dtype=ms.float32)
    
    if mode == 'constant':
        output_pt = torch.nn.functional.pad(input_pt, (2, 2), mode, 1.2)
        output_ms = ms.mint.nn.functional.pad(input_ms, [2, 2], mode, 1.2)
    else:
        output_pt = torch.nn.functional.pad(input_pt, (2, 2), mode)
        output_ms = ms.mint.nn.functional.pad(input_ms, [2, 2], mode)

    # 测试问题： 浮点类型的张量 执行circular模式下的这一函数会出现启动内核失败的问题
    assert np.allclose(output_pt.numpy(), output_ms.numpy(), rtol=1e-3)

def test_dtype(): # 测试不同类型的张量是否可用
    
    """
    (1a-1) 遍历全部类型的张量，判断是否可用
    """
    # mindspore框架
    
    for dtype in ms_int_lst:
        input_ms = ms.tensor([[1, -2, 3, -4], [5, -6, 7, -8]], dtype=dtype)
        try:
            output_ms = ms.mint.nn.functional.pad(input_ms, (2, 2), 'constant', -1)
        except Exception as e:
            print(f"mint.nn.functional.pad does not support dtype:{dtype}")
    
    
    for dtype in ms_uint_lst:
        input_ms = ms.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=dtype)
        try:
            output_ms = ms.mint.nn.functional.pad(input_ms, (2, 2), 'constant', 1)
        except Exception as e:
            print(f"mint.nn.functional.pad does not support dtype:{dtype}")
    
    
    for dtype in ms_float_lst:
        input_ms = ms.tensor([[1.1, 1.2, 1.3, 1.4], [1.5, 1.6, 1.7, 1.8]], dtype=dtype)
        try:
            output_ms = ms.mint.nn.functional.pad(input_ms, (2, 2), 'constant', 1.2)
        except Exception as e:
            print(f"mint.nn.functional.pad does not support dtype:{dtype}")
     
    
    for dtype in ms_complex_lst:
        input_ms = ms.tensor([[1 + 1j, 1 - 1j, 1 + 0.5j], [1 + 2j, 1 + 3j, 1 + 4j]], dtype=dtype)
        try:
            output_ms = ms.mint.nn.functional.pad(input_ms, (2, 2), 'constant', -1)
        except Exception as e:
            print(f"mint.nn.functional.pad does not support dtype:{dtype}")
    
        
    input_ms = ms.tensor([[True, True, False, False], [True, False, True, False]], dtype=ms.bool_)
    try:
        output_ms = ms.mint.nn.functional.pad(input_ms, (2, 2), 'constant', True)
    except Exception as e:
        print(f"mint.nn.functional.pad does not support dtype:{dtype}")
    
    # torch 框架
    
    for dtype in pt_int_lst:
        input_pt = torch.tensor([[1, -2, 3, -4], [5, -6, 7, -8]], dtype=dtype)
        try:
            output_pt = torch.nn.functional.pad(input_pt, (2, 2), 'constant', -1)
        except Exception as e:
            print(f"torch.nn.functional.pad does not support dtype:{dtype}")
      
    
    for dtype in pt_uint_lst:
        input_pt = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=dtype)
        try:
            output_pt = torch.nn.functional.pad(input_pt, (2, 2), 'constant', 1)
        except Exception as e:
            print(f"torch.nn.functional.pad does not support dtype:{dtype}")
        
    
    for dtype in pt_float_lst:
        input_pt = torch.tensor([[1.1, 1.2, 1.3, 1.4], [1.5, 1.6, 1.7, 1.8]], dtype=dtype)
        try:
            output_pt = torch.nn.functional.pad(input_pt, (2, 2), 'constant', 1)
        except Exception as e:
            print(f"torch.nn.functional.pad does not support dtype:{dtype}")
    
    
    for dtype in pt_complex_lst:
        input_pt = torch.tensor([[1 + 1j, 1 - 1j, 1 + 0.5j], [1 + 2j, 1 + 3j, 1 + 4j]], dtype=dtype)
        try:
            output_pt = torch.nn.functional.pad(input_pt, (2, 2), 'constant', -1)
        except Exception as e:
            print(f"torch.nn.functional.pad does not support dtype:{dtype}")
    
    input_pt = torch.tensor([[True, True, False, False], [True, False, True, False]], dtype=torch.bool)
    try:
        output_pt = torch.nn.functional.pad(input_pt, (2, 2), 'constant', True)    
    except Exception as e:
        print(f"torch.nn.functional.pad does not support dtype:{torch.bool}")
        
    # mindspore框架中全部类型均支持此函数
    # torch框架中 torch.float8_e4m3fn torch.float8_e5m2. torch.float8_e4m3fnuz, torch.float8_e5m2fnuz 不支持此函数

@pytest.mark.parametrize('ms_type,pt_type', [[ms.int8, torch.int8], [ms.int16, torch.int16], [ms.int32, torch.int32], [ms.int64, torch.int64], [ms.uint8, torch.uint8], [ms.uint16, torch.uint16], 
                                             [ms.uint32, torch.uint32], [ms.uint64, torch.uint64], [ms.float16, torch.float16], [ms.float32, torch.float32], [ms.float64, torch.float64], [ms.bfloat16, 
                                             torch.bfloat16], [ms.complex64, torch.complex64], [ms.complex128, torch.complex128], [ms.bool_, torch.bool]])
def test_dtype_diff(ms_type, pt_type):
    """
    (1a-2) 对于mindspore和pytorch均支持的数据类型进行遍历，测试输出差异
    """
    if (ms_type in ms_int_lst) and (pt_type in pt_int_lst):
        input_ms = ms.tensor([[1, -2, 3, -4], [5, -6, 7, -8]], dtype=ms_type)
        output_ms = ms.mint.nn.functional.pad(input_ms, (2, 2), 'constant', -1)
        
        input_pt = torch.tensor([[1, -2, 3, -4], [5, -6, 7, -8]], dtype=pt_type)
        output_pt = torch.nn.functional.pad(input_pt, (2, 2), 'constant', -1)
        assert np.allclose(output_pt.numpy(), output_ms.numpy(), rtol=1e-3)
    
    if (ms_type in ms_uint_lst) and (pt_type in pt_uint_lst):
        input_ms = ms.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=ms_type)
        output_ms = ms.mint.nn.functional.pad(input_ms, (2, 2), 'constant', 1)
        
        input_pt = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=pt_type)
        output_pt = torch.nn.functional.pad(input_pt, (2, 2), 'constant', 1)
        assert np.allclose(output_pt.numpy(), output_ms.numpy(), rtol=1e-3)
    
    if (ms_type in ms_float_lst) and (pt_type in pt_float_lst):
        input_ms = ms.tensor([[1.1, 1.2, 1.3, 1.4], [1.5, 1.6, 1.7, 1.8]], dtype=ms_type)
        output_ms = ms.mint.nn.functional.pad(input_ms, (2, 2), 'constant', 1.2)
        
        input_pt = torch.tensor([[1.1, 1.2, 1.3, 1.4], [1.5, 1.6, 1.7, 1.8]], dtype=pt_type)
        output_pt = torch.nn.functional.pad(input_pt, (2, 2), 'constant', 1.2)
        
        if ms_type == ms.bfloat16 and pt_type == torch.bfloat16:
            # 由于numpy不支持bfloat16类型，因此需要进行张量类型转换为float32后再比较差异
            cast_op = ops.Cast()
            output_ms = cast_op(output_ms, ms.float32)
            output_pt = output_pt.to(dtype=torch.float32)
            
        assert np.allclose(output_pt.numpy(), output_ms.numpy(), rtol=1e-3)
    
    if (ms_type in ms_complex_lst) and (pt_type in pt_complex_lst):
        input_ms = ms.tensor([[1 + 1j, 1 - 1j, 1 + 0.5j], [1 + 2j, 1 + 3j, 1 + 4j]], dtype=ms_type)
        output_ms = ms.mint.nn.functional.pad(input_ms, (2, 2), 'constant', True)
        
        input_pt = torch.tensor([[1 + 1j, 1 - 1j, 1 + 0.5j], [1 + 2j, 1 + 3j, 1 + 4j]], dtype=pt_type)
        output_pt = torch.nn.functional.pad(input_pt, (2, 2), 'constant', True)
        assert np.allclose(output_pt.numpy(), output_ms.numpy(), rtol=1e-3)
    
    if ms_type == ms.bool_ and pt_type == torch.bool:
        input_ms = ms.tensor([[True, True, False, False], [True, False, True, False]], dtype=ms.bool_)
        output_ms = ms.mint.nn.functional.pad(input_ms, (2, 2), 'constant', True)
        
        input_pt = torch.tensor([[True, True, False, False], [True, False, True, False]], dtype=torch.bool)
        output_pt = torch.nn.functional.pad(input_pt, (2, 2), 'constant', True) 
        assert np.allclose(output_pt.numpy(), output_ms.numpy(), rtol=1e-3)
        
    
def test_chaotic_input():
    """
    (1d) 测试混乱输入
    """
    # 打乱输入顺序
    reported_flag = False
    input_ms = ms.tensor([[1, -2, 3, -4], [5, -6, 7, -8]], dtype=ms.int64)
    try:
        output_ms = ms.mint.nn.functional.pad((2, 2), 'constant', -1, input_ms) # 乱序输入
    except Exception as e:
        e = str(e)
        reported_flag = True
        assert "tuple" in e and "Tensor" in e
    
    if reported_flag == False:
        assert "No error message was reported when the inputs are chaotic." == 1
        
    # 测试输入numpy数组
    reported_flag = False
    try:
        output_ms = ms.mint.nn.functional.pad(input_ms.numpy(), (2, 2), 'constant', -1) # 将张量替换为numpy数组
    except Exception as e:
        e = str(e)
        reported_flag = True
        assert "numpy" in e
    
    if reported_flag == False:
        assert "No error message was reported when the input is numpy array" == 1

def test_forward_backward():
    """
    (2) 测试两个框架的正向推理结果和反向梯度精度差异
    """
    data_lst = [[1.1, 1.2, 1.3, 1.4], [1.5, 1.6, 1.7, 1.8]]
    input_ms = ms.tensor(data_lst, dtype=ms.float32)
    input_pt= torch.tensor(data_lst, dtype=torch.float32, requires_grad=True)
    
    def ms_func(x):
        a = ms.mint.nn.functional.pad(x, (2, 2), 'constant', 1.2)
        return a.sum()
    
    def pt_func(x):
        a = torch.nn.functional.pad(x, (2, 2), 'constant', 1.2)
        return a.sum()
    
    grad_func = value_and_grad(ms_func)
    output_ms, gradient_ms = grad_func(input_ms)
    output_pt = pt_func(input_pt)
    output_pt.backward()
    gradient_pt = input_pt.grad
    
    assert np.allclose(output_ms.numpy(), output_pt.detach().numpy(), rtol=1e-3)
    assert np.allclose(gradient_ms.numpy(), gradient_pt.numpy(), rtol=1e-3)
    
ms.set_context(device_target='Ascend')
pytest.main(['-vs', 'test_pad.py', '--html', './report/report.html'])

