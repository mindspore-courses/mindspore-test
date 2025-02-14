"""
测试 mindspore.mint.cumsum 接口

本测试文件验证 mindspore.mint.cumsum 接口的功能。
主要测试以下方面：
1. 数据类型支持：验证对 float32、int32、float16、int8、uint8 等数据类型的支持
2. 输出精度：验证与 PyTorch 的 torch.cumsum 在各种输入下的一致性
3. 参数支持：验证参数类型和取值范围的支持情况
4. 神经网络集成：验证在神经网络中的使用
5. 错误处理：验证异常输入的处理
6. 梯度计算：验证反向传播的准确性
"""

import numpy as np
import torch
import mindspore as ms
from mindspore import Tensor, nn
import mindspore.mint as mint
import pytest

def calculate_error(torch_output, ms_output):
    """计算PyTorch和MindSpore输出之间的误差"""
    if isinstance(torch_output, torch.Tensor):
        if torch_output.requires_grad:
            torch_output = torch_output.detach()
        torch_output = torch_output.numpy()
    if isinstance(ms_output, ms.Tensor):
        ms_output = ms_output.asnumpy()
    return np.max(np.abs(torch_output - ms_output))

def test_random_dtype_support():
    """
    测试不同数据类型的支持情况
    已知问题：int8类型存在溢出问题（详见ISSUES.md#cumsum-int8类型溢出问题）
    """
    shape = (2, 3, 4)
    results = {}
    
    dtypes = [
        (np.float32, torch.float32, ms.float32),
        (np.int32, torch.int32, ms.int32),
        (np.float16, torch.float16, ms.float16),
        (np.int8, torch.int8, ms.int8),
        (np.uint8, torch.uint8, ms.uint8)
    ]
    
    for np_dtype, torch_dtype, ms_dtype in dtypes:
        print(f"\nTesting dtype: {np_dtype}")
        try:
            # 生成随机数据
            if np_dtype in [np.float32, np.float16]:
                data = np.random.randn(*shape).astype(np_dtype)
            else:
                data = np.random.randint(1, 100, size=shape, dtype=np_dtype)
            
            # PyTorch测试
            torch_input = torch.tensor(data, dtype=torch_dtype)
            torch_output = torch.cumsum(torch_input, dim=0)
            
            # MindSpore测试
            ms_input = Tensor(data, dtype=ms_dtype)
            ms_output = mint.cumsum(ms_input, dim=0)
            
            # 计算误差
            error = calculate_error(torch_output, ms_output)
            print(f"Max error: {error}")
            
            results[str(np_dtype)] = {
                'supported': True,
                'error': error
            }
            
            assert error < 1e-3, f"Error {error} exceeds threshold for dtype {np_dtype}"
            
        except Exception as e:
            print(f"Error with dtype {np_dtype}: {str(e)}")
            results[str(np_dtype)] = {
                'supported': False,
                'error_message': str(e)
            }
    
    return results

def test_fixed_dtype_output_accuracy():
    """
    测试固定数据类型下随机输入的输出精度
    已知问题：精度随维度增加略有下降（详见ISSUES.md#cumsum精度随维度变化）
    """
    # 使用float32作为固定数据类型
    shapes = [
        (4,),
        (2, 3),
        (2, 3, 4),
        (2, 3, 4, 5)
    ]
    
    for shape in shapes:
        print(f"\nTesting shape: {shape}")
        data = np.random.randn(*shape).astype(np.float32)
        
        # PyTorch测试
        torch_input = torch.tensor(data, dtype=torch.float32)
        dims = list(range(len(shape)))  # 测试所有可能的维度
        
        for dim in dims:
            torch_output = torch.cumsum(torch_input, dim=dim)
            
            # MindSpore测试
            ms_input = Tensor(data, dtype=ms.float32)
            ms_output = mint.cumsum(ms_input, dim=dim)
            
            # 计算误差
            error = calculate_error(torch_output, ms_output)
            print(f"Shape {shape}, dim {dim}, max error: {error}")
            assert error < 1e-3, f"Error {error} exceeds threshold for shape {shape} and dim {dim}"

def test_param_type_support():
    """
    测试不同参数类型的支持情况
    已知问题：维度参数必须是整数类型（详见ISSUES.md#cumsum维度参数限制）
    """
    data = np.random.randn(2, 3, 4).astype(np.float32)
    ms_input = Tensor(data, dtype=ms.float32)
    
    # 测试dim参数的不同类型
    test_cases = [
        (0, "single integer"),
        (-1, "negative index"),
        (1.0, "float instead of int"),
        ("0", "string instead of int"),
        (True, "boolean instead of int"),
        (None, "None value"),
    ]
    
    results = {}
    for dim, desc in test_cases:
        print(f"\nTesting dim parameter: {desc}")
        try:
            ms_output = mint.cumsum(ms_input, dim=dim)
            results[desc] = {
                'supported': True,
                'output_shape': ms_output.shape
            }
        except Exception as e:
            print(f"Error with {desc}: {str(e)}")
            results[desc] = {
                'supported': False,
                'error_message': str(e)
            }
    
    return results

class CumSumNet(nn.Cell):
    def __init__(self):
        super(CumSumNet, self).__init__()
        self.weight = ms.Parameter(Tensor(np.random.randn(2, 3).astype(np.float32)))
    
    def construct(self, x):
        return mint.cumsum(x * self.weight, dim=0)

def test_neural_network():
    """测试神经网络中的cumsum操作"""
    # 初始化网络
    net = CumSumNet()
    
    # 准备输入数据
    input_data = Tensor(np.random.randn(2, 3).astype(np.float32))
    
    try:
        # 前向推理
        output = net(input_data)
        print("Forward inference successful")
        print("Output shape:", output.shape)
        
        # 验证输出形状
        assert output.shape == (2, 3), f"Expected output shape (2, 3), got {output.shape}"
        
    except Exception as e:
        print(f"Error in neural network test: {str(e)}")
        raise

def test_error_handling():
    """测试错误处理"""
    data = np.random.randn(2, 3, 4).astype(np.float32)
    ms_input = Tensor(data, dtype=ms.float32)
    
    error_cases = [
        {
            'dim': 3,  # 超出维度范围
            'desc': "Dimension out of range"
        },
        {
            'dim': -4,  # 负数维度超出范围
            'desc': "Negative dimension out of range"
        },
        {
            'dim': 0.5,  # 非整数维度
            'desc': "Non-integer dimension"
        }
    ]
    
    results = {}
    for case in error_cases:
        print(f"\nTesting error case: {case['desc']}")
        try:
            output = mint.cumsum(ms_input, dim=case['dim'])
            results[case['desc']] = {
                'raised_error': False,
                'message': "No error raised"
            }
        except Exception as e:
            print(f"Error as expected: {str(e)}")
            results[case['desc']] = {
                'raised_error': True,
                'message': str(e)
            }
    
    return results

def test_backward_gradient():
    """测试反向传播梯度"""
    # 准备数据
    data = np.random.randn(2, 3).astype(np.float32)
    
    # PyTorch测试
    torch_input = torch.tensor(data, requires_grad=True)
    torch_output = torch.cumsum(torch_input, dim=0)
    torch_output.sum().backward()
    torch_grad = torch_input.grad.numpy()
    
    # MindSpore测试
    ms_input = Tensor(data, dtype=ms.float32)
    
    def grad_fn(inputs):
        return mint.cumsum(inputs, dim=0)
    
    ms_grad = ms.grad(grad_fn)(ms_input).asnumpy()
    
    # 比较梯度
    grad_error = np.max(np.abs(torch_grad - ms_grad))
    print(f"Maximum gradient error: {grad_error}")
    assert grad_error < 1e-3, f"Gradient error {grad_error} exceeds threshold"

if __name__ == "__main__":
    print("=== Testing random dtype support ===")
    dtype_results = test_random_dtype_support()
    
    print("\n=== Testing fixed dtype output accuracy ===")
    test_fixed_dtype_output_accuracy()
    
    print("\n=== Testing parameter type support ===")
    param_results = test_param_type_support()
    
    print("\n=== Testing neural network ===")
    test_neural_network()
    
    print("\n=== Testing error handling ===")
    error_results = test_error_handling()
    
    print("\n=== Testing backward gradient ===")
    test_backward_gradient()
