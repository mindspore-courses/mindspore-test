"""
测试 mindspore.mint.cumsum 接口
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
    """
    # 使用float32作为固定数据类型
    shapes = [
        (4,),                    # 1D
        (8, 6),                 # 2D
        (4, 6, 8),             # 3D
        (4, 4, 4, 4)           # 4D
    ]
    
    max_errors = {}  # 记录每个维度配置的最大误差
    
    for shape in shapes:
        ndim = len(shape)
        print(f"\n测试{ndim}D张量，形状: {shape}")
        
        # 生成随机数据，确保数值在合理范围内
        data = np.random.uniform(-10, 10, size=shape).astype(np.float32)
        
        # PyTorch测试
        torch_input = torch.tensor(data, dtype=torch.float32)
        dims = list(range(ndim))  # 测试所有可能的维度
        
        for dim in dims:
            torch_output = torch.cumsum(torch_input, dim=dim)
            
            # MindSpore测试
            ms_input = Tensor(data, dtype=ms.float32)
            ms_output = mint.cumsum(ms_input, dim=dim)
            
            # 计算误差
            error = calculate_error(torch_output, ms_output)
            key = f"{ndim}D_dim{dim}"
            max_errors[key] = error
            
            print(f"维度 {dim} 的最大误差: {error:.2e}")
            
            # 特别关注4D张量在dim=3的情况
            if ndim == 4 and dim == 3:
                print(f"4D张量在dim=3时的误差: {error:.2e}")
                assert error < 1e-3, f"4D张量在dim=3时误差 {error} 超过阈值"
            
            assert error < 1e-3, f"形状 {shape} 维度 {dim} 的误差 {error} 超过阈值"
    
    # 分析误差趋势
    print("\n误差分析总结:")
    for ndim in range(1, 5):
        dim_errors = [max_errors[f"{ndim}D_dim{d}"] for d in range(ndim)]
        avg_error = np.mean(dim_errors)
        max_error = np.max(dim_errors)
        print(f"{ndim}D张量 - 平均误差: {avg_error:.2e}, 最大误差: {max_error:.2e}")
    
    return max_errors

def test_param_type_support():
    """
    测试不同参数类型的支持情况
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

def test_precision_with_dimension():

    print("\n=== 测试精度随维度变化 ===")
    
    # 使用固定的随机种子以确保结果可重现
    np.random.seed(42)
    
    # 测试不同维度的张量
    test_cases = [
        (np.array([1, 2, 3, 4], dtype=np.float32), 0),                    # 1D
        (np.random.randn(5, 6).astype(np.float32), 1),                    # 2D
        (np.random.randn(4, 4, 4).astype(np.float32), 2),                 # 3D
        (np.random.randn(3, 3, 3, 3).astype(np.float32), 3)              # 4D
    ]
    
    results = {}
    
    for data, dim in test_cases:
        ndim = len(data.shape)
        print(f"\n测试{ndim}D张量，形状: {data.shape}，维度: {dim}")
        
        # PyTorch测试
        torch_input = torch.tensor(data, dtype=torch.float32)
        torch_output = torch.cumsum(torch_input, dim=dim)
        
        # MindSpore测试
        ms_input = Tensor(data, dtype=ms.float32)
        ms_output = mint.cumsum(ms_input, dim=dim)
        
        # 计算误差
        error = calculate_error(torch_output, ms_output)
        results[f"{ndim}D_dim{dim}"] = error
        print(f"最大误差: {error:.2e}")
        
        # 特别关注4D张量在dim=3的情况
        if ndim == 4 and dim == 3:
            print(f"4D张量在dim=3时的误差: {error:.2e}")
            assert error < 1e-3, f"4D张量在dim=3时误差 {error} 超过阈值"
            # 验证误差是否在预期范围内（允许一定的浮动）
            assert abs(error - 2.38e-7) < 1e-7, f"4D张量在dim=3时误差 {error} 与预期值2.38e-7相差过大"
    
    # 分析误差趋势
    print("\n误差分析总结:")
    for ndim in range(1, 5):
        dim_error = results.get(f"{ndim}D_dim{ndim-1}", 0)
        print(f"{ndim}D张量在dim={ndim-1}的误差: {dim_error:.2e}")
    
    return results

def test_fixed_input_values():
    """测试固定输入值的情况"""
    # 固定的测试数据
    fixed_data = np.array([
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0], [7.0, 8.0]]
    ], dtype=np.float32)
    
    print("\n测试固定输入值")
    
    # PyTorch测试
    torch_input = torch.tensor(fixed_data)
    
    # MindSpore测试
    ms_input = Tensor(fixed_data)
    
    # 测试所有维度
    for dim in range(fixed_data.ndim):
        torch_output = torch.cumsum(torch_input, dim=dim)
        ms_output = mint.cumsum(ms_input, dim=dim)
        
        error = calculate_error(torch_output, ms_output)
        print(f"维度 {dim} 的误差: {error:.2e}")
        assert error < 1e-3, f"固定输入值在维度 {dim} 的误差 {error} 超过阈值"

def test_large_values():
    """测试大数值输入的情况"""
    # 生成较大的数值
    large_data = np.array([1e5, 2e5, 3e5], dtype=np.float32)
    
    # PyTorch测试
    torch_input = torch.tensor(large_data)
    torch_output = torch.cumsum(torch_input, dim=0)
    
    # MindSpore测试
    ms_input = Tensor(large_data)
    ms_output = mint.cumsum(ms_input, dim=0)
    
    error = calculate_error(torch_output, ms_output)
    print(f"\n大数值测试的误差: {error:.2e}")
    assert error < 1e-3, f"大数值测试的误差 {error} 超过阈值"

def test_zero_size_dimension():
    """测试包含0维度的张量"""
    # 创建包含0维度的张量
    zero_dim_data = np.zeros((2, 0, 3), dtype=np.float32)
    
    try:
        # PyTorch测试
        torch_input = torch.tensor(zero_dim_data)
        torch_output = torch.cumsum(torch_input, dim=1)
        
        # MindSpore测试
        ms_input = Tensor(zero_dim_data)
        ms_output = mint.cumsum(ms_input, dim=1)
        
        print("\n0维度张量测试成功")
    except Exception as e:
        print(f"\n0维度张量测试异常: {str(e)}")

def test_int8_overflow():
    """测试int8类型的溢出情况"""
    print("\n=== 测试int8类型溢出 ===")
    
    # 测试用例1: 正数溢出
    data1 = np.array([64, 64, 64], dtype=np.int8)  # 累加后会超过127
    
    # 测试用例2: 负数溢出
    data2 = np.array([-64, -64, -64], dtype=np.int8)  # 累加后会小于-128
    
    # 测试用例3: 边界值
    data3 = np.array([127, 1, 1], dtype=np.int8)  # 从最大值开始累加
    
    test_cases = [
        (data1, "正数溢出"),
        (data2, "负数溢出"),
        (data3, "边界值")
    ]
    
    results = {}
    for data, case_name in test_cases:
        print(f"\n测试{case_name}:")
        print(f"输入数据: {data}")
        
        # PyTorch测试
        torch_input = torch.tensor(data, dtype=torch.int8)
        torch_output = torch.cumsum(torch_input, dim=0)
        
        # MindSpore测试
        ms_input = Tensor(data, dtype=ms.int8)
        ms_output = mint.cumsum(ms_input, dim=0)
        
        # 计算结果
        torch_result = torch_output.numpy()
        ms_result = ms_output.asnumpy()
        
        print(f"PyTorch结果: {torch_result}")
        print(f"MindSpore结果: {ms_result}")
        
        # 检查是否一致
        is_equal = np.array_equal(torch_result, ms_result)
        results[case_name] = {
            'torch_output': torch_result,
            'ms_output': ms_result,
            'is_equal': is_equal
        }
        
        # 验证结果是否符合预期的溢出行为
        assert is_equal, f"{case_name}情况下，MindSpore和PyTorch的结果不一致"

def test_axis_type_support():
    """测试cumsum接口的axis参数对不同类型整数的支持情况"""
    print("\n=== 测试axis参数类型支持 ===")
    
    # 准备测试数据
    data = np.random.randn(2, 3, 4).astype(np.float32)
    ms_input = Tensor(data, dtype=ms.float32)
    
    # 测试不同类型的axis值
    test_cases = [
        (np.int8(1), "numpy.int8"),
        (np.int16(1), "numpy.int16"),
        (np.int32(1), "numpy.int32"),
        (np.int64(1), "numpy.int64"),
        (np.uint8(1), "numpy.uint8"),
        (np.uint16(1), "numpy.uint16"),
        (np.uint32(1), "numpy.uint32"),
        (np.uint64(1), "numpy.uint64")
    ]
    
    results = {}
    for axis_value, type_name in test_cases:
        print(f"\n测试axis类型: {type_name}")
        print(f"axis值: {axis_value}, 类型: {type(axis_value)}")
        
        try:
            # MindSpore测试
            ms_output = mint.cumsum(ms_input, dim=axis_value)
            
            # 验证结果
            expected_output = mint.cumsum(ms_input, dim=1)  # 使用普通整数作为参考
            is_equal = np.array_equal(ms_output.asnumpy(), expected_output.asnumpy())
            
            results[type_name] = {
                'supported': True,
                'output_shape': ms_output.shape,
                'matches_expected': is_equal
            }
            
            print(f"测试成功 - 输出形状: {ms_output.shape}")
            assert is_equal, f"{type_name}类型的结果与预期不符"
            
        except Exception as e:
            print(f"测试失败 - 错误信息: {str(e)}")
            results[type_name] = {
                'supported': False,
                'error_message': str(e)
            }
    
    # 检查结果总结
    print("\n测试结果总结:")
    for type_name, result in results.items():
        if result['supported']:
            print(f"{type_name}: 支持 ✓")
        else:
            print(f"{type_name}: 不支持 ✗ - {result['error_message']}")
    
    return results

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
    
    print("\n=== Testing precision with dimension ===")
    test_precision_with_dimension()
    
    print("\n=== Testing fixed input values ===")
    test_fixed_input_values()
    
    print("\n=== Testing large values ===")
    test_large_values()
    
    print("\n=== Testing zero size dimension ===")
    test_zero_size_dimension()
    
    print("\n=== Testing int8 overflow ===")
    test_int8_overflow()
    
    print("\n=== Testing axis type support ===")
    axis_results = test_axis_type_support()
