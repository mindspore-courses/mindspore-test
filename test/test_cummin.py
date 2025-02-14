import numpy as np
import torch
import mindspore as ms
import mindspore.mint as mint
from mindspore import Tensor, Parameter
import pytest

def calculate_error(torch_output, ms_output):
    """计算两个输出之间的最大误差"""
    if isinstance(torch_output, tuple):
        torch_output = torch_output[0]  # 某些操作可能返回多个值
    if isinstance(ms_output, (tuple, list)):
        ms_output = ms_output[0]  # MindSpore可能返回列表
    return np.max(np.abs(torch_output.detach().numpy() - ms_output.asnumpy()))

def generate_random_data(shape, dtype=np.float32):
    """生成随机数据"""
    if dtype in [np.float16, np.float32, np.float64]:
        return np.random.randn(*shape).astype(dtype)
    elif dtype == np.uint8:
        return np.random.randint(0, 100, size=shape, dtype=dtype)
    else:
        return np.random.randint(-100, 100, size=shape, dtype=dtype)

def test_random_dtype_support():
    """测试不同数据类型的支持情况"""
    shape = (4, 5)
    dtypes = [
        (np.float32, torch.float32, ms.float32),  # 主要测试float32
        (np.int32, torch.int32, ms.int32),        # 主要测试int32
    ]
    
    results = []
    for np_dtype, torch_dtype, ms_dtype in dtypes:
        data = generate_random_data(shape, np_dtype)
        torch_support = True
        ms_support = True
        error_msg = {'torch': '', 'mindspore': ''}
        
        try:
            torch_input = torch.tensor(data, dtype=torch_dtype)
            torch_output = torch.cummin(torch_input, dim=0)
        except Exception as e:
            torch_support = False
            error_msg['torch'] = str(e)
        
        try:
            ms_input = Tensor(data, dtype=ms_dtype)
            ms_output = mint.cummin(ms_input, dim=0)
        except Exception as e:
            ms_support = False
            error_msg['mindspore'] = str(e)
        
        if torch_support and ms_support:
            try:
                error = calculate_error(torch_output, ms_output)
                status = "PASS" if error < 1e-3 else f"FAIL (error: {error:.6f})"
            except Exception as e:
                status = f"ERROR: {str(e)}"
        else:
            status = "ERROR"
        
        results.append({
            'dtype': str(np_dtype),
            'torch_support': torch_support,
            'ms_support': ms_support,
            'status': status,
            'torch_error': error_msg['torch'],
            'ms_error': error_msg['mindspore']
        })
    
    print("\n=== Data Type Support Results ===")
    print("\n{:<15} {:<15} {:<15} {:<20}".format(
        "Data Type", "PyTorch", "MindSpore", "Status"))
    print("-" * 65)
    
    for result in results:
        print("{:<15} {:<15} {:<15} {:<20}".format(
            result['dtype'],
            "✓" if result['torch_support'] else "✗",
            "✓" if result['ms_support'] else "✗",
            result['status']
        ))

def test_fixed_dtype_output_accuracy():
    """测试固定数据类型下随机输入的输出精度"""
    shapes_to_test = [
        (4,),      # 1D
        (4, 5),    # 2D
    ]
    
    print("\n=== Output Accuracy Test Results ===")
    print("\n{:<20} {:<15} {:<20}".format("Shape", "Max Error", "Status"))
    print("-" * 55)
    
    for shape in shapes_to_test:
        try:
            data = generate_random_data(shape, np.float32)
            
            # PyTorch测试
            torch_input = torch.tensor(data, dtype=torch.float32)
            torch_output = torch.cummin(torch_input, dim=0)
            
            # MindSpore测试
            ms_input = Tensor(data, dtype=ms.float32)
            ms_output = mint.cummin(ms_input, dim=0)
            
            # 计算误差
            error = calculate_error(torch_output, ms_output)
            status = "PASS" if error < 1e-3 else "FAIL"
            
            print("{:<20} {:<15.6f} {:<20}".format(
                str(shape), error, status))
            
        except Exception as e:
            print("{:<20} {:<15} {:<20}".format(
                str(shape), "ERROR", str(e)[:20]))

def test_simple_cases():
    """测试简单用例"""
    print("\n=== Simple Cases Test Results ===")
    
    test_cases = [
        {
            'name': 'Basic 1D',
            'data': np.array([3, 1, 4, 1, 5], dtype=np.float32),
            'dim': 0
        },
        {
            'name': 'Basic 2D',
            'data': np.array([[2, 1], [1, 2]], dtype=np.float32),
            'dim': 0
        }
    ]
    
    for case in test_cases:
        try:
            torch_input = torch.tensor(case['data'])
            ms_input = Tensor(case['data'])
            
            torch_output = torch.cummin(torch_input, dim=case['dim'])
            ms_output = mint.cummin(ms_input, dim=case['dim'])
            
            error = calculate_error(torch_output, ms_output)
            status = "PASS" if error < 1e-3 else "FAIL"
            
            print(f"\nTest case: {case['name']}")
            print(f"Input shape: {case['data'].shape}")
            print(f"Error: {error:.6f}")
            print(f"Status: {status}")
            
            # 打印实际输出值以进行比较
            print("\nPyTorch output:")
            print(torch_output[0].numpy())
            print("\nMindSpore output:")
            print(ms_output[0].asnumpy())
            
        except Exception as e:
            print(f"\nTest case: {case['name']}")
            print(f"Error: {str(e)}")
            print("Status: ERROR")

def test_param_type_support():
    """
    测试不同参数类型的支持情况
    已知问题：一维张量的axis必须为0（详见ISSUES.md#cummin一维张量限制）
    """
    print("\n=== Parameter Type Support Test Results ===")
    
    # 准备基础数据
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    torch_input = torch.tensor(data)
    ms_input = Tensor(data)
    
    # 测试不同类型的参数
    test_cases = [
        {'name': 'String dim', 'dim': 'invalid', 'axis': 'invalid'},
        {'name': 'Bool dim', 'dim': True, 'axis': True},
        {'name': 'Float dim', 'dim': 0.5, 'axis': 0.5},
        {'name': 'Out of range dim', 'dim': 10, 'axis': 10},
        {'name': 'Negative dim', 'dim': -3, 'axis': -3},
    ]
    
    print("\n{:<20} {:<30} {:<30}".format(
        "Test Case", "PyTorch Error", "MindSpore Error"))
    print("-" * 80)
    
    for case in test_cases:
        torch_error = ""
        ms_error = ""
        
        # PyTorch测试
        try:
            _ = torch.cummin(torch_input, dim=case['dim'])
        except Exception as e:
            torch_error = str(e)
        
        # MindSpore测试
        try:
            _ = mint.cummin(ms_input, dim=case['axis'])
        except Exception as e:
            ms_error = str(e)
        
        print("{:<20} {:<30} {:<30}".format(
            case['name'],
            torch_error[:30],
            ms_error[:30]
        ))

def test_error_handling():
    """
    测试错误处理
    已知问题：维度范围限制与PyTorch不同（详见ISSUES.md#cummin维度范围限制）
    """
    print("\n=== Error Handling Test Results ===")
    
    # 创建一个基本的测试数据
    base_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)  
    
    test_cases = [
        {
            'name': 'Invalid dim type (string)',
            'desc': '测试无效的维度类型(字符串)',
            'dim': 'invalid'
        },
        {
            'name': 'Invalid dim type (bool)',
            'desc': '测试无效的维度类型(布尔)',
            'dim': True
        },
        {
            'name': 'Invalid dim type (float)',
            'desc': '测试无效的维度类型(浮点)',
            'dim': 0.5
        },
        {
            'name': 'Out of range dim',
            'desc': '测试超出范围的维度',
            'dim': 2  
        },
        {
            'name': 'Negative out of range',
            'desc': '测试负数超出范围的维度',
            'dim': -3  
        }
    ]
    
    print("\n{:<20} {:<30} {:<30}".format(
        "Test Case", "PyTorch Error", "MindSpore Error"))
    print("-" * 80)
    
    for case in test_cases:
        # 每次创建新的输入数据，避免复用可能导致的问题
        torch_input = torch.tensor(base_data, requires_grad=True)
        ms_input = Tensor(base_data)
        
        torch_result = "Success"
        ms_result = "Success"
        
        # PyTorch测试
        try:
            _ = torch.cummin(torch_input, dim=case['dim'])
        except Exception as e:
            torch_result = str(e)[:27]
        
        # MindSpore测试
        try:
            _ = mint.cummin(ms_input, dim=case['dim'])
        except Exception as e:
            ms_result = str(e)[:27]
        
        print("{:<20} {:<30} {:<30}".format(
            case['name'],
            torch_result,
            ms_result
        ))

def test_cummin_forward_backward():
    """
    测试cummin前向推理和反向梯度的准确性
    已知问题：梯度计算存在量化问题（详见ISSUES.md#cummin梯度计算问题）
    """
    print("\n=== Testing cummin Forward and Backward ===")
    
    # 定义一个简单的网络来计算cummin
    class CumminNet(ms.nn.Cell):
        def __init__(self, dim):
            super(CumminNet, self).__init__()
            self.dim = dim
            
        def construct(self, x):
            return mint.cummin(x, dim=self.dim)[0]
    
    # 定义梯度计算
    def get_grad(net, inputs):
        grad_op = ms.ops.GradOperation()
        gradient_function = grad_op(net)
        return gradient_function(inputs)
    
    # 测试用例配置
    test_cases = [
        {'shape': (4,), 'dim': 0},      
        {'shape': (3, 4), 'dim': 0},    
        {'shape': (3, 4), 'dim': 1},    
        {'shape': (2, 3, 4), 'dim': 0}, 
        {'shape': (2, 3, 4), 'dim': 1}, 
        {'shape': (2, 3, 4), 'dim': 2}  
    ]
    
    for case in test_cases:
        shape, dim = case['shape'], case['dim']
        print(f"\nTesting shape {shape}, dim {dim}")
        
        # 生成固定输入数据
        np.random.seed(42)
        data = np.random.randn(*shape).astype(np.float32)
        
        try:
            # PyTorch设置
            torch_input = torch.tensor(data, requires_grad=True)
            torch_output = torch.cummin(torch_input, dim=dim)[0]
            
            # MindSpore设置
            ms_input = Tensor(data, dtype=ms.float32)
            net = CumminNet(dim)
            ms_output = net(ms_input)
            
            # 测试前向推理结果
            forward_error = calculate_error(torch_output, ms_output)
            print(f"Forward error: {forward_error:.6f}")
            
            if forward_error < 1e-3:
                print("✓ Forward test passed")
            else:
                print("✗ Forward test failed")
                print(f"PyTorch output:\n{torch_output.detach().numpy()}")
                print(f"MindSpore output:\n{ms_output.asnumpy()}")
            
            # 测试反向梯度
            try:
                # 生成随机梯度
                grad_data = np.random.randn(*shape).astype(np.float32)
                
                # PyTorch反向传播
                torch_output.backward(torch.tensor(grad_data))
                torch_grad = torch_input.grad.detach().numpy()
                
                # MindSpore反向传播
                ms_grad = get_grad(net, ms_input)
                ms_grad = ms_grad.asnumpy()
                
                # 计算梯度误差
                grad_error = np.max(np.abs(torch_grad - ms_grad))
                print(f"Gradient error: {grad_error:.6f}")
                
                if grad_error < 1e-3:
                    print("✓ Backward test passed")
                else:
                    print("✗ Backward test failed")
                    print(f"PyTorch gradient:\n{torch_grad}")
                    print(f"MindSpore gradient:\n{ms_grad}")
            
            except Exception as e:
                print(f"✗ Backward test failed with error: {str(e)}")
                
        except Exception as e:
            print(f"✗ Test failed with error: {str(e)}")

def test_cummin_in_network():
    """测试在神经网络中使用cummin的情况"""
    print("\n=== Testing cummin in Neural Network ===")
    
    class TorchNet(torch.nn.Module):
        def forward(self, x):
            return torch.cummin(x, dim=1)[0]
    
    class MindSporeNet(ms.nn.Cell):
        def construct(self, x):
            return mint.cummin(x, dim=1)[0]
    
    def get_grad(net, inputs):
        grad_op = ms.ops.GradOperation()
        gradient_function = grad_op(net)
        return gradient_function(inputs)
    
    # 初始化网络
    torch_net = TorchNet()
    ms_net = MindSporeNet()
    
    # 生成固定输入数据
    np.random.seed(42)
    data = np.random.randn(2, 3, 4).astype(np.float32)
    
    # PyTorch前向传播
    torch_input = torch.tensor(data, requires_grad=True)
    torch_output = torch_net(torch_input)
    
    # MindSpore前向传播
    ms_input = Tensor(data, dtype=ms.float32)
    ms_output = ms_net(ms_input)
    
    # 测试前向推理结果
    forward_error = calculate_error(torch_output, ms_output)
    print(f"Forward error in network: {forward_error:.6f}")
    
    if forward_error < 1e-3:
        print("✓ Network forward test passed")
    else:
        print("✗ Network forward test failed")
    
    try:
        # 生成随机梯度
        grad_data = np.random.randn(*data.shape).astype(np.float32)
        
        # PyTorch反向传播
        torch_output.backward(torch.tensor(grad_data))
        torch_grad = torch_input.grad.detach().numpy()
        
        # MindSpore反向传播
        ms_grad = get_grad(ms_net, ms_input)
        ms_grad = ms_grad.asnumpy()
        
        # 计算梯度误差
        grad_error = np.max(np.abs(torch_grad - ms_grad))
        print(f"Gradient error in network: {grad_error:.6f}")
        
        if grad_error < 1e-3:
            print("✓ Network backward test passed")
        else:
            print("✗ Network backward test failed")
    
    except Exception as e:
        print(f"✗ Network backward test failed with error: {str(e)}")

if __name__ == "__main__":
    
    print("\n=== Testing random dtype support ===")
    test_random_dtype_support()
    
    print("\n=== Testing fixed dtype output accuracy ===")
    test_fixed_dtype_output_accuracy()
    
    print("\n=== Testing simple cases ===")
    test_simple_cases()
    
    print("\n=== Testing parameter type support ===")
    test_param_type_support()
    
    print("\n=== Testing error handling ===")
    test_error_handling()
    
    print("\n=== Testing cummin functionality ===")
    test_cummin_forward_backward()
    test_cummin_in_network()
