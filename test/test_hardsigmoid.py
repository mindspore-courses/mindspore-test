import pytest
import numpy as np
import mindspore as ms
from mindspore import Tensor
import mindspore.mint as mint
import torch
import torch.nn as nn

def test_hardsigmoid_random_dtype():
    """
    测试不同数据类型的支持度对比
    """
    x = np.random.uniform(-1, 1, size=(3, 4, 5)).astype(np.float32)
    
    # MindSpore支持的数据类型测试
    dtypes_ms = [ms.float16, ms.float32, ms.bfloat16]
    ms_hardsigmoid = mint.nn.Hardsigmoid()
    
    for dtype in dtypes_ms:
        try:
            x_ms = Tensor(x, dtype=dtype)
            y_ms = ms_hardsigmoid(x_ms)
            print(f"MindSpore成功支持数据类型: {dtype}")
        except Exception as e:
            print(f"MindSpore在数据类型{dtype}下的错误: {str(e)}")
    
    # PyTorch支持的数据类型测试
    dtypes_torch = [torch.float16, torch.float32, torch.float64]
    torch_hardsigmoid = nn.Hardsigmoid()
    
    for dtype in dtypes_torch:
        try:
            x_torch = torch.tensor(x, dtype=dtype)
            y_torch = torch_hardsigmoid(x_torch)
            print(f"PyTorch成功支持数据类型: {dtype}")
        except Exception as e:
            print(f"PyTorch在数据类型{dtype}下的错误: {str(e)}")

def test_hardsigmoid_random_value():
    """
    测试随机输入值的精度对比
    """
    x = np.random.uniform(-1, 1, size=(3, 4, 5)).astype(np.float32)
    
    # MindSpore测试
    x_ms = Tensor(x, dtype=ms.float32)
    ms_hardsigmoid = mint.nn.Hardsigmoid()
    y_ms = ms_hardsigmoid(x_ms)
    
    # PyTorch测试
    x_torch = torch.tensor(x, dtype=torch.float32)
    torch_hardsigmoid = nn.Hardsigmoid()
    y_torch = torch_hardsigmoid(x_torch)
    
    # 计算误差
    error = np.abs(y_ms.asnumpy() - y_torch.detach().numpy())
    max_error = np.max(error)
    assert max_error < 1e-3, f"输出误差 {max_error} 超过阈值 1e-3"

def test_hardsigmoid_invalid_inputs():
    """
    测试无效输入的错误处理
    """
    ms_hardsigmoid = mint.nn.Hardsigmoid()
    
    # 测试字符串输入
    try:
        x_str = Tensor("invalid", dtype=ms.float32)
        y_ms = ms_hardsigmoid(x_str)
    except Exception as e:
        print(f"字符串输入错误信息: {str(e)}")
    
    # 测试布尔值输入
    try:
        x_bool = Tensor([True, False], dtype=ms.bool_)
        y_ms = ms_hardsigmoid(x_bool)
    except Exception as e:
        print(f"布尔值输入错误信息: {str(e)}")

def test_hardsigmoid_gradient():
    """
    测试梯度计算的准确性
    """
    # 固定输入和权重
    x = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    
    # MindSpore梯度测试
    x_ms = Tensor(x, dtype=ms.float32)
    ms_hardsigmoid = mint.nn.Hardsigmoid()
    
    # PyTorch梯度测试
    x_torch = torch.tensor(x, requires_grad=True)
    torch_hardsigmoid = nn.Hardsigmoid()
    
    # 计算前向结果
    y_ms = ms_hardsigmoid(x_ms)
    y_torch = torch_hardsigmoid(x_torch)
    
    # 计算梯度
    y_torch.sum().backward()
    grad_torch = x_torch.grad.numpy()
    
    # 比较梯度
    print(f"PyTorch梯度: {grad_torch}")
    # MindSpore的梯度计算需要使用GradOperation，这里仅作示例

def test_hardsigmoid_chaos_inputs():
    """
    测试各种混乱输入的错误处理
    """
    ms_hardsigmoid = mint.nn.Hardsigmoid()
    torch_hardsigmoid = nn.Hardsigmoid()
    
    # 测试None输入
    try:
        y_ms = ms_hardsigmoid(None)
    except Exception as e:
        print(f"MindSpore处理None输入的错误信息: {str(e)}")
    
    try:
        y_torch = torch_hardsigmoid(None)
    except Exception as e:
        print(f"PyTorch处理None输入的错误信息: {str(e)}")
    
    # 测试空Tensor
    try:
        x_ms = Tensor([], dtype=ms.float32)
        y_ms = ms_hardsigmoid(x_ms)
    except Exception as e:
        print(f"MindSpore处理空Tensor的错误信息: {str(e)}")
    
    try:
        x_torch = torch.tensor([], dtype=torch.float32)
        y_torch = torch_hardsigmoid(x_torch)
    except Exception as e:
        print(f"PyTorch处理空Tensor的错误信息: {str(e)}")
    
    # 测试异常shape
    try:
        x_ms = Tensor(np.array([[[1]]]), dtype=ms.float32)
        y_ms = ms_hardsigmoid(x_ms)
    except Exception as e:
        print(f"MindSpore处理异常shape的错误信息: {str(e)}")

def test_hardsigmoid_in_network():
    """
    在简单神经网络中测试Hardsigmoid
    参考自MindSpore官方示例
    """
    class SimpleNet(ms.nn.Cell):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.fc = ms.nn.Dense(10, 5)
            self.hardsigmoid = mint.nn.Hardsigmoid()
        
        def construct(self, x):
            x = self.fc(x)
            return self.hardsigmoid(x)
    
    class TorchSimpleNet(nn.Module):
        def __init__(self):
            super(TorchSimpleNet, self).__init__()
            self.fc = nn.Linear(10, 5)
            self.hardsigmoid = nn.Hardsigmoid()
        
        def forward(self, x):
            x = self.fc(x)
            return self.hardsigmoid(x)
    
    # 固定随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    ms.set_seed(42)
    
    # 创建网络实例
    net_ms = SimpleNet()
    net_torch = TorchSimpleNet()
    
    # 固定权重
    weight = np.random.randn(5, 10).astype(np.float32)
    bias = np.random.randn(5).astype(np.float32)
    
    # 设置MindSpore权重
    net_ms.fc.weight.set_data(Tensor(weight))
    net_ms.fc.bias.set_data(Tensor(bias))
    
    # 设置PyTorch权重
    with torch.no_grad():
        net_torch.fc.weight.copy_(torch.tensor(weight))
        net_torch.fc.bias.copy_(torch.tensor(bias))
    
    # 固定输入
    x = np.random.randn(32, 10).astype(np.float32)
    x_ms = Tensor(x)
    x_torch = torch.tensor(x)
    
    # 前向推理
    y_ms = net_ms(x_ms)
    y_torch = net_torch(x_torch)
    
    # 检查误差
    error = np.abs(y_ms.asnumpy() - y_torch.detach().numpy())
    max_error = np.max(error)
    assert max_error < 1e-3, f"网络输出误差 {max_error} 超过阈值 1e-3"

def test_hardsigmoid_gradient_complete():
    """
    完整的梯度测试，包括在网络中的梯度计算
    """
    class SimpleNet(ms.nn.Cell):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.hardsigmoid = mint.nn.Hardsigmoid()
        
        def construct(self, x):
            return self.hardsigmoid(x)
    
    # 固定输入
    x = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    
    # MindSpore梯度测试
    net_ms = SimpleNet()
    grad_ms = ms.ops.GradOperation(get_all=True)(net_ms)
    x_ms = Tensor(x, dtype=ms.float32)
    ms_grads = grad_ms(x_ms)
    
    # PyTorch梯度测试
    x_torch = torch.tensor(x, requires_grad=True)
    torch_hardsigmoid = nn.Hardsigmoid()
    y_torch = torch_hardsigmoid(x_torch)
    y_torch.sum().backward()
    torch_grads = x_torch.grad.numpy()
    
    # 比较梯度
    error = np.abs(ms_grads[0].asnumpy() - torch_grads)
    max_error = np.max(error)
    assert max_error < 1e-3, f"梯度误差 {max_error} 超过阈值 1e-3"

def test_hardsigmoid_different_shapes():
    """
    测试不同shape的输入
    """
    shapes = [
        (1,),           # 一维
        (2, 3),         # 二维
        (4, 5, 6),      # 三维
        (2, 3, 4, 5),   # 四维
        (32, 1, 1, 1),  # 带1的维度
        (1, 1, 1, 1),   # 全1维度
    ]
    
    ms_hardsigmoid = mint.nn.Hardsigmoid()
    torch_hardsigmoid = nn.Hardsigmoid()
    
    for shape in shapes:
        # 生成随机数据
        x = np.random.uniform(-1, 1, shape).astype(np.float32)
        
        # MindSpore测试
        x_ms = Tensor(x, dtype=ms.float32)
        y_ms = ms_hardsigmoid(x_ms)
        
        # PyTorch测试
        x_torch = torch.tensor(x, dtype=torch.float32)
        y_torch = torch_hardsigmoid(x_torch)
        
        # 检查形状
        assert y_ms.shape == y_torch.shape, f"shape {shape} 输出形状不一致"
        
        # 检查误差
        error = np.abs(y_ms.asnumpy() - y_torch.detach().numpy())
        max_error = np.max(error)
        assert max_error < 1e-3, f"shape {shape} 输出误差 {max_error} 超过阈值 1e-3"

def test_hardsigmoid_boundary_values():
    """
    测试边界值和特殊值
    """
    # 准备特殊值
    special_values = np.array([
        -3.0,           # 下边界
        3.0,            # 上边界
        0.0,            # 中间值
        -3.1,           # 小于下边界
        3.1,            # 大于上边界
        1e-7,           # 接近0的正数
        -1e-7,          # 接近0的负数
        np.inf,         # 正无穷
        -np.inf,        # 负无穷
        np.nan          # NaN
    ]).astype(np.float32)
    
    ms_hardsigmoid = mint.nn.Hardsigmoid()
    torch_hardsigmoid = nn.Hardsigmoid()
    
    # MindSpore测试
    x_ms = Tensor(special_values, dtype=ms.float32)
    try:
        y_ms = ms_hardsigmoid(x_ms)
        print("MindSpore特殊值输出:", y_ms)
    except Exception as e:
        print("MindSpore特殊值处理错误:", str(e))
    
    # PyTorch测试
    x_torch = torch.tensor(special_values, dtype=torch.float32)
    try:
        y_torch = torch_hardsigmoid(x_torch)
        print("PyTorch特殊值输出:", y_torch)
    except Exception as e:
        print("PyTorch特殊值处理错误:", str(e))

def test_hardsigmoid_dynamic_shape():
    """
    测试动态shape
    """
    ms_hardsigmoid = mint.nn.Hardsigmoid()
    torch_hardsigmoid = nn.Hardsigmoid()
    
    # 测试不同batch size
    batch_sizes = [1, 16, 32, 64, 128]
    for batch_size in batch_sizes:
        x = np.random.uniform(-1, 1, (batch_size, 10)).astype(np.float32)
        
        # MindSpore测试
        x_ms = Tensor(x, dtype=ms.float32)
        y_ms = ms_hardsigmoid(x_ms)
        
        # PyTorch测试
        x_torch = torch.tensor(x, dtype=torch.float32)
        y_torch = torch_hardsigmoid(x_torch)
        
        # 检查结果
        error = np.abs(y_ms.asnumpy() - y_torch.detach().numpy())
        max_error = np.max(error)
        assert max_error < 1e-3, f"batch_size {batch_size} 输出误差 {max_error} 超过阈值 1e-3"

def test_hardsigmoid_extreme_values():
    """
    测试极端情况下的数值
    """
    # 准备极端值
    extreme_values = np.array([
        1e-30,          # 极小正数
        -1e-30,         # 极小负数
        1e30,           # 极大正数
        -1e30,          # 极大负数
        np.finfo(np.float32).max,    # float32最大值
        np.finfo(np.float32).min,    # float32最小值
        np.finfo(np.float32).eps,    # float32最小精度
    ]).astype(np.float32)
    
    ms_hardsigmoid = mint.nn.Hardsigmoid()
    torch_hardsigmoid = nn.Hardsigmoid()
    
    # MindSpore测试
    x_ms = Tensor(extreme_values, dtype=ms.float32)
    try:
        y_ms = ms_hardsigmoid(x_ms)
        print("MindSpore极端值输出:", y_ms)
    except Exception as e:
        print("MindSpore极端值处理错误:", str(e))
    
    # PyTorch测试
    x_torch = torch.tensor(extreme_values, dtype=torch.float32)
    try:
        y_torch = torch_hardsigmoid(x_torch)
        print("PyTorch极端值输出:", y_torch)
    except Exception as e:
        print("PyTorch极端值处理错误:", str(e))

def test_hardsigmoid_broadcast():
    """
    测试广播机制
    """
    ms_hardsigmoid = mint.nn.Hardsigmoid()
    torch_hardsigmoid = nn.Hardsigmoid()
    
    # 测试不同的广播场景
    shapes = [
        ((1, 3), (4, 1, 3)),    # 维度扩展
        ((5, 1, 3), (5, 4, 3)), # 维度广播
        ((1, 1, 1), (2, 3, 4)), # 全广播
    ]
    
    for shape1, shape2 in shapes:
        # 生成数据
        x1 = np.random.uniform(-1, 1, shape1).astype(np.float32)
        x2 = np.broadcast_to(x1, shape2)
        
        # MindSpore测试
        x1_ms = Tensor(x1, dtype=ms.float32)
        x2_ms = Tensor(x2, dtype=ms.float32)
        y1_ms = ms_hardsigmoid(x1_ms)
        y2_ms = ms_hardsigmoid(x2_ms)
        
        # PyTorch测试
        x1_torch = torch.tensor(x1, dtype=torch.float32)
        x2_torch = torch.tensor(x2, dtype=torch.float32)
        y1_torch = torch_hardsigmoid(x1_torch)
        y2_torch = torch_hardsigmoid(x2_torch)
        
        # 检查广播结果
        error1 = np.abs(y1_ms.asnumpy() - y1_torch.detach().numpy())
        error2 = np.abs(y2_ms.asnumpy() - y2_torch.detach().numpy())
        max_error = max(np.max(error1), np.max(error2))
        assert max_error < 1e-3, f"广播测试 shapes {shape1}->{shape2} 输出误差 {max_error} 超过阈值 1e-3"
