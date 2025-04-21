'''
dtype: 
    pytorch - float16, float32, float64
    mindspore - float16, float32 （和文档一致）
'''





import numpy as np
import torch
import torch.nn.functional as F
import mindspore as ms
from mindspore import Tensor
import mindspore.mint.nn.functional as mint_F
import traceback

# 设置全局精度容差
TOLERANCE = 1e-3

def print_header(title):
    print(f"\n{'='*80}\n{title}\n{'='*80}")

def compare_outputs(pytorch_out, mindspore_out, name="输出"):
    """比较两个框架的输出是否在容差范围内"""
    pytorch_np = pytorch_out.detach().cpu().numpy()
    mindspore_np = mindspore_out.asnumpy()
    
    max_diff = np.max(np.abs(pytorch_np - mindspore_np))
    mean_diff = np.mean(np.abs(pytorch_np - mindspore_np))
    
    print(f"{name} 最大差异: {max_diff}")
    print(f"{name} 平均差异: {mean_diff}")
    
    if max_diff < TOLERANCE:
        print(f"✓ {name}在容差范围内一致 (< {TOLERANCE})")
        return True
    else:
        print(f"✗ {name}超出容差范围 (> {TOLERANCE})")
        return False

def test_dtype_support():
    """测试不同数据类型的支持度"""
    print_header("1.a) 测试不同数据类型(dtype)的支持度")
    
    shape = (3, 4)
    dtypes_pytorch = [torch.float16, torch.float32, torch.float64, torch.bfloat16]
    dtypes_mindspore = [ms.float16, ms.float32, ms.float64, ms.bfloat16]
    
    for pt_dtype, ms_dtype in zip(dtypes_pytorch, dtypes_mindspore):
        print(f"\n测试数据类型: PyTorch {pt_dtype}, MindSpore {ms_dtype}")
        
        # 生成随机输入 (0-1之间的值用于BCE)
        np_input = np.random.random(shape).astype(np.float32)
        np_target = np.random.random(shape).astype(np.float32)
        
        try:
            # PyTorch
            pt_input = torch.tensor(np_input, dtype=pt_dtype)
            pt_target = torch.tensor(np_target, dtype=pt_dtype)
            pt_output = F.binary_cross_entropy(pt_input, pt_target, reduction='mean')
            print(f"PyTorch 输出: {pt_output.item()}, shape: {pt_output.shape}")
            pt_support = "支持"
        except Exception as e:
            print(f"PyTorch 错误: {str(e)}")
            pt_support = "不支持"
            
        try:
            # MindSpore
            ms_input = Tensor(np_input, dtype=ms_dtype)
            ms_target = Tensor(np_target, dtype=ms_dtype)
            ms_output = mint_F.binary_cross_entropy(ms_input, ms_target, reduction='mean')
            print(f"MindSpore 输出: {ms_output.asnumpy().item()}, shape: {ms_output.shape}")
            ms_support = "支持"
        except Exception as e:
            print(f"MindSpore 错误: {str(e)}")
            ms_support = "不支持"
            
        print(f"PyTorch: {pt_support}, MindSpore: {ms_support}")

def test_random_inputs():
    """测试随机输入值的输出一致性"""
    print_header("1.b) 测试随机输入值的输出一致性")
    
    shapes = [(2, 3), (3, 4, 5), (2, 3, 4, 5)]
    
    for shape in shapes:
        print(f"\n测试shape: {shape}")
        
        # 生成随机输入 (0-1之间的值用于BCE)
        np_input = np.random.random(shape).astype(np.float32)
        np_target = np.random.random(shape).astype(np.float32)
        
        # PyTorch
        pt_input = torch.tensor(np_input, requires_grad=True)
        pt_target = torch.tensor(np_target)
        pt_output = F.binary_cross_entropy(pt_input, pt_target, reduction='mean')
        print(f"PyTorch 输出: {pt_output.item()}")
        
        # MindSpore
        ms_input = Tensor(np_input, dtype=ms.float32)
        ms_target = Tensor(np_target, dtype=ms.float32)
        ms_output = mint_F.binary_cross_entropy(ms_input, ms_target, reduction='mean')
        print(f"MindSpore 输出: {ms_output.asnumpy().item()}")
        
        compare_outputs(pt_output, ms_output)
        
        # 测试不同的reduction
        for reduction in ['none', 'sum', 'mean']:
            print(f"\n测试reduction: {reduction}")
            
            pt_output = F.binary_cross_entropy(pt_input, pt_target, reduction=reduction)
            ms_output = mint_F.binary_cross_entropy(ms_input, ms_target, reduction=reduction)
            
            compare_outputs(pt_output, ms_output, f"reduction={reduction}")

def test_param_support():
    """测试不同参数类型的支持度"""
    print_header("1.c) 测试不同参数类型的支持度")
    
    shape = (3, 4)
    np_input = np.random.random(shape).astype(np.float32)
    np_target = np.random.random(shape).astype(np.float32)
    np_weight = np.random.random(shape).astype(np.float32)
    
    # 基本输入
    pt_input = torch.tensor(np_input)
    pt_target = torch.tensor(np_target)
    pt_weight = torch.tensor(np_weight)
    
    ms_input = Tensor(np_input, dtype=ms.float32)
    ms_target = Tensor(np_target, dtype=ms.float32)
    ms_weight = Tensor(np_weight, dtype=ms.float32)
    
    # 测试不同reduction参数
    reductions = ['none', 'sum', 'mean', 'INVALID']
    
    for reduction in reductions:
        print(f"\n测试reduction参数: '{reduction}'")
        
        try:
            pt_output = F.binary_cross_entropy(pt_input, pt_target, weight=pt_weight, reduction=reduction)
            print(f"PyTorch: 支持 reduction='{reduction}'")
        except Exception as e:
            print(f"PyTorch 错误: {str(e)}")
        
        try:
            ms_output = mint_F.binary_cross_entropy(ms_input, ms_target, weight=ms_weight, reduction=reduction)
            print(f"MindSpore: 支持 reduction='{reduction}'")
        except Exception as e:
            print(f"MindSpore 错误: {str(e)}")
    
    # 测试权重参数
    print("\n测试weight参数:")
    
    # 有权重
    try:
        pt_output = F.binary_cross_entropy(pt_input, pt_target, weight=pt_weight)
        print("PyTorch: 支持带权重")
    except Exception as e:
        print(f"PyTorch 错误: {str(e)}")
    
    try:
        ms_output = mint_F.binary_cross_entropy(ms_input, ms_target, weight=ms_weight)
        print("MindSpore: 支持带权重")
    except Exception as e:
        print(f"MindSpore 错误: {str(e)}")
    
    # 无权重
    try:
        pt_output = F.binary_cross_entropy(pt_input, pt_target, weight=None)
        print("PyTorch: 支持无权重")
    except Exception as e:
        print(f"PyTorch 错误: {str(e)}")
    
    try:
        ms_output = mint_F.binary_cross_entropy(ms_input, ms_target, weight=None)
        print("MindSpore: 支持无权重")
    except Exception as e:
        print(f"MindSpore 错误: {str(e)}")

def test_error_handling():
    """测试错误处理的准确性"""
    print_header("1.d) 测试错误处理的准确性")
    
    # 测试输入和目标形状不匹配
    print("\n测试输入和目标形状不匹配:")
    
    pt_input = torch.rand(2, 3)
    pt_target = torch.rand(3, 2)
    
    ms_input = Tensor(np.random.random((2, 3)), dtype=ms.float32)
    ms_target = Tensor(np.random.random((3, 2)), dtype=ms.float32)
    
    try:
        pt_output = F.binary_cross_entropy(pt_input, pt_target)
        print("PyTorch结果:", pt_output.item())
    except Exception as e:
        print(f"PyTorch错误: {str(e)}")
    
    try:
        ms_output = mint_F.binary_cross_entropy(ms_input, ms_target)
        print("MindSpore结果:", ms_output.asnumpy().item())
    except Exception as e:
        print(f"MindSpore错误: {str(e)}")
    
    # 测试错误的输入类型
    print("\n测试错误的输入类型:")
    
    # 字符串输入
    try:
        pt_output = F.binary_cross_entropy(pt_input, "wrong_target")
        print("PyTorch支持字符串输入")
    except Exception as e:
        print(f"PyTorch错误: {type(e).__name__}: {str(e)}")
    
    try:
        ms_output = mint_F.binary_cross_entropy(ms_input, "wrong_target")
        print("MindSpore支持字符串输入")
    except Exception as e:
        print(f"MindSpore错误: {type(e).__name__}: {str(e)}")
    
    # 测试超出范围的值
    print("\n测试超出[0,1]范围的值:")
    
    pt_bad_input = torch.tensor([[-1.0, 0.5], [0.8, 2.0]])
    pt_bad_target = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    
    ms_bad_input = Tensor(np.array([[-1.0, 0.5], [0.8, 2.0]]), dtype=ms.float32)
    ms_bad_target = Tensor(np.array([[0.1, 0.2], [0.3, 0.4]]), dtype=ms.float32)
    
    try:
        pt_output = F.binary_cross_entropy(pt_bad_input, pt_bad_target)
        print(f"PyTorch结果: {pt_output.item()}")
    except Exception as e:
        print(f"PyTorch错误: {str(e)}")
    
    try:
        ms_output = mint_F.binary_cross_entropy(ms_bad_input, ms_bad_target)
        print(f"MindSpore结果: {ms_output.asnumpy().item()}")
    except Exception as e:
        print(f"MindSpore错误: {str(e)}")

def test_nn_implementation():
    """测试神经网络实现"""
    print_header("2.a/b) 测试神经网络实现和推理结果")
    
    # 简单的二分类网络
    class PTBinaryClassifier(torch.nn.Module):
        def __init__(self, input_dim):
            super(PTBinaryClassifier, self).__init__()
            self.linear = torch.nn.Linear(input_dim, 1)
            self.sigmoid = torch.nn.Sigmoid()
            
        def forward(self, x, target=None):
            output = self.sigmoid(self.linear(x))
            if target is not None:
                loss = F.binary_cross_entropy(output, target)
                return loss
            return output
    
    class MSBinaryClassifier(ms.nn.Cell):
        def __init__(self, input_dim):
            super(MSBinaryClassifier, self).__init__()
            self.linear = ms.nn.Dense(input_dim, 1)
            self.sigmoid = ms.nn.Sigmoid()
            
        def construct(self, x, target=None):
            output = self.sigmoid(self.linear(x))
            if target is not None:
                loss = mint_F.binary_cross_entropy(output, target)
                return loss
            return output
    
    # 固定输入和权重
    input_dim = 5
    batch_size = 3
    
    np_input = np.random.random((batch_size, input_dim)).astype(np.float32)
    np_target = np.random.random((batch_size, 1)).astype(np.float32)
    
    # 创建模型
    pt_model = PTBinaryClassifier(input_dim)
    ms_model = MSBinaryClassifier(input_dim)
    
    # 固定权重
    np_weight = np.random.random((input_dim, 1)).astype(np.float32)
    np_bias = np.random.random(1).astype(np.float32)
    
    pt_model.linear.weight.data = torch.tensor(np_weight).t()
    pt_model.linear.bias.data = torch.tensor(np_bias)
    
    ms_model.linear.weight.set_data(Tensor(np_weight.T, dtype=ms.float32))
    ms_model.linear.bias.set_data(Tensor(np_bias, dtype=ms.float32))
    
    # 前向传播测试
    pt_input = torch.tensor(np_input)
    pt_target = torch.tensor(np_target)
    
    ms_input = Tensor(np_input, dtype=ms.float32)
    ms_target = Tensor(np_target, dtype=ms.float32)
    
    # 输出测试
    pt_output = pt_model(pt_input)
    ms_output = ms_model(ms_input)
    
    print("测试模型输出:")
    compare_outputs(pt_output, ms_output, "模型输出")
    
    # 损失测试
    pt_loss = pt_model(pt_input, pt_target)
    ms_loss = ms_model(ms_input, ms_target)
    
    print("\n测试模型损失:")
    compare_outputs(pt_loss, ms_loss, "损失值")

def test_gradient():
    """测试反向传播和梯度计算"""
    print_header("2.c) 测试反向传播和梯度计算")
    
    # 函数的梯度测试
    shape = (3, 4)
    np_input = np.random.random(shape).astype(np.float32)
    np_target = np.random.random(shape).astype(np.float32)
    
    # PyTorch
    pt_input = torch.tensor(np_input, requires_grad=True)
    pt_target = torch.tensor(np_target)
    pt_output = F.binary_cross_entropy(pt_input, pt_target)
    pt_output.backward()
    pt_grad = pt_input.grad
    
    print("PyTorch梯度:")
    print(f"形状: {pt_grad.shape}")
    print(f"平均值: {pt_grad.mean().item()}")
    
    # MindSpore - 创建一个计算图用于计算梯度
    ms_input = ms.Parameter(Tensor(np_input, dtype=ms.float32))
    ms_target = Tensor(np_target, dtype=ms.float32)
    
    def forward_fn(x, target):
        return mint_F.binary_cross_entropy(x, target)
    
    grad_fn = ms.grad(forward_fn)
    ms_grad = grad_fn(ms_input, ms_target)
    
    print("\nMindSpore梯度:")
    print(f"形状: {ms_grad.shape}")
    print(f"平均值: {ms_grad.asnumpy().mean()}")
    
    # 比较梯度
    compare_outputs(pt_grad, ms_grad, "梯度")
    
    # 神经网络的参数梯度测试
    print("\n测试神经网络参数梯度:")
    
    input_dim = 5
    hidden_dim = 3
    batch_size = 4
    
    class PTSimpleNet(torch.nn.Module):
        def __init__(self):
            super(PTSimpleNet, self).__init__()
            self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
            self.fc2 = torch.nn.Linear(hidden_dim, 1)
            self.sigmoid = torch.nn.Sigmoid()
            
        def forward(self, x, target=None):
            h = torch.nn.functional.relu(self.fc1(x))
            output = self.sigmoid(self.fc2(h))
            if target is not None:
                loss = F.binary_cross_entropy(output, target)
                return loss
            return output
    
    class MSSimpleNet(ms.nn.Cell):
        def __init__(self):
            super(MSSimpleNet, self).__init__()
            self.fc1 = ms.nn.Dense(input_dim, hidden_dim)
            self.fc2 = ms.nn.Dense(hidden_dim, 1)
            self.relu = ms.nn.ReLU()
            self.sigmoid = ms.nn.Sigmoid()
            
        def construct(self, x, target=None):
            h = self.relu(self.fc1(x))
            output = self.sigmoid(self.fc2(h))
            if target is not None:
                loss = mint_F.binary_cross_entropy(output, target)
                return loss
            return output
    
    # 创建模型
    pt_net = PTSimpleNet()
    ms_net = MSSimpleNet()
    
    # 固定权重
    np_fc1_weight = np.random.random((hidden_dim, input_dim)).astype(np.float32)
    np_fc1_bias = np.random.random(hidden_dim).astype(np.float32)
    np_fc2_weight = np.random.random((1, hidden_dim)).astype(np.float32)
    np_fc2_bias = np.random.random(1).astype(np.float32)
    
    pt_net.fc1.weight.data = torch.tensor(np_fc1_weight)
    pt_net.fc1.bias.data = torch.tensor(np_fc1_bias)
    pt_net.fc2.weight.data = torch.tensor(np_fc2_weight)
    pt_net.fc2.bias.data = torch.tensor(np_fc2_bias)
    
    ms_net.fc1.weight.set_data(Tensor(np_fc1_weight, dtype=ms.float32))
    ms_net.fc1.bias.set_data(Tensor(np_fc1_bias, dtype=ms.float32))
    ms_net.fc2.weight.set_data(Tensor(np_fc2_weight, dtype=ms.float32))
    ms_net.fc2.bias.set_data(Tensor(np_fc2_bias, dtype=ms.float32))
    
    # 准备输入和目标
    np_net_input = np.random.random((batch_size, input_dim)).astype(np.float32)
    np_net_target = np.random.random((batch_size, 1)).astype(np.float32)
    
    pt_net_input = torch.tensor(np_net_input)
    pt_net_target = torch.tensor(np_net_target)
    
    ms_net_input = Tensor(np_net_input, dtype=ms.float32)
    ms_net_target = Tensor(np_net_target, dtype=ms.float32)
    
    # PyTorch计算梯度
    pt_optimizer = torch.optim.SGD(pt_net.parameters(), lr=0.1)
    pt_optimizer.zero_grad()
    pt_loss = pt_net(pt_net_input, pt_net_target)
    pt_loss.backward()
    
    # 获取PyTorch梯度
    pt_fc1_weight_grad = pt_net.fc1.weight.grad.numpy()
    pt_fc1_bias_grad = pt_net.fc1.bias.grad.numpy()
    pt_fc2_weight_grad = pt_net.fc2.weight.grad.numpy()
    pt_fc2_bias_grad = pt_net.fc2.bias.grad.numpy()
    
    print("PyTorch网络梯度:")
    print(f"fc1.weight梯度平均值: {pt_fc1_weight_grad.mean()}")
    print(f"fc1.bias梯度平均值: {pt_fc1_bias_grad.mean()}")
    print(f"fc2.weight梯度平均值: {pt_fc2_weight_grad.mean()}")
    print(f"fc2.bias梯度平均值: {pt_fc2_bias_grad.mean()}")
    
    # MindSpore计算梯度
    def ms_forward_fn(inputs, targets):
        return ms_net(inputs, targets)
    
    ms_grad_fn = ms.value_and_grad(ms_forward_fn, None, ms_net.trainable_params())
    ms_loss, ms_grads = ms_grad_fn(ms_net_input, ms_net_target)
    
    print("\nMindSpore网络梯度:")
    for i, param in enumerate(ms_net.trainable_params()):
        print(f"{param.name} 梯度平均值: {ms_grads[i].asnumpy().mean()}")
        
    # 比较具体参数的梯度 (注意可能需要转置比较)
    print("\n比较fc2.weight梯度:")
    ms_fc2_weight_grad = None
    for i, param in enumerate(ms_net.trainable_params()):
        if 'fc2.weight' in param.name:
            ms_fc2_weight_grad = ms_grads[i]
            break
    
    if ms_fc2_weight_grad is not None:
        # 注意MindSpore和PyTorch的权重矩阵可能需要转置后比较
        ms_fc2_weight_grad_np = ms_fc2_weight_grad.asnumpy()
        max_diff = np.max(np.abs(ms_fc2_weight_grad_np - pt_fc2_weight_grad))
        print(f"fc2.weight梯度最大差异: {max_diff}")
        if max_diff < TOLERANCE:
            print(f"✓ fc2.weight梯度在容差范围内一致 (< {TOLERANCE})")
        else:
            print(f"✗ fc2.weight梯度超出容差范围 (> {TOLERANCE})")

if __name__ == "__main__":
    # 运行所有测试
    test_dtype_support()
    test_random_inputs()
    test_param_support()
    # test_error_handling()
    test_nn_implementation()
    test_gradient()
    test_error_handling()
