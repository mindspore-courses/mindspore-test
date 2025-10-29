'''
dtype: 
    pytorch - float16, float32, float64
    mindspore - float16, float32, bfloat16
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

# 定义 MindSpore 的 mse_loss 替代函数，以防原始函数不可用
def mindspore_mse_loss(inputs, targets, reduction='mean'):
    squared_diff = ms.ops.square(inputs - targets)
    if reduction == 'none':
        return squared_diff
    elif reduction == 'mean':
        return ms.ops.mean(squared_diff)
    elif reduction == 'sum':
        return ms.ops.sum(squared_diff)
    else:
        raise ValueError(f"Unsupported reduction mode: {reduction}")

def test_dtype_support():
    """测试不同数据类型的支持度"""
    print_header("1.a) 测试不同数据类型(dtype)的支持度")
    
    shape = (3, 4)
    dtypes_pytorch = [torch.float16, torch.float32, torch.float64, torch.bfloat16]
    dtypes_mindspore = [ms.float16, ms.float32, ms.float64, ms.bfloat16]
    dtype_names = ["float16", "float32", "float64", "bfloat16"]
    
    for pt_dtype, ms_dtype, dtype_name in zip(dtypes_pytorch, dtypes_mindspore, dtype_names):
        print(f"\n测试数据类型: PyTorch {dtype_name}, MindSpore {dtype_name}")
        
        # 生成随机输入
        np_input = np.random.randn(*shape).astype(np.float32)
        np_target = np.random.randn(*shape).astype(np.float32)
        
        try:
            # PyTorch
            pt_input = torch.tensor(np_input, dtype=pt_dtype)
            pt_target = torch.tensor(np_target, dtype=pt_dtype)
            pt_output = F.mse_loss(pt_input, pt_target, reduction='mean')
            print(f"PyTorch 输出: {pt_output.item()}, shape: {pt_output.shape}")
            pt_support = "支持"
        except Exception as e:
            print(f"PyTorch 错误: {type(e).__name__}: {str(e)}")
            pt_support = "不支持"
            
        try:
            # 首先尝试 mint.nn.functional.mse_loss
            ms_input = Tensor(np_input, dtype=ms_dtype)
            ms_target = Tensor(np_target, dtype=ms_dtype)
            ms_using_mint = True
            
            try:
                ms_output = mint_F.mse_loss(ms_input, ms_target, reduction='mean')
            except (AttributeError, RuntimeError) as e:
                print(f"mint.nn.functional.mse_loss 不可用: {type(e).__name__}: {str(e)}")
                print("使用替代实现...")
                ms_output = mindspore_mse_loss(ms_input, ms_target, reduction='mean')
                ms_using_mint = False
            
            print(f"MindSpore 输出 ({'mint API' if ms_using_mint else '替代实现'}): {ms_output.asnumpy().item()}, shape: {ms_output.shape}")
            ms_support = "支持"
        except Exception as e:
            print(f"MindSpore 错误: {type(e).__name__}: {str(e)}")
            ms_support = "不支持"
            
        print(f"PyTorch {dtype_name}: {pt_support}, MindSpore {dtype_name}: {ms_support}")

def test_random_inputs():
    """测试随机输入值的输出一致性"""
    print_header("1.b) 测试随机输入值的输出一致性")
    
    shapes = [(2, 3), (3, 4, 5), (2, 3, 4, 5)]
    
    # 检查mint API是否可用
    ms_using_mint = True
    try:
        _ = mint_F.mse_loss
    except (AttributeError, RuntimeError):
        ms_using_mint = False
        print("mint.nn.functional.mse_loss 不可用，使用替代实现...")
    
    mse_loss_fn = mint_F.mse_loss if ms_using_mint else mindspore_mse_loss
    
    for shape in shapes:
        print(f"\n测试shape: {shape}")
        
        # 生成随机输入
        np_input = np.random.randn(*shape).astype(np.float32)
        np_target = np.random.randn(*shape).astype(np.float32)
        
        # PyTorch
        pt_input = torch.tensor(np_input, requires_grad=True)
        pt_target = torch.tensor(np_target)
        pt_output = F.mse_loss(pt_input, pt_target, reduction='mean')
        print(f"PyTorch 输出: {pt_output.item()}")
        
        # MindSpore
        ms_input = Tensor(np_input, dtype=ms.float32)
        ms_target = Tensor(np_target, dtype=ms.float32)
        ms_output = mse_loss_fn(ms_input, ms_target, reduction='mean')
        print(f"MindSpore 输出 ({'mint API' if ms_using_mint else '替代实现'}): {ms_output.asnumpy().item()}")
        
        compare_outputs(pt_output, ms_output)
        
        # 测试不同的reduction
        for reduction in ['none', 'sum', 'mean']:
            print(f"\n测试reduction: {reduction}")
            
            pt_output = F.mse_loss(pt_input, pt_target, reduction=reduction)
            ms_output = mse_loss_fn(ms_input, ms_target, reduction=reduction)
            
            compare_outputs(pt_output, ms_output, f"reduction={reduction}")

def test_param_support():
    """测试不同参数类型的支持度"""
    print_header("1.c) 测试不同参数类型的支持度")
    
    shape = (3, 4)
    np_input = np.random.randn(*shape).astype(np.float32)
    np_target = np.random.randn(*shape).astype(np.float32)
    
    # 基本输入
    pt_input = torch.tensor(np_input)
    pt_target = torch.tensor(np_target)
    
    ms_input = Tensor(np_input, dtype=ms.float32)
    ms_target = Tensor(np_target, dtype=ms.float32)
    
    # 检查mint API是否可用
    ms_using_mint = True
    try:
        _ = mint_F.mse_loss
    except (AttributeError, RuntimeError):
        ms_using_mint = False
        print("mint.nn.functional.mse_loss 不可用，使用替代实现...")
    
    mse_loss_fn = mint_F.mse_loss if ms_using_mint else mindspore_mse_loss
    
    # 测试不同reduction参数
    reductions = ['none', 'sum', 'mean', 'INVALID']
    
    for reduction in reductions:
        print(f"\n测试reduction参数: '{reduction}'")
        
        try:
            pt_output = F.mse_loss(pt_input, pt_target, reduction=reduction)
            print(f"PyTorch: 支持 reduction='{reduction}'")
        except Exception as e:
            print(f"PyTorch 错误: {str(e)}")
        
        try:
            ms_output = mse_loss_fn(ms_input, ms_target, reduction=reduction)
            print(f"MindSpore ({'mint API' if ms_using_mint else '替代实现'}): 支持 reduction='{reduction}'")
        except Exception as e:
            print(f"MindSpore 错误: {str(e)}")
    
    # 使用随机输入进行多次测试
    num_tests = 3  # 测试3组不同的输入
    for i in range(num_tests):
        # 生成新的随机输入
        np_random_input = np.random.randn(*shape).astype(np.float32)
        np_random_target = np.random.randn(*shape).astype(np.float32)
        
        pt_random_input = torch.tensor(np_random_input)
        ms_random_input = Tensor(np_random_input, dtype=ms.float32)
        pt_random_target = torch.tensor(np_random_target)
        ms_random_target = Tensor(np_random_target, dtype=ms.float32)
        
        print(f"\n测试随机输入组 #{i+1}:")
        print(f"随机输入平均值: {np_random_input.mean():.4f}, 随机目标平均值: {np_random_target.mean():.4f}")
        
        for reduction in ['none', 'sum', 'mean']:
            try:
                pt_output = F.mse_loss(pt_random_input, pt_random_target, reduction=reduction)
                if reduction != 'none':
                    print(f"PyTorch {reduction}输出: {pt_output.item():.6f}")
                else:
                    print(f"PyTorch {reduction}输出形状: {pt_output.shape}")
                pt_ok = True
            except Exception as e:
                print(f"PyTorch 错误: {str(e)}")
                pt_ok = False
            
            try:
                ms_output = mse_loss_fn(ms_random_input, ms_random_target, reduction=reduction)
                if reduction != 'none':
                    print(f"MindSpore {reduction}输出: {ms_output.asnumpy().item():.6f}")
                else:
                    print(f"MindSpore {reduction}输出形状: {ms_output.shape}")
                ms_ok = True
            except Exception as e:
                print(f"MindSpore 错误: {str(e)}")
                ms_ok = False
            
            if pt_ok and ms_ok:
                compare_outputs(pt_output, ms_output, f"随机输入 #{i+1}, reduction={reduction}")

def test_error_handling():
    """测试错误处理的准确性"""
    print_header("1.d) 测试错误处理的准确性")
    
    # 检查mint API是否可用
    ms_using_mint = True
    try:
        _ = mint_F.mse_loss
    except (AttributeError, RuntimeError):
        ms_using_mint = False
        print("mint.nn.functional.mse_loss 不可用，使用替代实现...")
    
    mse_loss_fn = mint_F.mse_loss if ms_using_mint else mindspore_mse_loss
    
    # 测试输入和目标形状不匹配
    print("\n测试输入和目标形状不匹配:")
    
    pt_input = torch.randn(2, 3)
    pt_target = torch.randn(3, 2)
    
    ms_input = Tensor(np.random.randn(2, 3), dtype=ms.float32)
    ms_target = Tensor(np.random.randn(3, 2), dtype=ms.float32)
    
    try:
        pt_output = F.mse_loss(pt_input, pt_target)
        print("PyTorch结果:", pt_output.item())
    except Exception as e:
        print(f"PyTorch错误: {str(e)}")
    
    try:
        ms_output = mse_loss_fn(ms_input, ms_target)
        print("MindSpore结果:", ms_output.asnumpy().item())
    except Exception as e:
        print(f"MindSpore错误: {str(e)}")
    
    # 测试错误的输入类型
    print("\n测试错误的输入类型:")
    
    # 字符串输入
    try:
        pt_output = F.mse_loss(pt_input, "wrong_target")
        print("PyTorch支持字符串输入")
    except Exception as e:
        print(f"PyTorch错误: {type(e).__name__}: {str(e)}")
    
    try:
        ms_output = mse_loss_fn(ms_input, "wrong_target")
        print("MindSpore支持字符串输入")
    except Exception as e:
        print(f"MindSpore错误: {type(e).__name__}: {str(e)}")
    
    # 测试各种奇怪输入
    print("\n测试各种奇怪输入:")
    
    # 测试None输入
    try:
        pt_output = F.mse_loss(None, pt_target)
        print("PyTorch支持None输入")
    except Exception as e:
        print(f"PyTorch错误: {type(e).__name__}: {str(e)}")
    
    try:
        ms_output = mse_loss_fn(None, ms_target)
        print("MindSpore支持None输入")
    except Exception as e:
        print(f"MindSpore错误: {type(e).__name__}: {str(e)}")
    
    # 测试0维张量
    try:
        pt_output = F.mse_loss(torch.tensor(5.0), torch.tensor(3.0))
        print(f"PyTorch支持0维张量, 输出: {pt_output.item()}")
    except Exception as e:
        print(f"PyTorch错误: {type(e).__name__}: {str(e)}")
    
    try:
        ms_output = mse_loss_fn(Tensor(5.0, ms.float32), Tensor(3.0, ms.float32))
        print(f"MindSpore支持0维张量, 输出: {ms_output.asnumpy().item()}")
    except Exception as e:
        print(f"MindSpore错误: {type(e).__name__}: {str(e)}")

def test_nn_implementation():
    """测试神经网络实现"""
    print_header("2.a/b) 测试神经网络实现和推理结果")
    
    # 检查mint API是否可用
    ms_using_mint = True
    try:
        _ = mint_F.mse_loss
    except (AttributeError, RuntimeError):
        ms_using_mint = False
        print("mint.nn.functional.mse_loss 不可用，使用替代实现...")
    
    mse_loss_fn = mint_F.mse_loss if ms_using_mint else mindspore_mse_loss
    
    # 简单的回归网络
    class PTRegressionNet(torch.nn.Module):
        def __init__(self, input_dim, output_dim):
            super(PTRegressionNet, self).__init__()
            self.linear = torch.nn.Linear(input_dim, output_dim)
            
        def forward(self, x, target=None):
            output = self.linear(x)
            if target is not None:
                loss = F.mse_loss(output, target)
                return loss
            return output
    
    class MSRegressionNet(ms.nn.Cell):
        def __init__(self, input_dim, output_dim, use_mint=True):
            super(MSRegressionNet, self).__init__()
            self.linear = ms.nn.Dense(input_dim, output_dim)
            self.use_mint = use_mint
            
        def construct(self, x, target=None):
            output = self.linear(x)
            if target is not None:
                if self.use_mint:
                    loss = mint_F.mse_loss(output, target)
                else:
                    loss = mindspore_mse_loss(output, target)
                return loss
            return output
    
    # 固定输入和权重
    input_dim = 5
    output_dim = 2
    batch_size = 3
    
    np_input = np.random.randn(batch_size, input_dim).astype(np.float32)
    np_target = np.random.randn(batch_size, output_dim).astype(np.float32)
    
    # 创建模型
    pt_model = PTRegressionNet(input_dim, output_dim)
    ms_model = MSRegressionNet(input_dim, output_dim, use_mint=ms_using_mint)
    
    # 固定权重 - 不需要转置
    np_weight = np.random.randn(output_dim, input_dim).astype(np.float32)
    np_bias = np.random.randn(output_dim).astype(np.float32)
    
    pt_model.linear.weight.data = torch.tensor(np_weight)
    pt_model.linear.bias.data = torch.tensor(np_bias)
    
    ms_model.linear.weight.set_data(Tensor(np_weight, dtype=ms.float32))
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
    
    # 检查mint API是否可用
    ms_using_mint = True
    try:
        _ = mint_F.mse_loss
    except (AttributeError, RuntimeError):
        ms_using_mint = False
        print("mint.nn.functional.mse_loss 不可用，使用替代实现...")
    
    mse_loss_fn = mint_F.mse_loss if ms_using_mint else mindspore_mse_loss
    
    # 函数的梯度测试
    shape = (3, 4)
    np_input = np.random.randn(*shape).astype(np.float32)
    np_target = np.random.randn(*shape).astype(np.float32)
    
    # PyTorch
    pt_input = torch.tensor(np_input, requires_grad=True)
    pt_target = torch.tensor(np_target)
    pt_output = F.mse_loss(pt_input, pt_target)
    pt_output.backward()
    pt_grad = pt_input.grad
    
    print("PyTorch梯度:")
    print(f"形状: {pt_grad.shape}")
    print(f"平均值: {pt_grad.mean().item()}")
    
    # MindSpore - 创建一个计算图用于计算梯度
    ms_input = ms.Parameter(Tensor(np_input, dtype=ms.float32))
    ms_target = Tensor(np_target, dtype=ms.float32)
    
    def forward_fn(x, target):
        return mse_loss_fn(x, target)
    
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
    output_dim = 2
    batch_size = 4
    
    class PTSimpleNet(torch.nn.Module):
        def __init__(self):
            super(PTSimpleNet, self).__init__()
            self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
            self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
            
        def forward(self, x, target=None):
            h = torch.nn.functional.relu(self.fc1(x))
            output = self.fc2(h)
            if target is not None:
                loss = F.mse_loss(output, target)
                return loss
            return output
    
    class MSSimpleNet(ms.nn.Cell):
        def __init__(self, use_mint=True):
            super(MSSimpleNet, self).__init__()
            self.fc1 = ms.nn.Dense(input_dim, hidden_dim)
            self.fc2 = ms.nn.Dense(hidden_dim, output_dim)
            self.relu = ms.nn.ReLU()
            self.use_mint = use_mint
            
        def construct(self, x, target=None):
            h = self.relu(self.fc1(x))
            output = self.fc2(h)
            if target is not None:
                if self.use_mint:
                    loss = mint_F.mse_loss(output, target)
                else:
                    loss = mindspore_mse_loss(output, target)
                return loss
            return output
    
    # 创建模型
    pt_net = PTSimpleNet()
    ms_net = MSSimpleNet(use_mint=ms_using_mint)
    
    # 固定权重 - 不需要转置
    np_fc1_weight = np.random.randn(hidden_dim, input_dim).astype(np.float32)
    np_fc1_bias = np.random.randn(hidden_dim).astype(np.float32)
    np_fc2_weight = np.random.randn(output_dim, hidden_dim).astype(np.float32)
    np_fc2_bias = np.random.randn(output_dim).astype(np.float32)
    
    pt_net.fc1.weight.data = torch.tensor(np_fc1_weight)
    pt_net.fc1.bias.data = torch.tensor(np_fc1_bias)
    pt_net.fc2.weight.data = torch.tensor(np_fc2_weight)
    pt_net.fc2.bias.data = torch.tensor(np_fc2_bias)
    
    ms_net.fc1.weight.set_data(Tensor(np_fc1_weight, dtype=ms.float32))
    ms_net.fc1.bias.set_data(Tensor(np_fc1_bias, dtype=ms.float32))
    ms_net.fc2.weight.set_data(Tensor(np_fc2_weight, dtype=ms.float32))
    ms_net.fc2.bias.set_data(Tensor(np_fc2_bias, dtype=ms.float32))
    
    # 准备输入和目标
    np_net_input = np.random.randn(batch_size, input_dim).astype(np.float32)
    np_net_target = np.random.randn(batch_size, output_dim).astype(np.float32)
    
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
        
    # 比较具体参数的梯度 (不需要转置)
    print("\n比较fc2.weight梯度:")
    ms_fc2_weight_grad = None
    for i, param in enumerate(ms_net.trainable_params()):
        if 'fc2.weight' in param.name:
            ms_fc2_weight_grad = ms_grads[i]
            break
    
    if ms_fc2_weight_grad is not None:
        ms_fc2_weight_grad_np = ms_fc2_weight_grad.asnumpy()
        max_diff = np.max(np.abs(ms_fc2_weight_grad_np - pt_fc2_weight_grad))
        print(f"fc2.weight梯度最大差异: {max_diff}")
        if max_diff < TOLERANCE:
            print(f"✓ fc2.weight梯度在容差范围内一致 (< {TOLERANCE})")
        else:
            print(f"✗ fc2.weight梯度超出容差范围 (> {TOLERANCE})")

if __name__ == "__main__":
    # 检查mint.nn.functional.mse_loss是否可用
    try:
        _ = mint_F.mse_loss
        print("mint.nn.functional.mse_loss 可用，使用官方API进行测试")
    except (AttributeError, RuntimeError) as e:
        print(f"mint.nn.functional.mse_loss 不可用: {e}")
        print("将使用自定义实现进行测试...")
    
    # 运行所有测试
    test_dtype_support()
    test_random_inputs()
    test_param_support()
    test_error_handling()
    test_nn_implementation()
    test_gradient()
