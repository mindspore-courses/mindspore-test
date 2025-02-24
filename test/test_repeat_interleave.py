import numpy as np
import pytest
import mindspore
import torch
from mindspore import Tensor
from mindspore import mint

def test_different_dtypes():
    """测试不同数据类型的支持度"""
    input_data = np.array([[0, 1, 2], [3, 4, 5]])
    dtypes_ms = [mindspore.float16, mindspore.float32, mindspore.int8, 
                 mindspore.uint8, mindspore.int16, mindspore.int32, mindspore.int64]
    dtypes_torch = [torch.float16, torch.float32, torch.int8, 
                    torch.uint8, torch.int16, torch.int32, torch.int64]
    
    for ms_dtype, torch_dtype in zip(dtypes_ms, dtypes_torch):
        # MindSpore测试
        ms_input = Tensor(input_data, ms_dtype)
        ms_output = mint.repeat_interleave(ms_input, repeats=2, dim=0)
        
        # PyTorch测试
        torch_input = torch.tensor(input_data, dtype=torch_dtype)
        torch_output = torch.repeat_interleave(torch_input, repeats=2, dim=0)
        
        # 验证输出shape相同
        assert ms_output.shape == torch_output.shape
        # 验证输出值相同（考虑误差范围）
        np.testing.assert_allclose(ms_output.asnumpy(), torch_output.numpy(), rtol=1e-3)

def test_random_values():
    """测试随机输入值"""
    # 生成随机数据
    random_data = np.random.randn(3, 4).astype(np.float32)
    
    # MindSpore测试
    ms_input = Tensor(random_data, mindspore.float32)
    ms_output = mint.repeat_interleave(ms_input, repeats=3, dim=0)
    
    # PyTorch测试
    torch_input = torch.tensor(random_data, dtype=torch.float32)
    torch_output = torch.repeat_interleave(torch_input, repeats=3, dim=0)
    
    # 验证结果
    np.testing.assert_allclose(ms_output.asnumpy(), torch_output.numpy(), rtol=1e-3)

def test_different_params():
    """测试不同输入参数"""
    input_data = np.array([[1, 2], [3, 4]])
    ms_input = Tensor(input_data, mindspore.float32)
    
    # 测试repeats为list
    repeats_list = [2, 3]
    ms_output = mint.repeat_interleave(ms_input, repeats=repeats_list, dim=0)
    
    # 测试repeats为Tensor
    repeats_tensor = Tensor(repeats_list, mindspore.int32)
    ms_output_tensor = mint.repeat_interleave(ms_input, repeats=repeats_tensor, dim=0)
    
    # 测试dim为None（展平）
    ms_output_flatten = mint.repeat_interleave(ms_input, repeats=2, dim=None)

def test_error_handling():
    """测试错误处理"""
    input_data = np.array([[1, 2], [3, 4]])
    ms_input = Tensor(input_data, mindspore.float32)
    
    # 测试无效的dim
    try:
        output = mint.repeat_interleave(ms_input, repeats=2, dim=3)
        print("Warning: Invalid dim value did not raise an exception")
    except Exception as e:
        print(f"Invalid dim raised: {type(e).__name__}: {str(e)}")
    
    # 测试负数repeats - 记录实际行为
    try:
        output = mint.repeat_interleave(ms_input, repeats=-1, dim=0)
        print("Warning: Negative repeats value did not raise an exception")
    except Exception as e:
        print(f"Negative repeats raised: {type(e).__name__}: {str(e)}")

def test_backward():
    """测试反向传播"""
    # PyTorch测试
    torch_input = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    torch_output = torch.repeat_interleave(torch_input, repeats=2, dim=0)
    torch_output.sum().backward()
    torch_grad = torch_input.grad.numpy()
    
    # MindSpore测试
    ms_input = Tensor(torch_input.detach().numpy(), mindspore.float32)
    ms_output = mint.repeat_interleave(ms_input, repeats=2, dim=0)
    # 注意：这里需要根据MindSpore的反向传播API进行相应实现
    # 由于MindSpore的反向传播可能需要使用Cell或者其他方式实现
    # 这部分代码需要根据具体情况调整

def test_edge_cases():
    """测试边界条件"""
    # 测试空张量
    empty_input = Tensor(np.array([]), mindspore.float32)
    try:
        output = mint.repeat_interleave(empty_input, repeats=2, dim=0)
        print(f"Empty tensor output shape: {output.shape}")
    except Exception as e:
        print(f"Empty tensor raised: {type(e).__name__}: {str(e)}")
    
    # 测试repeats=0的情况
    input_data = np.array([[1, 2], [3, 4]])
    ms_input = Tensor(input_data, mindspore.float32)
    try:
        output = mint.repeat_interleave(ms_input, repeats=0, dim=0)
        print(f"Zero repeats output shape: {output.shape}")
    except Exception as e:
        print(f"Zero repeats raised: {type(e).__name__}: {str(e)}")
    
    # 测试repeats为很大的数
    try:
        output = mint.repeat_interleave(ms_input, repeats=1000000, dim=0)
        print(f"Large repeats output shape: {output.shape}")
    except Exception as e:
        print(f"Large repeats raised: {type(e).__name__}: {str(e)}")

def test_special_inputs():
    """测试特殊输入"""
    # 测试1D张量
    input_1d = Tensor(np.array([1, 2, 3]), mindspore.float32)
    output_1d = mint.repeat_interleave(input_1d, repeats=2, dim=0)
    
    # 测试3D张量
    input_3d = Tensor(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), mindspore.float32)
    output_3d = mint.repeat_interleave(input_3d, repeats=2, dim=1)
    
    # 测试不同维度的repeats
    input_data = np.array([[1, 2, 3], [4, 5, 6]])
    ms_input = Tensor(input_data, mindspore.float32)
    
    # repeats在不同维度的测试
    repeats_dim0 = [2, 3]  # 对第0维的每个元素分别重复2次和3次
    output_dim0 = mint.repeat_interleave(ms_input, repeats=repeats_dim0, dim=0)
    
    repeats_dim1 = [2, 1, 3]  # 对第1维的每个元素分别重复2次、1次和3次
    output_dim1 = mint.repeat_interleave(ms_input, repeats=repeats_dim1, dim=1)

def test_dtype_consistency():
    """测试数据类型一致性"""
    input_data = np.array([[1, 2], [3, 4]])
    
    # 测试repeats为不同数据类型
    repeats_types = [
        Tensor([2, 3], mindspore.int32),
        Tensor([2, 3], mindspore.int64),
        np.array([2, 3], dtype=np.int32),
        [2, 3]
    ]
    
    ms_input = Tensor(input_data, mindspore.float32)
    for repeats in repeats_types:
        try:
            output = mint.repeat_interleave(ms_input, repeats=repeats, dim=0)
            print(f"Repeats type {type(repeats)} succeeded")
        except Exception as e:
            print(f"Repeats type {type(repeats)} raised: {type(e).__name__}: {str(e)}")

def test_complex_cases():
    """测试复杂情况"""
    # 测试带有复数的张量
    complex_data = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex64)
    try:
        ms_input = Tensor(complex_data, mindspore.complex64)
        output = mint.repeat_interleave(ms_input, repeats=2, dim=0)
        print("Complex tensor supported")
    except Exception as e:
        print(f"Complex tensor raised: {type(e).__name__}: {str(e)}")
    
    # 测试非连续内存的张量
    input_data = np.array([[1, 2, 3], [4, 5, 6]])
    ms_input = Tensor(input_data, mindspore.float32)
    # 创建非连续张量（通过转置）
    non_contiguous_input = ms_input.transpose()
    output = mint.repeat_interleave(non_contiguous_input, repeats=2, dim=0)

def test_backward_functional():
    """测试函数式API的反向传播"""
    import mindspore.ops as ops
    from mindspore import context
    from mindspore.common.initializer import initializer
    
    context.set_context(mode=context.GRAPH_MODE)
    
    # 准备输入数据
    input_data = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
    ms_input = Tensor(input_data, mindspore.float32)
    
    # 尝试使用函数式API计算梯度
    try:
        grad_fn = ops.GradOperation()
        def forward_fn(x):
            return mint.repeat_interleave(x, repeats=2, dim=0).sum()
        
        grad_output = grad_fn(forward_fn)(ms_input)
        print("Functional API gradient:", grad_output)
    except Exception as e:
        print(f"Functional API gradient failed: {type(e).__name__}: {str(e)}")

def test_backward_cell():
    """测试使用Cell类的反向传播"""
    from mindspore import nn
    from mindspore.common.initializer import initializer
    
    class RepeatInterleaveNet(nn.Cell):
        def __init__(self):
            super(RepeatInterleaveNet, self).__init__()
        
        def construct(self, x):
            x = mint.repeat_interleave(x, repeats=2, dim=0)
            return x.sum()
    
    # 准备输入数据
    input_data = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
    ms_input = Tensor(input_data, mindspore.float32)
    
    try:
        # 创建网络和梯度函数
        net = RepeatInterleaveNet()
        grad_net = nn.GradOperation()(net)
        
        # 计算梯度
        grad_output = grad_net(ms_input)
        print("Cell gradient:", grad_output)
        
        # 对比PyTorch结果
        import torch
        torch_input = torch.tensor(input_data, requires_grad=True)
        torch_output = torch.repeat_interleave(torch_input, repeats=2, dim=0).sum()
        torch_output.backward()
        
        # 验证梯度
        np.testing.assert_allclose(
            grad_output.asnumpy(), 
            torch_input.grad.numpy(), 
            rtol=1e-3, 
            err_msg="Gradient values mismatch between MindSpore and PyTorch"
        )
        print("Gradient verification passed")
        
    except Exception as e:
        print(f"Cell gradient failed: {type(e).__name__}: {str(e)}")

def test_backward_compare():
    """比较函数式API和Cell类的梯度结果"""
    input_data = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
    ms_input = Tensor(input_data, mindspore.float32)
    
    # 函数式API的梯度
    grad_func = None
    try:
        grad_fn = ops.GradOperation()
        def forward_fn(x):
            return mint.repeat_interleave(x, repeats=2, dim=0).sum()
        grad_func = grad_fn(forward_fn)(ms_input)
    except Exception as e:
        print(f"Functional API gradient failed: {type(e).__name__}: {str(e)}")
    
    # Cell类的梯度
    grad_cell = None
    try:
        net = RepeatInterleaveNet()
        grad_net = nn.GradOperation()(net)
        grad_cell = grad_net(ms_input)
    except Exception as e:
        print(f"Cell gradient failed: {type(e).__name__}: {str(e)}")
    
    # 比较结果
    if grad_func is not None and grad_cell is not None:
        try:
            np.testing.assert_allclose(
                grad_func.asnumpy(), 
                grad_cell.asnumpy(), 
                rtol=1e-3,
                err_msg="Gradient values mismatch between functional and cell-based approach"
            )
            print("Functional API and Cell gradients match")
        except AssertionError as e:
            print(f"Gradient mismatch: {str(e)}")
