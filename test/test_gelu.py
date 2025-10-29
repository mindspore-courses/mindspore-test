import numpy as np
import pytest
import torch
import mindspore as ms
from mindspore import Tensor
import mindspore.mint as mint

class TestGELU:
    """GELU算子测试类"""
    
    @pytest.mark.parametrize('dtype', [ms.float32, ms.float64])
    def test_gelu_dtype_support(self, dtype):
        """测试不同数据类型的支持情况"""
        x = np.array([-1.0, 0.0, 1.0, 2.0, 3.0]).astype(np.float32)
        
        # MindSpore实现
        x_ms = Tensor(x, dtype=dtype)
        ms_gelu = mint.nn.GELU()
        y_ms = ms_gelu(x_ms)
        
        # PyTorch实现
        x_torch = torch.tensor(x, dtype=torch.float32)
        torch_gelu = torch.nn.GELU()
        y_torch = torch_gelu(x_torch)
        
        # 比较结果
        np.testing.assert_allclose(y_ms.asnumpy(), y_torch.detach().numpy(), rtol=1e-3)
        print(f"数据类型 {dtype} 测试通过")

    @pytest.mark.parametrize('shape', [
        (2, 3),           # 2D输入
        (2, 3, 4),        # 3D输入
        (2, 3, 4, 5),     # 4D输入
    ])
    def test_gelu_random_input(self, shape):
        """测试随机输入数据"""
        np.random.seed(42)
        x_np = np.random.randn(*shape).astype(np.float32)
        
        # MindSpore实现
        x_ms = Tensor(x_np, dtype=ms.float32)
        ms_gelu = mint.nn.GELU()
        y_ms = ms_gelu(x_ms)
        
        # PyTorch实现
        x_torch = torch.tensor(x_np, dtype=torch.float32)
        torch_gelu = torch.nn.GELU()
        y_torch = torch_gelu(x_torch)
        
        # 比较结果，增加atol容差
        np.testing.assert_allclose(y_ms.asnumpy(), y_torch.detach().numpy(), rtol=1e-3, atol=1e-4)
        print(f"形状 {shape} 测试通过")

    def test_gelu_grad(self):
        """测试GELU的梯度计算"""
        np.random.seed(42)
        x_np = np.random.randn(2, 3).astype(np.float32)
        
        # MindSpore实现
        x_ms = Tensor(x_np, dtype=ms.float32)
        ms_gelu = mint.nn.GELU()
        
        # 定义前向和梯度函数
        def forward_fn(x):
            return ms_gelu(x)
        
        grad_fn = ms.grad(forward_fn)
        
        # PyTorch实现
        x_torch = torch.tensor(x_np, requires_grad=True)
        torch_gelu = torch.nn.GELU()
        
        # 前向传播
        y_torch = torch_gelu(x_torch)
        
        # 反向传播
        y_torch.sum().backward()
        
        # 计算MindSpore梯度
        grad_ms = grad_fn(x_ms)
        
        # 比较梯度
        np.testing.assert_allclose(grad_ms.asnumpy(), x_torch.grad.numpy(), rtol=1e-3, atol=1e-4)
        print("梯度计算测试通过")

    def test_gelu_invalid_input(self):
        """测试无效输入的处理"""
        ms_gelu = mint.nn.GELU()
        
        # 测试空输入
        with pytest.raises(ValueError):
            x_ms = Tensor(np.array([]), dtype=ms.float32)
            ms_gelu(x_ms)
            
        # 测试维度为0的输入
        with pytest.raises(ValueError):
            x_ms = Tensor(np.array([[]]), dtype=ms.float32)
            ms_gelu(x_ms)
        
        print("无效输入测试通过")
