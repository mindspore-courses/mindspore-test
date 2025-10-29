import numpy as np
import pytest
import torch
import mindspore as ms
from mindspore import Tensor
import mindspore.mint as mint

class TestHardshrink:
    """Hardshrink算子测试类"""
    
    @pytest.mark.parametrize('dtype', [
        ms.float16,
        ms.float32,
        ms.bfloat16,
    ])
    def test_hardshrink_dtype_support(self, dtype):
        """测试不同数据类型的支持情况"""
        x = np.array([-2.0, -1.5, -0.5, 0.0, 0.5, 1.5, 2.0]).astype(np.float32)
        
        # MindSpore实现
        x_ms = Tensor(x, dtype=dtype)
        ms_hardshrink = mint.nn.Hardshrink(lambd=0.5)
        y_ms = ms_hardshrink(x_ms)
        
        # PyTorch实现
        x_torch = torch.tensor(x, dtype=torch.float32)
        torch_hardshrink = torch.nn.Hardshrink(lambd=0.5)
        y_torch = torch_hardshrink(x_torch)
        
        # 比较结果
        np.testing.assert_allclose(y_ms.asnumpy(), y_torch.detach().numpy(), rtol=1e-3, atol=1e-4)
        print(f"数据类型 {dtype} 测试通过")

    @pytest.mark.parametrize('shape', [
        (2, 3),           # 2D输入
        (2, 3, 4),        # 3D输入
        (2, 3, 4, 5),     # 4D输入
    ])
    def test_hardshrink_random_input(self, shape):
        """测试随机输入数据"""
        np.random.seed(42)
        x_np = np.random.uniform(-2, 2, size=shape).astype(np.float32)
        
        # MindSpore实现
        x_ms = Tensor(x_np, dtype=ms.float32)
        ms_hardshrink = mint.nn.Hardshrink(lambd=0.5)
        y_ms = ms_hardshrink(x_ms)
        
        # PyTorch实现
        x_torch = torch.tensor(x_np, dtype=torch.float32)
        torch_hardshrink = torch.nn.Hardshrink(lambd=0.5)
        y_torch = torch_hardshrink(x_torch)
        
        # 比较结果
        np.testing.assert_allclose(y_ms.asnumpy(), y_torch.detach().numpy(), rtol=1e-3, atol=1e-4)
        print(f"形状 {shape} 测试通过")

    @pytest.mark.parametrize('lambd', [0.0, 0.3, 0.5, 0.7, 1.0])
    def test_hardshrink_lambda(self, lambd):
        """测试不同lambda值"""
        x_np = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]).astype(np.float32)
        
        # MindSpore实现
        x_ms = Tensor(x_np, dtype=ms.float32)
        ms_hardshrink = mint.nn.Hardshrink(lambd=lambd)
        y_ms = ms_hardshrink(x_ms)
        
        # PyTorch实现
        x_torch = torch.tensor(x_np, dtype=torch.float32)
        torch_hardshrink = torch.nn.Hardshrink(lambd=lambd)
        y_torch = torch_hardshrink(x_torch)
        
        # 比较结果
        np.testing.assert_allclose(y_ms.asnumpy(), y_torch.detach().numpy(), rtol=1e-3, atol=1e-4)
        print(f"lambda值 {lambd} 测试通过")

    def test_hardshrink_grad(self):
        """测试Hardshrink的梯度计算"""
        np.random.seed(42)
        x_np = np.random.uniform(-2, 2, size=(2, 3)).astype(np.float32)
        
        # MindSpore实现
        x_ms = Tensor(x_np, dtype=ms.float32)
        ms_hardshrink = mint.nn.Hardshrink(lambd=0.5)
        
        # 定义前向和梯度函数
        def forward_fn(x):
            return ms_hardshrink(x)
        
        grad_fn = ms.grad(forward_fn)
        
        # PyTorch实现
        x_torch = torch.tensor(x_np, requires_grad=True)
        torch_hardshrink = torch.nn.Hardshrink(lambd=0.5)
        
        # 前向传播
        y_torch = torch_hardshrink(x_torch)
        
        # 反向传播
        y_torch.sum().backward()
        
        # 计算MindSpore梯度
        grad_ms = grad_fn(x_ms)
        
        # 比较梯度
        np.testing.assert_allclose(grad_ms.asnumpy(), x_torch.grad.numpy(), rtol=1e-3, atol=1e-4)
        print("梯度计算测试通过")

    def test_hardshrink_edge_cases(self):
        """测试边界情况"""
        # 测试边界值
        x_np = np.array([-0.5001, -0.5, -0.4999, 0.0, 0.4999, 0.5, 0.5001]).astype(np.float32)
        
        # MindSpore实现
        x_ms = Tensor(x_np, dtype=ms.float32)
        ms_hardshrink = mint.nn.Hardshrink(lambd=0.5)
        y_ms = ms_hardshrink(x_ms)
        
        # PyTorch实现
        x_torch = torch.tensor(x_np, dtype=torch.float32)
        torch_hardshrink = torch.nn.Hardshrink(lambd=0.5)
        y_torch = torch_hardshrink(x_torch)
        
        # 比较结果
        np.testing.assert_allclose(y_ms.asnumpy(), y_torch.detach().numpy(), rtol=1e-3, atol=1e-4)
        print("边界值测试通过")

    def test_hardshrink_invalid_input(self):
        """测试无效输入的处理"""
        ms_hardshrink = mint.nn.Hardshrink(lambd=0.5)
    
        # 测试不支持的数据类型
        with pytest.raises(TypeError, match="For primitive.*must be a type of.*Float"):
            x_ms = Tensor(np.array([1.0]), dtype=ms.int32)
            ms_hardshrink(x_ms)
            
        # 测试空输入
        with pytest.raises(ValueError, match="Input tensor cannot be empty"):
            x_ms = Tensor(np.array([]), dtype=ms.float32)
            ms_hardshrink(x_ms)
            
        # 测试维度为0的输入
        with pytest.raises(ValueError, match="Input tensor must have at least one dimension"):
            x_ms = Tensor(np.array([[]]), dtype=ms.float32)
            ms_hardshrink(x_ms)
        
        # 测试负的lambda值
        with pytest.raises(ValueError, match="Lambda value must be non-negative"):
            mint.nn.Hardshrink(lambd=-0.5)
            
        print("无效输入测试通过")

    def test_hardshrink_sequence(self):
        """测试序列数据的处理"""
        # 生成序列数据
        seq_length = 10
        batch_size = 2
        hidden_size = 16
        
        np.random.seed(42)
        x_np = np.random.uniform(-2, 2, size=(batch_size, seq_length, hidden_size)).astype(np.float16)
        
        # MindSpore实现
        x_ms = Tensor(x_np, dtype=ms.float16)
        ms_hardshrink = mint.nn.Hardshrink(lambd=0.5)
        y_ms = ms_hardshrink(x_ms)
        
        # PyTorch实现
        x_torch = torch.tensor(x_np, dtype=torch.float16)
        torch_hardshrink = torch.nn.Hardshrink(lambd=0.5)
        y_torch = torch_hardshrink(x_torch)
        
        # 比较结果
        np.testing.assert_allclose(
            y_ms.asnumpy(),
            y_torch.detach().numpy(),
            rtol=1e-3,
            atol=1e-3
        )
        print("序列数据测试通过")

    def test_hardshrink_in_network(self):
        """
        测试Hardshrink在神经网络中的使用
        1. 构建简单的前馈网络
        2. 测试正向推理结果
        3. 测试参数梯度
        注意：在Ascend平台上使用float16数据类型
        """
        # 构建MindSpore网络
        class MSNet(ms.nn.Cell):
            def __init__(self):
                super().__init__()
                self.linear = ms.nn.Dense(4, 2)
                self.hardshrink = mint.nn.Hardshrink(lambd=0.5)

            def construct(self, x):
                x = self.linear(x)
                return self.hardshrink(x)

        # 构建PyTorch网络
        class PTNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 2)
                self.hardshrink = torch.nn.Hardshrink(lambd=0.5)

            def forward(self, x):
                x = self.linear(x)
                return self.hardshrink(x)

        # 初始化网络
        ms_net = MSNet()
        # 将整个网络转换为float16
        ms_net = ms_net.to_float(ms.float16)
        pt_net = PTNet()

        # 固定权重 - 使用float16
        ms_weight = Tensor(np.array([[0.1, 0.2, 0.3, 0.4],
                                    [0.5, 0.6, 0.7, 0.8]]), ms.float16)
        ms_bias = Tensor(np.array([0.1, 0.2]), ms.float16)
        
        pt_weight = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                                 [0.5, 0.6, 0.7, 0.8]], dtype=torch.float16)
        pt_bias = torch.tensor([0.1, 0.2], dtype=torch.float16)

        ms_net.linear.weight.set_data(ms_weight)
        ms_net.linear.bias.set_data(ms_bias)
        pt_net.linear.weight.data = pt_weight
        pt_net.linear.bias.data = pt_bias

        # 固定输入 - 使用float16
        input_data = np.array([[1.0, -2.0, 3.0, -4.0]], dtype=np.float16)
        ms_input = Tensor(input_data, ms.float16)
        pt_input = torch.tensor(input_data, dtype=torch.float16)

        # 测试正向推理
        ms_output = ms_net(ms_input)
        pt_output = pt_net(pt_input)

        # 比较结果时，将MindSpore的float16结果转换为float32进行比较
        np.testing.assert_allclose(
            ms_output.asnumpy().astype(np.float32),
            pt_output.detach().numpy(),
            rtol=1e-3,
            atol=1e-3,
            err_msg="正向推理结果误差过大"
        )

        # 测试梯度
        ms_grad_fn = ms.grad(ms_net, grad_position=None, weights=ms_net.trainable_params())
        pt_output.sum().backward()

        # 测试linear.weight的梯度
        ms_weight_grad = ms_grad_fn(ms_input)[0]
        pt_weight_grad = pt_net.linear.weight.grad

        np.testing.assert_allclose(
            ms_weight_grad.asnumpy().astype(np.float32),
            pt_weight_grad.numpy(),
            rtol=1e-3,
            atol=1e-3,
            err_msg="权重梯度误差过大"
        )

        # 测试linear.bias的梯度
        ms_bias_grad = ms_grad_fn(ms_input)[1]
        pt_bias_grad = pt_net.linear.bias.grad

        np.testing.assert_allclose(
            ms_bias_grad.asnumpy().astype(np.float32),
            pt_bias_grad.numpy(),
            rtol=1e-3,
            atol=1e-3,
            err_msg="偏置梯度误差过大"
        )
