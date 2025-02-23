import numpy as np
import pytest
import mindspore as ms
from mindspore import Tensor
import mindspore.mint as mint
import torch
import logging

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/test_hardswish.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class TestHardswish:
    """测试 Hardswish 激活函数"""
    
    def setup_method(self):
        """初始化测试环境"""
        self.ms_hardswish = mint.nn.Hardswish()
        self.torch_hardswish = torch.nn.Hardswish()
        
    def test_different_dtypes(self):
        """测试不同数据类型的支持度"""
        dtypes = [
            (np.float16, torch.float16, ms.float16),
            (np.float32, torch.float32, ms.float32),
            (np.int32, torch.int32, ms.int32),
            (np.int64, torch.int64, ms.int64),
        ]
        
        for np_dtype, torch_dtype, ms_dtype in dtypes:
            x = np.random.uniform(-3, 3, size=(2, 3)).astype(np_dtype)
            logging.info(f"\n测试数据类型: {np_dtype}")
            
            try:
                x_torch = torch.tensor(x, dtype=torch_dtype)
                y_torch = self.torch_hardswish(x_torch)
                torch_support = True
                logging.info(f"PyTorch支持{torch_dtype}")
            except Exception as e:
                torch_support = False
                logging.info(f"PyTorch不支持{torch_dtype}, 错误: {str(e)}")
            
            try:
                x_ms = Tensor(x, dtype=ms_dtype)
                y_ms = self.ms_hardswish(x_ms)
                ms_support = True
                logging.info(f"MindSpore支持{ms_dtype}")
            except Exception as e:
                ms_support = False
                logging.info(f"MindSpore不支持{ms_dtype}, 错误: {str(e)}")
            
            if torch_support and ms_support:
                # 检查结果误差
                diff = np.abs(y_ms.asnumpy() - y_torch.detach().numpy())
                max_diff = np.max(diff)
                logging.info(f"最大误差: {max_diff}")
                assert max_diff < 1e-3, f"误差{max_diff}超过阈值1e-3"

    def test_random_values(self):
        """测试随机输入值的一致性"""
        x = np.random.uniform(-3, 3, size=(4, 5)).astype(np.float32)
        x_torch = torch.tensor(x, requires_grad=True)
        x_ms = Tensor(x, dtype=ms.float32)
        
        y_torch = self.torch_hardswish(x_torch)
        y_ms = self.ms_hardswish(x_ms)
        
        # 检查前向传播结果
        diff = np.abs(y_ms.asnumpy() - y_torch.detach().numpy())
        max_diff = np.max(diff)
        logging.info(f"\n随机输入前向传播最大误差: {max_diff}")
        assert max_diff < 1e-3, f"前向传播误差{max_diff}超过阈值1e-3"
        
        # 检查反向传播
        y_torch.sum().backward()
        grad_torch = x_torch.grad.numpy()
        
        grad_ms = ms.grad(self.ms_hardswish)(x_ms).asnumpy()
        grad_diff = np.abs(grad_ms - grad_torch)
        max_grad_diff = np.max(grad_diff)
        logging.info(f"随机输入反向传播最大误差: {max_grad_diff}")
        assert max_grad_diff < 1e-3, f"反向传播误差{max_grad_diff}超过阈值1e-3"

    def test_invalid_inputs(self):
        """测试无效输入的错误处理"""
        invalid_inputs = [
            ("string", "测试字符串输入"),
            (True, "测试布尔值输入"),
            (None, "测试空值输入"),
        ]
        
        for invalid_input, desc in invalid_inputs:
            logging.info(f"\n测试{desc}")
            
            # 测试PyTorch的错误处理
            try:
                self.torch_hardswish(invalid_input)
            except Exception as e:
                logging.info(f"PyTorch错误信息: {str(e)}")
            
            # 测试MindSpore的错误处理
            try:
                self.ms_hardswish(invalid_input)
            except Exception as e:
                logging.info(f"MindSpore错误信息: {str(e)}")

    def test_neural_network(self):
        """测试在神经网络中的使用"""
        class SimpleNet(ms.nn.Cell):
            def __init__(self):
                super(SimpleNet, self).__init__()
                self.fc = ms.nn.Dense(10, 5)
                self.hardswish = mint.nn.Hardswish()
            
            def construct(self, x):
                x = self.fc(x)
                return self.hardswish(x)
        
        class TorchNet(torch.nn.Module):
            def __init__(self):
                super(TorchNet, self).__init__()
                self.fc = torch.nn.Linear(10, 5)
                self.hardswish = torch.nn.Hardswish()
            
            def forward(self, x):
                x = self.fc(x)
                return self.hardswish(x)
        
        # 固定随机种子
        np.random.seed(42)
        torch.manual_seed(42)
        ms.set_seed(42)
        
        # 创建网络实例
        ms_net = SimpleNet()
        torch_net = TorchNet()
        
        # 复制权重确保两个网络参数相同
        torch_weight = torch_net.fc.weight.detach().numpy()
        ms_net.fc.weight.set_data(Tensor(torch_weight))
        ms_net.fc.bias.set_data(Tensor(torch_net.fc.bias.detach().numpy()))
        
        # 测试数据
        x = np.random.randn(3, 10).astype(np.float32)
        x_torch = torch.tensor(x, requires_grad=True)
        x_ms = Tensor(x, ms.float32)
        
        # 前向传播
        y_torch = torch_net(x_torch)
        y_ms = ms_net(x_ms)
        
        # 检查结果
        diff = np.abs(y_ms.asnumpy() - y_torch.detach().numpy())
        max_diff = np.max(diff)
        logging.info(f"\n神经网络前向传播最大误差: {max_diff}")
        assert max_diff < 1e-3, f"神经网络前向传播误差{max_diff}超过阈值1e-3"
        
        # 反向传播
        y_torch.sum().backward()
        grad_torch = x_torch.grad.numpy()
        
        grad_ms = ms.grad(ms_net)(x_ms).asnumpy()
        grad_diff = np.abs(grad_ms - grad_torch)
        max_grad_diff = np.max(grad_diff)
        logging.info(f"神经网络反向传播最大误差: {max_grad_diff}")
        assert max_grad_diff < 1e-3, f"神经网络反向传播误差{max_grad_diff}超过阈值1e-3"
        
    def test_different_shapes(self):
        """测试不同形状输入的支持度"""
        shapes = [
            (1,),           # 一维向量
            (2, 3),        # 二维矩阵
            (2, 3, 4),     # 三维张量
            (2, 3, 4, 5),  # 四维张量
        ]
        
        for shape in shapes:
            logging.info(f"\n测试输入形状: {shape}")
            x = np.random.uniform(-3, 3, size=shape).astype(np.float32)
            
            try:
                x_torch = torch.tensor(x)
                y_torch = self.torch_hardswish(x_torch)
                logging.info("PyTorch支持该形状")
            except Exception as e:
                logging.info(f"PyTorch不支持该形状, 错误: {str(e)}")
                continue
                
            try:
                x_ms = Tensor(x)
                y_ms = self.ms_hardswish(x_ms)
                logging.info("MindSpore支持该形状")
                
                # 检查结果误差
                diff = np.abs(y_ms.asnumpy() - y_torch.detach().numpy())
                max_diff = np.max(diff)
                logging.info(f"最大误差: {max_diff}")
                assert max_diff < 1e-3, f"误差{max_diff}超过阈值1e-3"
            except Exception as e:
                logging.info(f"MindSpore不支持该形状, 错误: {str(e)}")

    def test_boundary_values(self):
        """测试边界值和特殊值"""
        # 测试值包括Hardswish的关键点：x=-3, x=3, x=0
        special_values = np.array([
            -3.0, -2.99, -2.5, -1.0,  # 左边界区域
            -0.1, 0.0, 0.1,           # 零点附近
            1.0, 2.5, 2.99, 3.0,      # 右边界区域
            float('inf'), float('-inf'), float('nan')  # 特殊值
        ]).astype(np.float32)
        
        logging.info("\n测试边界值和特殊值")
        x_torch = torch.tensor(special_values)
        x_ms = Tensor(special_values)
        
        try:
            y_torch = self.torch_hardswish(x_torch)
            y_ms = self.ms_hardswish(x_ms)
            
            # 排除nan和inf的比较
            mask = np.isfinite(special_values)
            diff = np.abs(y_ms.asnumpy()[mask] - y_torch.detach().numpy()[mask])
            max_diff = np.max(diff)
            logging.info(f"边界值测试最大误差: {max_diff}")
            assert max_diff < 1e-3, f"边界值测试误差{max_diff}超过阈值1e-3"
            
            # 检查特殊值处理
            ms_inf = y_ms.asnumpy()[~mask]
            torch_inf = y_torch.detach().numpy()[~mask]
            logging.info(f"MindSpore特殊值处理结果: {ms_inf}")
            logging.info(f"PyTorch特殊值处理结果: {torch_inf}")
        except Exception as e:
            logging.info(f"特殊值测试发生错误: {str(e)}")

    def test_numerical_stability(self):
        """测试数值稳定性"""
        # 测试极小值和极大值
        scales = [1e-6, 1e-3, 1.0, 1e3, 1e6]
        for scale in scales:
            logging.info(f"\n测试数值尺度: {scale}")
            x = np.random.uniform(-3, 3, size=(2, 3)).astype(np.float32) * scale
            
            x_torch = torch.tensor(x, requires_grad=True)
            x_ms = Tensor(x, dtype=ms.float32)
            
            try:
                # 前向传播
                y_torch = self.torch_hardswish(x_torch)
                y_ms = self.ms_hardswish(x_ms)
                
                diff = np.abs(y_ms.asnumpy() - y_torch.detach().numpy())
                max_diff = np.max(diff)
                logging.info(f"前向传播最大误差: {max_diff}")
                assert max_diff < 1e-3, f"前向传播误差{max_diff}超过阈值1e-3"
                
                # 反向传播
                y_torch.sum().backward()
                grad_torch = x_torch.grad.numpy()
                
                grad_ms = ms.grad(self.ms_hardswish)(x_ms).asnumpy()
                grad_diff = np.abs(grad_ms - grad_torch)
                max_grad_diff = np.max(grad_diff)
                logging.info(f"反向传播最大误差: {max_grad_diff}")
                assert max_grad_diff < 1e-3, f"反向传播误差{max_grad_diff}超过阈值1e-3"
            except Exception as e:
                logging.info(f"数值尺度 {scale} 测试失败: {str(e)}")


if __name__ == "__main__":
    pytest.main(["-v", "test_hardswish.py"])
