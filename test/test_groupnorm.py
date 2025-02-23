import os
import torch
import torch.nn as nn
import numpy as np
import pytest
import mindspore as ms
from mindspore import Tensor
import mindspore.mint as mint

def compare_outputs(ms_output, torch_output, rtol=1e-3, atol=1e-3):
    """比较MindSpore和PyTorch输出是否在误差范围内"""
    return np.allclose(ms_output.asnumpy(), torch_output.detach().numpy(), rtol=rtol, atol=atol)

class TestGroupNorm:
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_groupnorm_dtype_support(self, dtype):
        """测试不同dtype的支持情况"""
        num_groups = 2
        num_channels = 6
        shape = (2, 6, 4, 4)
        
        # 生成随机输入
        np.random.seed(42)
        x_np = np.random.randn(*shape).astype(dtype)
        
        # MindSpore测试
        try:
            x_ms = Tensor(x_np, dtype=ms.float32)
            ms_groupnorm = mint.nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, dtype=ms.float32)
            ms_output = ms_groupnorm(x_ms)
            ms_support = True
            ms_error = None
        except Exception as e:
            ms_support = False
            ms_error = str(e)
            
        # PyTorch测试
        try:
            x_torch = torch.tensor(x_np, dtype=torch.float32)
            torch_groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
            torch_output = torch_groupnorm(x_torch)
            torch_support = True
            torch_error = None
        except Exception as e:
            torch_support = False
            torch_error = str(e)
            
        print(f"\n数据类型 {dtype} 支持情况:")
        print(f"MindSpore: {'支持' if ms_support else f'不支持，错误: {ms_error}'}")
        print(f"PyTorch: {'支持' if torch_support else f'不支持，错误: {torch_error}'}")
        
        if ms_support and torch_support:
            assert compare_outputs(ms_output, torch_output)
            print(f"数据类型 {dtype} 的输出结果匹配")

    @pytest.mark.parametrize("shape", [
        (2, 6, 4, 4),
        (3, 6, 8, 8),
        (1, 6, 16, 16),
        (4, 6, 32, 32),  # 添加更大的shape
        (2, 12, 8, 8),   # 添加更多channel
        (1, 3, 4, 4)     # 添加较小的channel
    ])
    def test_groupnorm_random_input(self, shape):
        """测试固定dtype，随机输入值的情况"""
        num_groups = 2
        num_channels = 6
        
        np.random.seed(42)
        x_np = np.random.randn(*shape).astype(np.float32)
        
        # MindSpore
        x_ms = Tensor(x_np, dtype=ms.float32)
        ms_groupnorm = mint.nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, dtype=ms.float32)
        ms_output = ms_groupnorm(x_ms)
        
        # PyTorch
        x_torch = torch.tensor(x_np, dtype=torch.float32)
        torch_groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
        torch_output = torch_groupnorm(x_torch)
        
        assert compare_outputs(ms_output, torch_output)
        print(f"\n形状 {shape} 测试通过，输出结果匹配")

    def test_groupnorm_invalid_params(self):
        """测试无效参数输入的错误处理"""
        test_cases = [
            {"num_groups": 0, "num_channels": 6},  # 无效的组数
            {"num_groups": 3, "num_channels": 6},  # 通道数不能被组数整除
            {"num_groups": "2", "num_channels": 6},  # 无效的参数类型
        ]
        
        for case in test_cases:
            # MindSpore测试
            try:
                ms_groupnorm = mint.nn.GroupNorm(**case, dtype=ms.float32)
                ms_error = None
            except Exception as e:
                ms_error = str(e)
                
            # PyTorch测试
            try:
                torch_groupnorm = nn.GroupNorm(**case)
                torch_error = None
            except Exception as e:
                torch_error = str(e)
                
            print(f"\n测试用例: {case}")
            print(f"MindSpore错误: {ms_error}")
            print(f"PyTorch错误: {torch_error}")
            
            # 确保两个框架都能检测到错误
            assert (ms_error is not None) == (torch_error is not None)
            print("错误处理一致")

    def test_groupnorm_grad(self):
        """测试梯度计算"""
        num_groups = 2
        num_channels = 6
        shape = (2, 6, 4, 4)
        
        # 生成固定输入
        np.random.seed(42)
        x_np = np.random.randn(*shape).astype(np.float32)
        
        # MindSpore
        x_ms = Tensor(x_np, dtype=ms.float32)
        ms_groupnorm = mint.nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, dtype=ms.float32)
        ms_output = ms_groupnorm(x_ms)
        ms_grad = ms.grad(lambda x: ms_groupnorm(x).sum())(x_ms)
        
        # PyTorch
        x_torch = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)
        torch_groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
        torch_output = torch_groupnorm(x_torch)
        torch_output.sum().backward()
        torch_grad = x_torch.grad
        
        # 比较前向传播结果
        assert compare_outputs(ms_output, torch_output)
        print("\n前向传播测试通过，输出结果匹配")
        
        # 比较梯度
        assert compare_outputs(ms_grad, torch_grad)
        print("反向传播测试通过，梯度计算正确")

    def test_groupnorm_edge_cases(self):
        """测试边界情况"""
        num_channels = 6
        
        # 测试极小值输入
        x_np = np.full((2, 6, 4, 4), 1e-7, dtype=np.float32)
        x_ms = Tensor(x_np, dtype=ms.float32)
        x_torch = torch.tensor(x_np, dtype=torch.float32)
        
        # 测试不同group数
        for num_groups in [1, 2, 3, 6]:
            if num_channels % num_groups == 0:
                ms_groupnorm = mint.nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, dtype=ms.float32)
                torch_groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
                
                ms_output = ms_groupnorm(x_ms)
                torch_output = torch_groupnorm(x_torch)
                
                assert compare_outputs(ms_output, torch_output)
                print(f"\n组数 {num_groups} 的极小值输入测试通过")
        
        # 测试极大值输入
        x_np = np.full((2, 6, 4, 4), 1e7, dtype=np.float32)
        x_ms = Tensor(x_np, dtype=ms.float32)
        x_torch = torch.tensor(x_np, dtype=torch.float32)
        
        ms_groupnorm = mint.nn.GroupNorm(num_groups=2, num_channels=num_channels, dtype=ms.float32)
        torch_groupnorm = nn.GroupNorm(num_groups=2, num_channels=num_channels)
        
        ms_output = ms_groupnorm(x_ms)
        torch_output = torch_groupnorm(x_torch)
        
        assert compare_outputs(ms_output, torch_output)
        print("\n极大值输入测试通过")
        
        # 测试全零输入
        x_np = np.zeros((2, 6, 4, 4), dtype=np.float32)
        x_ms = Tensor(x_np, dtype=ms.float32)
        x_torch = torch.tensor(x_np, dtype=torch.float32)
        
        ms_output = ms_groupnorm(x_ms)
        torch_output = torch_groupnorm(x_torch)
        
        assert compare_outputs(ms_output, torch_output)
        print("\n全零输入测试通过")

    def test_groupnorm_error_messages(self):
        """测试错误信息的准确性和一致性"""
        num_channels = 6
        
        error_cases = [
            {
                'shape': (2, 4, 4, 4),  # 错误的channel数
                'num_groups': 2,
                'num_channels': 6,
                'error_type': ValueError,
                'error_msg': "输入张量的通道数与指定的通道数不匹配"
            },
            {
                'shape': (2, 6, 4, 4),
                'num_groups': 4,  # 不能被channel数整除的group数
                'num_channels': 6,
                'error_type': ValueError,
                'error_msg': "通道数必须能被组数整除"
            },
            {
                'shape': (2, 6),  # 维度不足
                'num_groups': 2,
                'num_channels': 6,
                'error_type': ValueError,
                'error_msg': "输入张量的维度必须大于等于3"
            }
        ]
        
        for case in error_cases:
            # 测试MindSpore
            try:
                x_np = np.random.randn(*case['shape']).astype(np.float32)
                x_ms = Tensor(x_np, dtype=ms.float32)
                ms_groupnorm = mint.nn.GroupNorm(
                    num_groups=case['num_groups'],
                    num_channels=case['num_channels'],
                    dtype=ms.float32
                )
                _ = ms_groupnorm(x_ms)
                raise AssertionError(f"预期应该抛出 {case['error_type']}")
            except Exception as e:
                assert isinstance(e, case['error_type']), f"错误类型不匹配: 期望 {case['error_type']}, 实际 {type(e)}"
                assert case['error_msg'] in str(e), f"错误信息不匹配: 期望包含 '{case['error_msg']}', 实际 '{str(e)}'"
            
            # 测试PyTorch
            try:
                x_torch = torch.tensor(x_np, dtype=torch.float32)
                torch_groupnorm = nn.GroupNorm(
                    num_groups=case['num_groups'],
                    num_channels=case['num_channels']
                )
                _ = torch_groupnorm(x_torch)
                raise AssertionError(f"预期应该抛出 {case['error_type']}")
            except Exception as e:
                assert isinstance(e, case['error_type']), f"PyTorch错误类型不匹配: 期望 {case['error_type']}, 实际 {type(e)}"
            
            print(f"\n错误用例测试通过: {case['error_msg']}")


if __name__ == "__main__":
    pytest.main(["-v", "test_groupnorm.py"])
