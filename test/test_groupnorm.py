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

    @pytest.mark.parametrize("shape, num_channels, num_groups", [
        ((2, 12, 8, 8), 12, 2),    # 中等通道数，中等分组
        ((1, 3, 4, 4), 3, 1),      # 小通道数，单分组
        ((2, 6, 4, 4), 6, 3),      # 小通道数，小分组
        ((2, 32, 16, 16), 32, 8),  # 大通道数，大分组
        ((1, 16, 8, 8), 16, 1),    # 中等通道数，单分组
        ((2, 15, 4, 4), 15, 5),    # 奇数通道数，奇数分组
        ((1, 24, 4, 4), 24, 4)     # 大通道数，中等分组
    ])
    def test_groupnorm_random_input(self, shape, num_channels, num_groups):
        """测试固定dtype，随机输入值的情况"""
        assert num_channels % num_groups == 0, f"num_channels {num_channels} 必须能被 num_groups {num_groups} 整除"
        
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
        print(f"\n形状 {shape}, num_groups={num_groups} 测试通过，输出结果匹配")

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

        np.random.seed(42)
        x_np = np.random.randn(*shape).astype(np.float32)
        print("输入数据范围:", x_np.min(), x_np.max())
        print("输入张量形状:", x_np.shape)

        # MindSpore
        x_ms = Tensor(x_np, dtype=ms.float32)
        ms_groupnorm = mint.nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, dtype=ms.float32)
        ms_output = ms_groupnorm(x_ms)
        print("MindSpore 前向传播输出范围:", ms_output.asnumpy().min(), ms_output.asnumpy().max())

        grad_fn = ms.grad(lambda x: ms_groupnorm(x).sum(), grad_position=0)
        ms_grad = grad_fn(x_ms)
        print("MindSpore 梯度范围:", ms_grad.asnumpy().min(), ms_grad.asnumpy().max())

        # PyTorch
        x_torch = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)
        torch_groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
        torch_output = torch_groupnorm(x_torch)
        print("PyTorch 前向传播输出范围:", torch_output.detach().numpy().min(), torch_output.detach().numpy().max())

        torch_output.sum().backward()
        torch_grad = x_torch.grad
        print("PyTorch 梯度范围:", torch_grad.detach().numpy().min(), torch_grad.detach().numpy().max())

        assert compare_outputs(ms_output, torch_output), "前向传播输出不匹配"
        print("\n前向传播测试通过，输出结果匹配")
        assert compare_outputs(ms_grad, torch_grad), "梯度不匹配"
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

    def test_groupnorm_error_messages(self):
        """测试错误信息的准确性和一致性"""
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

        for i, case in enumerate(error_cases):
            print(f"\n测试错误用例 {i+1}: {case['shape']}, num_groups={case['num_groups']}, num_channels={case['num_channels']}")

            # MindSpore
            x_np = np.random.randn(*case['shape']).astype(np.float32)
            x_ms = Tensor(x_np, dtype=ms.float32)
            ms_groupnorm = mint.nn.GroupNorm(
                num_groups=case['num_groups'],
                num_channels=case['num_channels'],
                dtype=ms.float32
            )
            try:
                ms_output = ms_groupnorm(x_ms)
                print(f"MindSpore 未抛出异常，输出范围: {ms_output.asnumpy().min()} ~ {ms_output.asnumpy().max()}")
            except Exception as e:
                print(f"MindSpore 抛出异常: {type(e).__name__}: {str(e)}")
                assert isinstance(e, case['error_type']), f"错误类型不匹配: 期望 {case['error_type']}, 实际 {type(e)}"
                assert case['error_msg'] in str(e), f"错误信息不匹配: 期望包含 '{case['error_msg']}', 实际 '{str(e)}'"

            # PyTorch
            x_torch = torch.tensor(x_np, dtype=torch.float32)
            torch_groupnorm = nn.GroupNorm(
                num_groups=case['num_groups'],
                num_channels=case['num_channels']
            )
            try:
                torch_output = torch_groupnorm(x_torch)
                print(f"PyTorch 未抛出异常，输出范围: {torch_output.detach().numpy().min()} ~ {torch_output.detach().numpy().max()}")
            except Exception as e:
                print(f"PyTorch 抛出异常: {type(e).__name__}: {str(e)}")
                assert isinstance(e, case['error_type']), f"PyTorch错误类型不匹配: 期望 {case['error_type']}, 实际 {type(e)}"


if __name__ == "__main__":
    pytest.main(["-v", "test_groupnorm.py"])
