'''
dtype: 
    pytorch - float32, float64
    mindspore - float32, float64

mode:
    pytorch - bilinear, nearest, bicubic
    mindspore - bilinear, nearest

grid:
    grid参数最后一维不为2时，mindspore直到使用输出值时才爆错
    代码：
    ms_output = mint_F.grid_sample(ms_input, ms_wrong_grid_dim)
    print("MindSpore支持网格维度不是2")
    print(ms_output)

报错信息:
    PyTorch错误: ValueError: nn.functional.grid_sample(): expected padding_mode to be 'zeros', 'border', or 'reflection', but got: 'invalid_padding'
    MindSpore错误: ValueError: Failed to convert the value "invalid_padding" of input 'padding_mode' of 'GridSampler2D' to enum.

    PyTorch错误: ValueError: nn.functional.grid_sample(): expected mode to be 'bilinear', 'nearest' or 'bicubic', but got: 'invalid_mode'
    MindSpore错误: ValueError: Failed to convert the value "invalid_mode" of input 'interpolation_mode' of 'GridSampler2D' to enum.
    
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

# 定义 MindSpore 的 grid_sample 替代函数，以防原始函数不可用
def mindspore_grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=False):
    raise NotImplementedError("手动实现grid_sample非常复杂，本测试使用内置函数")

def test_dtype_support():
    """测试不同数据类型的支持度"""
    print_header("1.a) 测试不同数据类型(dtype)的支持度")
    
    batch_size = 2
    channels = 3
    height = 8
    width = 8
    
    # 创建网格大小
    grid_height = 6
    grid_width = 6

    # dtypes_pytorch = [torch.bfloat16, torch.float16, torch.float32, torch.float64]
    # dtypes_mindspore = [ms.bfloat16, ms.float16, ms.float32, ms.float64]
    # dtype_names = ["bfloat16", "float16", "float32", "float64"]
    dtypes_pytorch = [torch.float16, torch.float32, torch.float64]
    dtypes_mindspore = [ms.float16, ms.float32, ms.float64]
    dtype_names = ["float16", "float32", "float64"]
    
    for pt_dtype, ms_dtype, dtype_name in zip(dtypes_pytorch, dtypes_mindspore, dtype_names):
        print(f"\n测试数据类型: PyTorch {dtype_name}, MindSpore {dtype_name}")
        
        # 生成随机输入
        np_input = np.random.randn(batch_size, channels, height, width).astype(np.float32)
        
        # 创建采样网格：范围在[-1,1]之间的随机值
        np_grid = np.random.uniform(-1, 1, (batch_size, grid_height, grid_width, 2)).astype(np.float32)
        
        try:
            # PyTorch
            pt_input = torch.tensor(np_input, dtype=pt_dtype)
            pt_grid = torch.tensor(np_grid, dtype=pt_dtype)
            pt_output = F.grid_sample(pt_input, pt_grid, mode='bilinear', padding_mode='zeros', align_corners=False)
            print(f"PyTorch 输出: shape={pt_output.shape}")
            pt_support = "支持"
        except Exception as e:
            print(f"PyTorch 错误: {type(e).__name__}: {str(e)}")
            pt_support = "不支持"
            
        try:
            # 首先尝试 mint.nn.functional.grid_sample
            ms_input = Tensor(np_input, dtype=ms_dtype)
            ms_grid = Tensor(np_grid, dtype=ms_dtype)
            ms_using_mint = True
            
            try:
                ms_output = mint_F.grid_sample(ms_input, ms_grid, mode='bilinear', padding_mode='zeros', align_corners=False)
            except (AttributeError, RuntimeError) as e:
                print(f"mint.nn.functional.grid_sample 不可用: {type(e).__name__}: {str(e)}")
                print("使用替代实现...")
                ms_output = mindspore_grid_sample(ms_input, ms_grid, mode='bilinear', padding_mode='zeros', align_corners=False)
                ms_using_mint = False
            
            print(f"MindSpore 输出 ({'mint API' if ms_using_mint else '替代实现'}): shape={ms_output.shape}")
            ms_support = "支持"
        except Exception as e:
            print(f"MindSpore 错误: {type(e).__name__}: {str(e)}")
            ms_support = "不支持"
            
        print(f"PyTorch {dtype_name}: {pt_support}, MindSpore {dtype_name}: {ms_support}")

def test_random_inputs():
    """测试随机输入值的输出一致性"""
    print_header("1.b) 测试随机输入值的输出一致性")
    
    # 检查mint API是否可用
    ms_using_mint = True
    try:
        _ = mint_F.grid_sample
    except (AttributeError, RuntimeError):
        ms_using_mint = False
        print("mint.nn.functional.grid_sample 不可用，无法进行测试...")
        return  # 退出测试函数
    
    # 不同的输入尺寸
    input_shapes = [
        (2, 1, 8, 8),      # 单通道
        (2, 3, 16, 16),    # 三通道，较大尺寸
        (1, 3, 32, 32)     # 批次为1
    ]
    
    grid_shapes = [
        (2, 6, 6, 2),      # 较小采样网格
        (2, 10, 10, 2),    # 较大采样网格
        (1, 16, 16, 2)     # 批次为1
    ]
    
    for i, (input_shape, grid_shape) in enumerate(zip(input_shapes, grid_shapes)):
        print(f"\n测试输入尺寸 #{i+1}: 输入={input_shape}, 网格={grid_shape}")
        
        # 生成随机输入
        np_input = np.random.randn(*input_shape).astype(np.float32)
        np_grid = np.random.uniform(-1, 1, grid_shape).astype(np.float32)
        
        # PyTorch
        pt_input = torch.tensor(np_input)
        pt_grid = torch.tensor(np_grid)
        pt_output = F.grid_sample(pt_input, pt_grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        print(f"PyTorch 输出: shape={pt_output.shape}")
        
        # MindSpore
        ms_input = Tensor(np_input, dtype=ms.float32)
        ms_grid = Tensor(np_grid, dtype=ms.float32)
        ms_output = mint_F.grid_sample(ms_input, ms_grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        print(f"MindSpore 输出: shape={ms_output.shape}")
        
        compare_outputs(pt_output, ms_output, f"输入尺寸 #{i+1}")

def test_param_support():
    """测试不同参数的支持度"""
    print_header("1.c) 测试不同参数的支持度")
    
    # 检查mint API是否可用
    ms_using_mint = True
    try:
        _ = mint_F.grid_sample
    except (AttributeError, RuntimeError):
        ms_using_mint = False
        print("mint.nn.functional.grid_sample 不可用，无法进行测试...")
        return  # 退出测试函数
    
    # 固定输入形状
    batch_size = 2
    channels = 3
    height = 16
    width = 16
    grid_height = 10
    grid_width = 10
    
    np_input = np.random.randn(batch_size, channels, height, width).astype(np.float32)
    np_grid = np.random.uniform(-1, 1, (batch_size, grid_height, grid_width, 2)).astype(np.float32)
    
    pt_input = torch.tensor(np_input)
    pt_grid = torch.tensor(np_grid)
    
    ms_input = Tensor(np_input, dtype=ms.float32)
    ms_grid = Tensor(np_grid, dtype=ms.float32)
    
    # 测试不同的插值模式 (mode)
    modes = ['bilinear', 'nearest', 'bicubic']
    
    for mode in modes:
        print(f"\n测试mode='{mode}':")
        
        try:
            pt_output = F.grid_sample(pt_input, pt_grid, mode=mode, padding_mode='zeros', align_corners=False)
            print(f"PyTorch mode='{mode}': 支持")
            pt_mode_supported = True
        except Exception as e:
            print(f"PyTorch 错误: {str(e)}")
            pt_mode_supported = False
        
        try:
            ms_output = mint_F.grid_sample(ms_input, ms_grid, mode=mode, padding_mode='zeros', align_corners=False)
            print(f"MindSpore mode='{mode}': 支持")
            ms_mode_supported = True
        except Exception as e:
            print(f"MindSpore 错误: {str(e)}")
            ms_mode_supported = False
        
        if pt_mode_supported and ms_mode_supported:
            compare_outputs(pt_output, ms_output, f"mode='{mode}'")
    
    # 测试不同的填充模式 (padding_mode)
    padding_modes = ['zeros', 'border', 'reflection']
    
    for padding_mode in padding_modes:
        print(f"\n测试padding_mode='{padding_mode}':")
        
        try:
            pt_output = F.grid_sample(pt_input, pt_grid, mode='bilinear', padding_mode=padding_mode, align_corners=False)
            print(f"PyTorch padding_mode='{padding_mode}': 支持")
            pt_padding_mode_supported = True
        except Exception as e:
            print(f"PyTorch 错误: {str(e)}")
            pt_padding_mode_supported = False
        
        try:
            ms_output = mint_F.grid_sample(ms_input, ms_grid, mode='bilinear', padding_mode=padding_mode, align_corners=False)
            print(f"MindSpore padding_mode='{padding_mode}': 支持")
            ms_padding_mode_supported = True
        except Exception as e:
            print(f"MindSpore 错误: {str(e)}")
            ms_padding_mode_supported = False
        
        if pt_padding_mode_supported and ms_padding_mode_supported:
            compare_outputs(pt_output, ms_output, f"padding_mode='{padding_mode}'")
    
    # 测试align_corners参数
    for align_corners in [True, False]:
        print(f"\n测试align_corners={align_corners}:")
        
        try:
            pt_output = F.grid_sample(pt_input, pt_grid, mode='bilinear', padding_mode='zeros', align_corners=align_corners)
            print(f"PyTorch align_corners={align_corners}: 支持")
            pt_align_corners_supported = True
        except Exception as e:
            print(f"PyTorch 错误: {str(e)}")
            pt_align_corners_supported = False
        
        try:
            ms_output = mint_F.grid_sample(ms_input, ms_grid, mode='bilinear', padding_mode='zeros', align_corners=align_corners)
            print(f"MindSpore align_corners={align_corners}: 支持")
            ms_align_corners_supported = True
        except Exception as e:
            print(f"MindSpore 错误: {str(e)}")
            ms_align_corners_supported = False
        
        if pt_align_corners_supported and ms_align_corners_supported:
            compare_outputs(pt_output, ms_output, f"align_corners={align_corners}")
    
    # 测试各种参数组合
    param_combinations = [
        {'mode': 'bilinear', 'padding_mode': 'zeros', 'align_corners': False},
        {'mode': 'nearest', 'padding_mode': 'border', 'align_corners': True},
        {'mode': 'bicubic', 'padding_mode': 'reflection', 'align_corners': True}
    ]
    
    for i, params in enumerate(param_combinations):
        print(f"\n测试参数组合 #{i+1}: {params}")
        
        try:
            pt_output = F.grid_sample(pt_input, pt_grid, **params)
            print(f"PyTorch 参数组合 #{i+1}: 支持")
            pt_supported = True
        except Exception as e:
            print(f"PyTorch 错误: {str(e)}")
            pt_supported = False
        
        try:
            ms_output = mint_F.grid_sample(ms_input, ms_grid, **params)
            print(f"MindSpore 参数组合 #{i+1}: 支持")
            ms_supported = True
        except Exception as e:
            print(f"MindSpore 错误: {str(e)}")
            ms_supported = False
        
        if pt_supported and ms_supported:
            compare_outputs(pt_output, ms_output, f"参数组合 #{i+1}")

def test_error_handling():
    """测试错误处理的准确性"""
    print_header("1.d) 测试错误处理的准确性")
    
    # 检查mint API是否可用
    ms_using_mint = True
    try:
        _ = mint_F.grid_sample
    except (AttributeError, RuntimeError):
        ms_using_mint = False
        print("mint.nn.functional.grid_sample 不可用，无法进行测试...")
        return  # 退出测试函数
    
    # 创建基本有效输入
    batch_size = 2
    channels = 3
    height = 8
    width = 8
    grid_height = 6
    grid_width = 6
    
    np_input = np.random.randn(batch_size, channels, height, width).astype(np.float32)
    np_grid = np.random.uniform(-1, 1, (batch_size, grid_height, grid_width, 2)).astype(np.float32)
    
    pt_input = torch.tensor(np_input)
    pt_grid = torch.tensor(np_grid)
    
    ms_input = Tensor(np_input, dtype=ms.float32)
    ms_grid = Tensor(np_grid, dtype=ms.float32)
    
    # 测试批次大小不匹配
    print("\n测试批次大小不匹配:")
    
    np_wrong_grid = np.random.uniform(-1, 1, (batch_size+1, grid_height, grid_width, 2)).astype(np.float32)
    pt_wrong_grid = torch.tensor(np_wrong_grid)
    ms_wrong_grid = Tensor(np_wrong_grid, dtype=ms.float32)
    
    try:
        pt_output = F.grid_sample(pt_input, pt_wrong_grid)
        print(f"PyTorch支持批次大小不匹配: 输出shape={pt_output.shape}")
    except Exception as e:
        print(f"PyTorch错误: {type(e).__name__}: {str(e)}")
    
    try:
        ms_output = mint_F.grid_sample(ms_input, ms_wrong_grid)
        print(f"MindSpore支持批次大小不匹配: 输出shape={ms_output.shape}")
    except Exception as e:
        print(f"MindSpore错误: {type(e).__name__}: {str(e)}")
    
    # 测试网格最后一维不是2
    print("\n测试网格最后一维不是2:")
    
    np_wrong_grid_dim = np.random.uniform(-1, 1, (batch_size, grid_height, grid_width, 3)).astype(np.float32)
    pt_wrong_grid_dim = torch.tensor(np_wrong_grid_dim)
    ms_wrong_grid_dim = Tensor(np_wrong_grid_dim, dtype=ms.float32)
    
    try:
        pt_output = F.grid_sample(pt_input, pt_wrong_grid_dim)
        print("PyTorch支持网格维度不是2")
    except Exception as e:
        print(f"PyTorch错误: {type(e).__name__}: {str(e)}")
    
    try:
        ms_output = mint_F.grid_sample(ms_input, ms_wrong_grid_dim)
        print("MindSpore支持网格维度不是2")
        print(ms_output)
    except Exception as e:
        print(f"MindSpore错误: {type(e).__name__}: {str(e)}")
    
    # 测试无效的插值模式
    print("\n测试无效的插值模式:")
    
    try:
        pt_output = F.grid_sample(pt_input, pt_grid, mode='invalid_mode')
        print("PyTorch支持无效的插值模式")
    except Exception as e:
        print(f"PyTorch错误: {type(e).__name__}: {str(e)}")
    
    try:
        ms_output = mint_F.grid_sample(ms_input, ms_grid, mode='invalid_mode')
        print("MindSpore支持无效的插值模式")
    except Exception as e:
        print(f"MindSpore错误: {type(e).__name__}: {str(e)}")
    
    # 测试无效的填充模式
    print("\n测试无效的填充模式:")
    
    try:
        pt_output = F.grid_sample(pt_input, pt_grid, padding_mode='invalid_padding')
        print("PyTorch支持无效的填充模式")
    except Exception as e:
        print(f"PyTorch错误: {type(e).__name__}: {str(e)}")
    
    try:
        ms_output = mint_F.grid_sample(ms_input, ms_grid, padding_mode='invalid_padding')
        print("MindSpore支持无效的填充模式")
    except Exception as e:
        print(f"MindSpore错误: {type(e).__name__}: {str(e)}")

def test_specific_cases():
    """测试特定的边界情况"""
    print_header("1.e) 测试特定的边界情况")
    
    # 检查mint API是否可用
    ms_using_mint = True
    try:
        _ = mint_F.grid_sample
    except (AttributeError, RuntimeError):
        ms_using_mint = False
        print("mint.nn.functional.grid_sample 不可用，无法进行测试...")
        return  # 退出测试函数
    
    batch_size = 2
    channels = 3
    height = 8
    width = 8
    grid_height = 6
    grid_width = 6
    
    # 创建基本有效输入
    np_input = np.random.randn(batch_size, channels, height, width).astype(np.float32)
    
    # 测试极限值的网格
    print("\n测试网格坐标为极限值 (-1, -1) 和 (1, 1):")
    
    # 创建坐标为极限值的网格
    np_extreme_grid = np.zeros((batch_size, grid_height, grid_width, 2), dtype=np.float32)
    # 左上角为 (-1, -1)
    np_extreme_grid[0, 0, 0] = [-1, -1]
    # 右下角为 (1, 1)
    np_extreme_grid[0, -1, -1] = [1, 1]
    
    pt_input = torch.tensor(np_input)
    pt_extreme_grid = torch.tensor(np_extreme_grid)
    
    ms_input = Tensor(np_input, dtype=ms.float32)
    ms_extreme_grid = Tensor(np_extreme_grid, dtype=ms.float32)
    
    try:
        pt_output = F.grid_sample(pt_input, pt_extreme_grid)
        print(f"PyTorch 支持极限值网格, 输出shape={pt_output.shape}")
        pt_supported = True
    except Exception as e:
        print(f"PyTorch 错误: {str(e)}")
        pt_supported = False
    
    try:
        ms_output = mint_F.grid_sample(ms_input, ms_extreme_grid)
        print(f"MindSpore 支持极限值网格, 输出shape={ms_output.shape}")
        ms_supported = True
    except Exception as e:
        print(f"MindSpore 错误: {str(e)}")
        ms_supported = False
    
    if pt_supported and ms_supported:
        compare_outputs(pt_output, ms_output, "极限值网格")
    
    # 测试超出范围的网格坐标
    print("\n测试超出范围的网格坐标:")
    
    # 创建超出[-1,1]范围的网格
    np_out_of_range_grid = np.random.uniform(-2, 2, (batch_size, grid_height, grid_width, 2)).astype(np.float32)
    
    pt_out_of_range_grid = torch.tensor(np_out_of_range_grid)
    ms_out_of_range_grid = Tensor(np_out_of_range_grid, dtype=ms.float32)
    
    try:
        pt_output = F.grid_sample(pt_input, pt_out_of_range_grid)
        print(f"PyTorch 支持超出范围的网格坐标, 输出shape={pt_output.shape}")
        pt_supported = True
    except Exception as e:
        print(f"PyTorch 错误: {str(e)}")
        pt_supported = False
    
    try:
        ms_output = mint_F.grid_sample(ms_input, ms_out_of_range_grid)
        print(f"MindSpore 支持超出范围的网格坐标, 输出shape={ms_output.shape}")
        ms_supported = True
    except Exception as e:
        print(f"MindSpore 错误: {str(e)}")
        ms_supported = False
    
    if pt_supported and ms_supported:
        compare_outputs(pt_output, ms_output, "超出范围的网格坐标")
    
    # 测试网格中包含NaN值
    print("\n测试网格中包含NaN值:")
    
    # 创建包含NaN的网格
    np_nan_grid = np.random.uniform(-1, 1, (batch_size, grid_height, grid_width, 2)).astype(np.float32)
    np_nan_grid[0, 0, 0, 0] = np.nan  # 设置一个NaN值
    
    pt_nan_grid = torch.tensor(np_nan_grid)
    ms_nan_grid = Tensor(np_nan_grid, dtype=ms.float32)
    
    try:
        pt_output = F.grid_sample(pt_input, pt_nan_grid)
        print(f"PyTorch 支持网格中包含NaN值, 输出shape={pt_output.shape}")
        pt_supported = True
    except Exception as e:
        print(f"PyTorch 错误: {str(e)}")
        pt_supported = False
    
    try:
        ms_output = mint_F.grid_sample(ms_input, ms_nan_grid)
        print(f"MindSpore 支持网格中包含NaN值, 输出shape={ms_output.shape}")
        ms_supported = True
    except Exception as e:
        print(f"MindSpore 错误: {str(e)}")
        ms_supported = False
    
    if pt_supported and ms_supported:
        compare_outputs(pt_output, ms_output, "网格中包含NaN值")

def test_nn_implementation():
    """测试神经网络实现"""
    print_header("2.a/b) 测试神经网络实现和推理结果")
    
    # 检查mint API是否可用
    ms_using_mint = True
    try:
        _ = mint_F.grid_sample
    except (AttributeError, RuntimeError):
        ms_using_mint = False
        print("mint.nn.functional.grid_sample 不可用，无法进行测试...")
        return  # 退出测试函数
    
    # 定义使用grid_sample的简单网络
    class PTSpatialTransformerNet(torch.nn.Module):
        def __init__(self):
            super(PTSpatialTransformerNet, self).__init__()
            self.conv = torch.nn.Conv2d(3, 3, kernel_size=3, padding=1)
            
        def forward(self, x, grid):
            x = self.conv(x)
            return F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    
    class MSSpatialTransformerNet(ms.nn.Cell):
        def __init__(self):
            super(MSSpatialTransformerNet, self).__init__()
            self.conv = ms.nn.Conv2d(3, 3, kernel_size=3, padding=1, pad_mode='pad', has_bias=True)
            
        def construct(self, x, grid):
            x = self.conv(x)
            return mint_F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    
    # 创建模型
    pt_model = PTSpatialTransformerNet()
    ms_model = MSSpatialTransformerNet()
    
    # 固定权重
    np_weight = np.random.randn(3, 3, 3, 3).astype(np.float32) * 0.1
    np_bias = np.random.randn(3).astype(np.float32) * 0.1
    
    pt_model.conv.weight.data = torch.tensor(np_weight)
    pt_model.conv.bias.data = torch.tensor(np_bias)
    
    ms_model.conv.weight.set_data(Tensor(np_weight, dtype=ms.float32))
    ms_model.conv.bias.set_data(Tensor(np_bias, dtype=ms.float32))
    
    # 创建输入
    batch_size = 2
    height = 16
    width = 16
    grid_height = 12
    grid_width = 12
    
    np_input = np.random.randn(batch_size, 3, height, width).astype(np.float32)
    np_grid = np.random.uniform(-1, 1, (batch_size, grid_height, grid_width, 2)).astype(np.float32)
    
    pt_input = torch.tensor(np_input)
    pt_grid = torch.tensor(np_grid)
    
    ms_input = Tensor(np_input, dtype=ms.float32)
    ms_grid = Tensor(np_grid, dtype=ms.float32)
    
    # 前向传播
    pt_output = pt_model(pt_input, pt_grid)
    ms_output = ms_model(ms_input, ms_grid)
    
    print(f"PyTorch模型输出shape: {pt_output.shape}")
    print(f"MindSpore模型输出shape: {ms_output.shape}")
    
    compare_outputs(pt_output, ms_output, "模型输出")

def test_gradient():
    """测试反向传播和梯度计算"""
    print_header("2.c) 测试反向传播和梯度计算")
    
    # 检查mint API是否可用
    ms_using_mint = True
    try:
        _ = mint_F.grid_sample
    except (AttributeError, RuntimeError):
        ms_using_mint = False
        print("mint.nn.functional.grid_sample 不可用，无法进行测试...")
        return  # 退出测试函数
    
    # 输入数据
    batch_size = 2
    channels = 3
    height = 8
    width = 8
    grid_height = 6
    grid_width = 6
    
    np_input = np.random.randn(batch_size, channels, height, width).astype(np.float32)
    np_grid = np.random.uniform(-0.9, 0.9, (batch_size, grid_height, grid_width, 2)).astype(np.float32)
    
    # PyTorch计算输入的梯度
    pt_input = torch.tensor(np_input, requires_grad=True)
    pt_grid = torch.tensor(np_grid, requires_grad=True)
    pt_output = F.grid_sample(pt_input, pt_grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    
    # 设置输出梯度为全1
    pt_grad_output = torch.ones_like(pt_output)
    pt_output.backward(pt_grad_output)
    
    pt_input_grad = pt_input.grad
    pt_grid_grad = pt_grid.grad
    
    print("PyTorch梯度:")
    print(f"输入梯度shape: {pt_input_grad.shape}, 平均值: {pt_input_grad.mean().item()}")
    print(f"网格梯度shape: {pt_grid_grad.shape}, 平均值: {pt_grid_grad.mean().item()}")
    
    # MindSpore计算输入的梯度
    ms_input = ms.Parameter(Tensor(np_input, dtype=ms.float32))
    ms_grid = ms.Parameter(Tensor(np_grid, dtype=ms.float32))
    
    # 使用MindSpore计算输入图像的梯度
    def forward_fn_input(x, grid):
        return mint_F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    
    grad_fn_input = ms.grad(forward_fn_input, 0)  # 对输入参数求导
    ms_input_grad = grad_fn_input(ms_input, ms_grid)
    
    print("\nMindSpore输入梯度:")
    print(f"输入梯度shape: {ms_input_grad.shape}, 平均值: {ms_input_grad.asnumpy().mean()}")
    
    # 比较输入梯度
    compare_outputs(pt_input_grad, ms_input_grad, "输入梯度")
    
    # 使用MindSpore计算网格的梯度
    def forward_fn_grid(x, grid):
        return mint_F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    
    grad_fn_grid = ms.grad(forward_fn_grid, 1)  # 对网格参数求导
    ms_grid_grad = grad_fn_grid(ms_input, ms_grid)
    
    print("\nMindSpore网格梯度:")
    print(f"网格梯度shape: {ms_grid_grad.shape}, 平均值: {ms_grid_grad.asnumpy().mean()}")
    
    # 比较网格梯度
    compare_outputs(pt_grid_grad, ms_grid_grad, "网格梯度")
    
    # 测试网络模型中的梯度
    print("\n测试神经网络中的梯度:")
    
    class PTSpatialTransformerNet(torch.nn.Module):
        def __init__(self):
            super(PTSpatialTransformerNet, self).__init__()
            self.conv = torch.nn.Conv2d(3, 3, kernel_size=3, padding=1)
            
        def forward(self, x, grid):
            x = self.conv(x)
            return F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    
    class MSSpatialTransformerNet(ms.nn.Cell):
        def __init__(self):
            super(MSSpatialTransformerNet, self).__init__()
            self.conv = ms.nn.Conv2d(3, 3, kernel_size=3, padding=1, pad_mode='pad', has_bias=True)
            
        def construct(self, x, grid):
            x = self.conv(x)
            return mint_F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    
    # 创建模型
    pt_model = PTSpatialTransformerNet()
    ms_model = MSSpatialTransformerNet()
    
    # 固定权重
    np_weight = np.random.randn(3, 3, 3, 3).astype(np.float32) * 0.1
    np_bias = np.random.randn(3).astype(np.float32) * 0.1
    
    pt_model.conv.weight.data = torch.tensor(np_weight)
    pt_model.conv.bias.data = torch.tensor(np_bias)
    
    ms_model.conv.weight.set_data(Tensor(np_weight, dtype=ms.float32))
    ms_model.conv.bias.set_data(Tensor(np_bias, dtype=ms.float32))
    
    # 创建输入和目标
    batch_size = 2
    height = 16
    width = 16
    grid_height = 12
    grid_width = 12
    
    np_input = np.random.randn(batch_size, 3, height, width).astype(np.float32)
    np_grid = np.random.uniform(-1, 1, (batch_size, grid_height, grid_width, 2)).astype(np.float32)
    
    # 生成伪标签/目标
    np_target = np.random.randn(batch_size, 3, grid_height, grid_width).astype(np.float32)
    
    # PyTorch设置
    pt_input = torch.tensor(np_input, requires_grad=True)
    pt_grid = torch.tensor(np_grid, requires_grad=True)
    pt_target = torch.tensor(np_target)
    
    # 前向传播
    pt_output = pt_model(pt_input, pt_grid)
    pt_loss = torch.nn.functional.mse_loss(pt_output, pt_target)
    
    # 反向传播
    pt_loss.backward()
    
    # 获取权重梯度
    pt_conv_weight_grad = pt_model.conv.weight.grad
    pt_conv_bias_grad = pt_model.conv.bias.grad
    
    print("\nPyTorch网络梯度:")
    print(f"conv.weight梯度平均值: {pt_conv_weight_grad.mean().item()}")
    print(f"conv.bias梯度平均值: {pt_conv_bias_grad.mean().item()}")
    
    # MindSpore设置
    ms_input = Tensor(np_input, dtype=ms.float32)
    ms_grid = Tensor(np_grid, dtype=ms.float32)
    ms_target = Tensor(np_target, dtype=ms.float32)
    
    # 定义损失函数和前向计算
    def forward_fn(model, x, grid, target):
        output = model(x, grid)
        loss = ms.nn.MSELoss()(output, target)
        return loss
    
    # 计算梯度
    grad_fn = ms.value_and_grad(forward_fn, None, ms_model.trainable_params(), has_aux=False)
    loss, grads = grad_fn(ms_model, ms_input, ms_grid, ms_target)
    
    print("\nMindSpore网络梯度:")
    for i, param in enumerate(ms_model.trainable_params()):
        print(f"{param.name}梯度平均值: {grads[i].asnumpy().mean()}")
    
    # 比较卷积权重梯度
    ms_conv_weight_grad = None
    for i, param in enumerate(ms_model.trainable_params()):
        if 'conv.weight' in param.name:
            ms_conv_weight_grad = grads[i]
            break
    
    if ms_conv_weight_grad is not None:
        print("\n比较conv.weight梯度:")
        compare_outputs(pt_conv_weight_grad, ms_conv_weight_grad, "conv.weight梯度")

def test_real_world_usage():
    """测试实际应用场景"""
    print_header("2.d) 测试实际应用场景")
    
    # 检查mint API是否可用
    ms_using_mint = True
    try:
        _ = mint_F.grid_sample
    except (AttributeError, RuntimeError):
        ms_using_mint = False
        print("mint.nn.functional.grid_sample 不可用，无法进行测试...")
        return  # 退出测试函数
    
    # 实际应用场景 1: 仿射变换
    print("\n测试场景1: 仿射变换")
    
    batch_size = 1
    channels = 3
    height = 32
    width = 32
    
    # 创建输入图像
    np_input = np.random.randn(batch_size, channels, height, width).astype(np.float32)
    
    # 创建仿射变换矩阵 (旋转45度)
    angle = 45.0 * np.pi / 180.0
    cos_val = np.cos(angle).astype(np.float32)  # 确保是float32
    sin_val = np.sin(angle).astype(np.float32)  # 确保是float32
    
    # 旋转矩阵
    affine_matrix = np.array([
        [cos_val, -sin_val, 0],
        [sin_val, cos_val, 0]
    ], dtype=np.float32)
    
    # 生成采样网格
    grid_x, grid_y = np.meshgrid(
        np.linspace(-1, 1, width, dtype=np.float32),  # 指定dtype=np.float32
        np.linspace(-1, 1, height, dtype=np.float32)  # 指定dtype=np.float32
    )
    grid = np.stack([grid_x, grid_y], axis=2)
    grid = np.broadcast_to(grid, (batch_size, height, width, 2))
    
    # 应用仿射变换到网格
    grid_x = grid[:, :, :, 0].reshape(batch_size, -1)
    grid_y = grid[:, :, :, 1].reshape(batch_size, -1)
    ones = np.ones_like(grid_x)
    points = np.stack([grid_x, grid_y, ones], axis=2)
    
    # 应用变换
    transformed_points = np.matmul(points, affine_matrix.T)
    transformed_grid = transformed_points.reshape(batch_size, height, width, 2)
    
    # 确保变换网格是float32类型
    transformed_grid = transformed_grid.astype(np.float32)
    
    # PyTorch
    pt_input = torch.tensor(np_input, dtype=torch.float32)  # 明确指定dtype
    pt_grid = torch.tensor(transformed_grid, dtype=torch.float32)  # 明确指定dtype
    
    # MindSpore
    ms_input = Tensor(np_input, dtype=ms.float32)
    ms_grid = Tensor(transformed_grid, dtype=ms.float32)
    
    # 执行采样
    pt_output = F.grid_sample(pt_input, pt_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    ms_output = mint_F.grid_sample(ms_input, ms_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    
    print("仿射变换结果:")
    compare_outputs(pt_output, ms_output, "旋转45度")
    
    # 实际应用场景 2: 图像缩放
    print("\n测试场景2: 图像缩放")
    
    # 缩小一半
    scale = 0.5
    
    # 创建缩放网格
    grid_x, grid_y = np.meshgrid(
        np.linspace(-scale, scale, width, dtype=np.float32),  # 指定dtype=np.float32
        np.linspace(-scale, scale, height, dtype=np.float32)  # 指定dtype=np.float32
    )
    scaled_grid = np.stack([grid_x, grid_y], axis=2)
    scaled_grid = np.broadcast_to(scaled_grid, (batch_size, height, width, 2))
    scaled_grid = scaled_grid.astype(np.float32)  # 确保是float32
    
    # PyTorch
    pt_scaled_grid = torch.tensor(scaled_grid, dtype=torch.float32)  # 明确指定dtype
    
    # MindSpore
    ms_scaled_grid = Tensor(scaled_grid, dtype=ms.float32)
    
    # 执行采样
    pt_scaled_output = F.grid_sample(pt_input, pt_scaled_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    ms_scaled_output = mint_F.grid_sample(ms_input, ms_scaled_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    
    print("缩放图像结果:")
    compare_outputs(pt_scaled_output, ms_scaled_output, "缩小一半")

    
def test_bfloat16():
    batch_size = 2
    channels = 3
    height = 8
    width = 8
    grid_height = 6
    grid_width = 6
    np_input = np.random.randn(batch_size, channels, height, width).astype(np.float32)
    np_grid = np.random.uniform(-1, 1, (batch_size, grid_height, grid_width, 2)).astype(np.float32)
    ms_dtype=ms.bfloat16
    try:
        # 首先尝试 mint.nn.functional.grid_sample
        ms_input = Tensor(np_input, dtype=ms_dtype)
        ms_grid = Tensor(np_grid, dtype=ms_dtype)
        ms_output = mint_F.grid_sample(ms_input, ms_grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        print(f"MindSpore 输出 ({'mint API' if ms_using_mint else '替代实现'}): shape={ms_output.shape}")
        ms_support = "支持"
    except Exception as e:
        print(f"MindSpore 错误: {type(e).__name__}: {str(e)}")
        ms_support = "不支持"
    
    input_shape = (2, 1, 8, 8)      # 单通道

    grid_shape = (2, 6, 6, 2)
    
    print(f"\n测试输入尺寸: 输入={input_shape}, 网格={grid_shape}")

    # 生成随机输入
    np_input = np.random.randn(*input_shape).astype(np.float32)
    np_grid = np.random.uniform(-1, 1, grid_shape).astype(np.float32)

    # MindSpore
    ms_input = Tensor(np_input, dtype=ms.float32)
    ms_grid = Tensor(np_grid, dtype=ms.float32)
    ms_output = mint_F.grid_sample(ms_input, ms_grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    print(f"MindSpore 输出: shape={ms_output.shape}")

    mindspore_np = mindspore_out.asnumpy()
    
    
if __name__ == "__main__":

    mint_available = True

    
    if mint_available:
        # 运行所有测试
        test_dtype_support()
        test_random_inputs()
        test_param_support()
        test_error_handling()
        test_specific_cases()
        test_nn_implementation()
        test_gradient()
        test_real_world_usage()
        test_bfloat16()
    else:
        print("由于API不可用，测试已跳过")
