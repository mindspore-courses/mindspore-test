import numpy as np
import torch
import mindspore
import mindspore.ops as ops
from mindspore import Tensor

def compare_outputs(pytorch_output, mindspore_output, name=""):
    """比较 PyTorch 和 MindSpore 的输出结果"""
    print(f"\n=== {name} 对比结果 ===")
    
    # 转换为numpy进行比较
    if isinstance(pytorch_output, torch.Tensor):
        pytorch_output = pytorch_output.detach().cpu().numpy()
    if isinstance(mindspore_output, Tensor):
        mindspore_output = mindspore_output.asnumpy()
    
    print(f"PyTorch 输出形状: {pytorch_output.shape}")
    print(f"MindSpore 输出形状: {mindspore_output.shape}")
    
    max_diff = np.max(np.abs(pytorch_output - mindspore_output))
    print(f"最大差值: {max_diff}")
    
    if max_diff > 1e-5:
        print("警告：输出差异较大！")
        print(f"PyTorch 输出: \n{pytorch_output}")
        print(f"MindSpore 输出: \n{mindspore_output}")
    
    return max_diff

def test_dtype_support():
    """测试不同数据类型的支持情况"""
    print("\n=== 测试数据类型支持 ===")
    dtypes = [
        np.float16, np.float32, np.float64,
        np.int8, np.int16, np.int32, np.int64,
        np.uint8
    ]
    
    for dtype in dtypes:
        print(f"\n测试数据类型: {dtype}")
        
        # 生成测试数据
        if dtype in [np.float16, np.float32, np.float64]:
            data = np.random.randn(2, 3).astype(dtype)
        else:
            data = np.random.randint(0, 100, size=(2, 3), dtype=dtype)
            
        repeats = 2
        
        # PyTorch测试
        try:
            pytorch_input = torch.tensor(data)
            pytorch_output = torch.repeat_interleave(pytorch_input, repeats, dim=0)
            print("PyTorch: 支持")
        except Exception as e:
            print(f"PyTorch错误: {str(e)}")
            continue
            
        # MindSpore测试
        try:
            mindspore_input = Tensor(data)
            mindspore_output = ops.repeat_interleave(mindspore_input, repeats, axis=0)
            print("MindSpore: 支持")
        except Exception as e:
            print(f"MindSpore错误: {str(e)}")
            continue
            
        compare_outputs(pytorch_output, mindspore_output, f"数据类型={dtype}")

def test_shapes():
    """
    测试不同形状的支持
    注意：MindSpore使用axis参数而不是dim（详见ISSUES.md#repeat_interleave参数命名差异）
    """
    print("\n=== 测试不同形状 ===\n")
    shapes = [(5,), (3, 4), (2, 3, 4), (2, 3, 2, 2)]
    
    for shape in shapes:
        print(f"测试形状: {shape}")
        data = np.random.randn(*shape).astype(np.float32)
        repeats = 2
        
        # PyTorch测试
        pytorch_input = torch.tensor(data)
        pytorch_output = torch.repeat_interleave(pytorch_input, repeats, dim=0)
        
        # MindSpore测试
        mindspore_input = Tensor(data)
        mindspore_output = ops.repeat_interleave(mindspore_input, repeats, axis=0)
        
        compare_outputs(pytorch_output, mindspore_output, f"shape={shape}")
        print()

def test_param_variations():
    """
    测试不同参数组合
    已知问题：不支持直接使用numpy数组作为repeats参数（详见ISSUES.md#repeat_interleave参数类型限制）
    """
    print("\n=== 测试参数组合 ===\n")
    
    data = np.random.randn(2, 3, 4).astype(np.float32)
    test_cases = [
        {"repeats": 2, "axis": 0, "desc": "基础重复"},
        {"repeats": 3, "axis": 1, "desc": "不同轴向重复"},
        {"repeats": np.array([1, 2, 3]), "axis": 1, "desc": "numpy数组repeats"},
        {"repeats": Tensor(np.array([1, 2, 3])), "axis": 1, "desc": "Tensor类型repeats"},
    ]
    
    for case in test_cases:
        print(f"测试场景: {case['desc']}")
        
        try:
            # PyTorch测试
            pytorch_input = torch.tensor(data)
            if isinstance(case["repeats"], (Tensor, np.ndarray)):
                pytorch_repeats = torch.tensor(case["repeats"].asnumpy() if isinstance(case["repeats"], Tensor) 
                                            else case["repeats"])
            else:
                pytorch_repeats = case["repeats"]
            
            pytorch_output = torch.repeat_interleave(pytorch_input, 
                                                   repeats=pytorch_repeats, 
                                                   dim=case["axis"])
            print("PyTorch: 成功")
            
            # MindSpore测试
            mindspore_input = Tensor(data)
            mindspore_output = ops.repeat_interleave(mindspore_input, 
                                                   repeats=case["repeats"],
                                                   axis=case["axis"])
            print("MindSpore: 成功")
            
            compare_outputs(pytorch_output, mindspore_output, 
                          f"repeats={case['repeats']}, axis={case['axis']}")
        except Exception as e:
            print(f"错误信息: {str(e)}")
        print()

def test_edge_cases():
    """测试边界情况"""
    print("\n=== 测试边界情况 ===\n")
    
    # 测试一维数组
    print("测试一维数组:")
    data_1d = np.array([1., 2., 3., 4.], dtype=np.float32)
    
    pytorch_input = torch.tensor(data_1d)
    pytorch_output = torch.repeat_interleave(pytorch_input, repeats=2)
    
    mindspore_input = Tensor(data_1d)
    mindspore_output = ops.repeat_interleave(mindspore_input, repeats=2)
    
    compare_outputs(pytorch_output, mindspore_output, "一维数组")
    
    # 测试单元素数组
    print("\n测试单元素数组:")
    data_single = np.array([[1.]], dtype=np.float32)
    
    pytorch_input = torch.tensor(data_single)
    pytorch_output = torch.repeat_interleave(pytorch_input, repeats=2, dim=0)
    
    mindspore_input = Tensor(data_single)
    mindspore_output = ops.repeat_interleave(mindspore_input, repeats=2, axis=0)
    
    compare_outputs(pytorch_output, mindspore_output, "单元素数组")
    
    # 测试零数组
    print("\n测试零数组:")
    data_zero = np.zeros((2, 3), dtype=np.float32)
    
    pytorch_input = torch.tensor(data_zero)
    pytorch_output = torch.repeat_interleave(pytorch_input, repeats=2, dim=0)
    
    mindspore_input = Tensor(data_zero)
    mindspore_output = ops.repeat_interleave(mindspore_input, repeats=2, axis=0)
    
    compare_outputs(pytorch_output, mindspore_output, "零数组")
    
    # 测试错误输入
    print("\n测试错误输入:")
    try:
        # 测试负数repeats
        data = np.array([1., 2., 3.], dtype=np.float32)
        mindspore_input = Tensor(data)
        output = ops.repeat_interleave(mindspore_input, repeats=-1)
        print("错误：负数repeats应该报错")
    except Exception as e:
        print(f"预期的错误: {str(e)}")
    
    try:
        # 测试错误的axis值
        data = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
        mindspore_input = Tensor(data)
        output = ops.repeat_interleave(mindspore_input, repeats=2, axis=2)
        print("错误：无效的axis值应该报错")
    except Exception as e:
        print(f"预期的错误: {str(e)}")

def test_gradient():
    """测试梯度"""
    print("\n=== 测试梯度计算 ===\n")
    
    input_data = np.random.randn(2, 3).astype(np.float32)
    input_tensor = Tensor(input_data, mindspore.float32)
    
    def grad_fn(inputs):
        return ops.repeat_interleave(inputs, repeats=2, axis=0)
    
    grad_fn_value = mindspore.grad(grad_fn)
    try:
        gradient = grad_fn_value(input_tensor)
        print("梯度计算成功")
        print(f"梯度形状: {gradient.shape}")
        print(f"梯度值: {gradient}")
    except Exception as e:
        print(f"梯度计算错误: {str(e)}")

def test_attention_mechanism():
    """
    测试在注意力机制场景下的表现
    测试结果详见ISSUES.md#repeat_interleave在注意力机制中的应用
    """
    print("\n=== 测试注意力机制场景 ===")
    
    # 测试参数
    batch_size = 2
    seq_len = 4
    num_heads = 8
    kv_num_heads = 2
    head_dim = 16
    
    # 创建输入数据
    key = np.random.randn(batch_size * seq_len, kv_num_heads, head_dim).astype(np.float32)
    value = np.random.randn(batch_size * seq_len, kv_num_heads, head_dim).astype(np.float32)
    
    repeat_num = num_heads // kv_num_heads
    
    # PyTorch前向测试
    pytorch_key = torch.tensor(key, requires_grad=True)
    pytorch_value = torch.tensor(value, requires_grad=True)
    
    pytorch_key_repeated = torch.repeat_interleave(pytorch_key, repeat_num, dim=1)
    pytorch_value_repeated = torch.repeat_interleave(pytorch_value, repeat_num, dim=1)
    
    # MindSpore前向测试
    mindspore_key = Tensor(key, dtype=mindspore.float32)
    mindspore_value = Tensor(value, dtype=mindspore.float32)
    
    mindspore_key_repeated = ops.repeat_interleave(mindspore_key, repeat_num, axis=1)
    mindspore_value_repeated = ops.repeat_interleave(mindspore_value, repeat_num, axis=1)
    
    # 比较前向结果
    key_diff = compare_outputs(pytorch_key_repeated, mindspore_key_repeated, "Key扩展")
    value_diff = compare_outputs(pytorch_value_repeated, mindspore_value_repeated, "Value扩展")
    
    print(f"\n前向传播最大差异:")
    print(f"Key差异: {key_diff}")
    print(f"Value差异: {value_diff}")
    
    # PyTorch梯度测试
    loss = pytorch_key_repeated.sum() + pytorch_value_repeated.sum()
    loss.backward()
    
    # MindSpore梯度测试
    def grad_fn(key_input, value_input):
        key_out = ops.repeat_interleave(key_input, repeat_num, axis=1)
        value_out = ops.repeat_interleave(value_input, repeat_num, axis=1)
        return key_out.sum() + value_out.sum()
    
    grad_fn_value = mindspore.grad(grad_fn, grad_position=(0, 1))
    mindspore_grads = grad_fn_value(mindspore_key, mindspore_value)
    
    # 比较梯度
    key_grad_diff = compare_outputs(pytorch_key.grad, mindspore_grads[0], "Key梯度")
    value_grad_diff = compare_outputs(pytorch_value.grad, mindspore_grads[1], "Value梯度")
    
    print(f"\n反向传播最大差异:")
    print(f"Key梯度差异: {key_grad_diff}")
    print(f"Value梯度差异: {value_grad_diff}")

if __name__ == "__main__":
    test_dtype_support()
    test_shapes()
    test_param_variations()
    test_edge_cases()
    test_gradient()
    test_attention_mechanism()
