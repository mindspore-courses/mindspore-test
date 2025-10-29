import mindspore
import numpy as np
from mindspore import mint
import mindspore as ms
from mindspore import Tensor, mint, ops
import mindspore.nn as mnn

import numpy as np
import torch
import torch.nn as nn

# 设置随机种子
seed = 42
torch.manual_seed(seed)
mindspore.set_seed(seed)

# 设置上下文，指定设备为 Ascend
ms.set_context(device_target="Ascend", device_id=0) 

'''
1.对应Pytorch 的相应接口进行测试：
a) 测试random输入不同dtype，对比两个框架的支持度
'''
def test_cos_all_dtypes():
    mindspore_dtypes = [
        mindspore.int8, mindspore.int16, mindspore.int32, mindspore.int64,
        mindspore.uint8, mindspore.float16, mindspore.float32, mindspore.float64,
        mindspore.bfloat16, mindspore.complex64, mindspore.complex128
    ]
    
    torch_dtypes = [
        torch.int8, torch.int16, torch.int32, torch.int64,
        torch.uint8, torch.float16, torch.float32, torch.float64,
        torch.bfloat16, torch.complex64, torch.complex128
    ]
    
    ms_supported_dtypes = []
    ms_unsupported_dtypes = []
    torch_supported_dtypes = []
    torch_unsupported_dtypes = []


    # 遍历所有 dtype
    for ms_dtype, torch_dtype in zip(mindspore_dtypes, torch_dtypes):
        # 生成随机数据
        input_data = np.random.randn(5).astype(np.float32)  # 生成随机数据
        ms_input = mindspore.tensor(input_data, ms_dtype)
        torch_input = torch.tensor(input_data, dtype=torch_dtype)

        try:
            # 调用 cos 函数
            ms_output = mint.cos(ms_input)
            assert ms_output.shape == ms_input.shape
            ms_supported_dtypes.append(ms_dtype)
        except Exception as e:
            ms_unsupported_dtypes.append(ms_dtype)
        
        try:
            # 调用 cos 函数
            torch_output = torch.cos(torch_input)
            assert torch_output.shape == torch_input.shape
            torch_supported_dtypes.append(torch_dtype)
        except Exception as e:
            torch_unsupported_dtypes.append(torch_dtype)


    print(f"MindSpore supported dtypes: {ms_supported_dtypes}")
    print(f"MindSpore unsupported dtypes: {ms_unsupported_dtypes}")
    print(f"PyTorch supported dtypes: {torch_supported_dtypes}")
    print(f"PyTorch unsupported dtypes: {torch_unsupported_dtypes}")

'''
b) 测试固定dtype，random输入值，对比两个框架输出是否相等（误差范围为小于1e-3）
'''
def test_cos_fixed_dtype_random_input():
    # 固定dtype
    dtypes = [
        mindspore.int8, mindspore.int16, mindspore.int32, mindspore.int64,
        mindspore.uint8, mindspore.float16, mindspore.float32, mindspore.float64,
        mindspore.bfloat16, mindspore.complex64, mindspore.complex128
    ]
    
    torch_dtypes = [
        torch.int8, torch.int16, torch.int32, torch.int64,
        torch.uint8, torch.float16, torch.float32, torch.float64,
        torch.bfloat16, torch.complex64, torch.complex128
    ]
    for dtype, torch_dtype in zip(dtypes, torch_dtypes):
        for _ in range(100):
            # 随机生成输入值
            input_data = np.abs(np.random.randn(10)).astype(np.float32)
            ms_input = mindspore.tensor(input_data, dtype)
            torch_input = torch.tensor(input_data, dtype=torch_dtype)
            
            # 计算输出
            ms_output = mint.cos(ms_input)
            torch_output = torch.cos(torch_input)
            
            # 对比输出
            if dtype == mindspore.bfloat16:
                np.testing.assert_allclose(ms_output.astype(ms.float32).asnumpy(), torch_output.to(torch.float32).numpy(), rtol=1e-3)
            else:
                np.testing.assert_allclose(ms_output.asnumpy(), torch_output.numpy(), rtol=1e-3)
    
    print("MindSpore and PyTorch outputs are equal within tolerance.")

'''
c) 测试固定shape，固定输入值，不同输入参数（string/bool等类型），两个框架的支持度
'''
def test_cos_different_input_types():
    # 固定 shape 为 (3,)
    shape = (3,)
    # 定义不同的输入类型
    input_data = [1.1, 2.5, -1.5]  # 基础数据
    input_types = {
        "list": input_data,  # Python list
        "np.array": np.array(input_data, dtype=np.float32),  # NumPy array
        "mindspore.tensor": mindspore.tensor(input_data, mindspore.float32),  # MindSpore mindspore.tensor
        "torch.tensor": torch.tensor(input_data, dtype=torch.float32),  # PyTorch tensor
        "tuple": tuple(input_data),  # Python tuple
        "string": "1.1, 2.5, -1.5",  # String
    }
    
    # 初始化支持和不支持的列表
    ms_supported_types = []
    ms_unsupported_types = []
    torch_supported_types = []
    torch_unsupported_types = []
    
    # 遍历所有输入类型
    for input_name, input_value in input_types.items():
        try:
            # 测试 MindSpore
            ms_output = mint.cos(input_value)
            # shape 与预期一致
            assert ms_output.shape == shape
            ms_supported_types.append(input_name)
        except Exception as e:
            ms_unsupported_types.append(input_name)
        
        try:
            # 测试 PyTorch
            torch_output = torch.cos(input_value)
            # shape 与预期一致
            assert torch_output.shape == shape
            torch_supported_types.append(input_name)
        except Exception as e:
            torch_unsupported_types.append(input_name)
    
    # 打印支持和不支持的输入类型
    print(f"MindSpore supported input types: {ms_supported_types}")
    print(f"MindSpore unsupported input types: {ms_unsupported_types}")
    print(f"PyTorch supported input types: {torch_supported_types}")
    print(f"PyTorch unsupported input types: {torch_unsupported_types}")

'''
d) 测试随机混乱输入，报错信息的准确性
'''
def test_cos_random_chaotic_input():
    # 定义随机混乱输入
    chaotic_inputs = [
        np.array([1.1, 2.5, -1.5], dtype=np.float32),  # np.array 输入
        torch.tensor([1.1, 2.5, -1.5], dtype=torch.float32),  # PyTorch tensor 浮点数
        torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32),  # PyTorch tensor 整型
        mindspore.tensor(np.array([1.1, 2.5, -1.5]), dtype=mindspore.float32),  # MindSpore mindspore.tensor 浮点数
        mindspore.tensor(np.array([[1, 2, 3], [4, 5, 6]]), dtype=mindspore.int32),  # MindSpore mindspore.tensor 整型
        [1.1, 2.5, -1.5],  # Python list
        (1.1, 2.5, -1.5),  # Python tuple
    ]
    
    # 初始化支持和不支持的列表
    ms_supported_inputs = []
    # 收集报错信息
    ms_unsupported_inputs = []
    torch_supported_inputs = []
    # 收集报错信息
    torch_unsupported_inputs = []
    
    # 遍历所有混乱输入
    for input_data in chaotic_inputs:
        try:
            ms_output = mint.cos(input_data)
            assert ms_output.asnumpy() is not None
            ms_supported_inputs.append((str(input_data), str(type(input_data))))
        except Exception as e:
            ms_unsupported_inputs.append((str(input_data), str(type(input_data)), str(e)))
        
        try:
            torch_output = torch.cos(input_data)
            assert torch_output.numpy() is not None
            torch_supported_inputs.append((str(input_data), str(type(input_data))))
        except Exception as e:
            torch_unsupported_inputs.append((str(input_data), str(type(input_data)), str(e)))
    
    # 打印支持和不支持的输入类型及报错信息
    print("MindSpore supported inputs:")
    for input, input_type in ms_supported_inputs:
        print(f"  input: {input}\n  input type:  {input_type}\n")
    
    print("\nMindSpore unsupported inputs:")
    for input, input_type, error_msg in ms_unsupported_inputs:
        # print(f"  {input_type}: {error_msg}")
        print("*"*80)
        print(f"   input:  {input}")
        print(f"   input type:  {input_type}")
        print(f"   error message:  {error_msg}")
        print("*"*80)

    
    print("\nPyTorch supported inputs:")
    for input, input_type in torch_supported_inputs:
        # print(f"  {input_type}")
        print(f"  input: {input}\n  input type:  {input_type}\n")
    
    print("\nPyTorch unsupported inputs:")
    for input, input_type, error_msg in torch_unsupported_inputs:
        # print(f"  {input_type}: {error_msg}")
        print("*"*80)
        print(f"   input:  {input}")
        print(f"   input type:  {input_type}")
        print(f"   error message:  {error_msg}")
        print("*"*80)


'''
2. 测试使用接口构造函数/神经网络的准确性
a) Github搜索带有该接口的代码片段/神经网络
b) 使用Pytorch和MindSpore，固定输入和权重，测试正向推理结果（误差范围小于1e-3，若报错则记录报错信息）
c) 测试该神经网络/函数反向，如果是神经网络，则测试Parameter的梯度，如果是函数，则测试函数输入的梯度
'''
def test_forward_backward():
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.linear = nn.Linear(3, 3)  # 输入维度为3，输出维度也为3

        def forward(self, x):
            return torch.cos(self.linear(x))

    class SimpleModelMS(mnn.Cell):
        def __init__(self):
            super(SimpleModelMS, self).__init__()
            self.linear = mnn.Dense(3, 3)  # 输入维度为3，输出维度也为3

        def construct(self, x):
            return mint.cos(self.linear(x))

    # 创建固定输入和权重
    input_data_np = np.array([[1.5, -2.3, 3.6], [-1.2, 0.4, -0.9]], dtype=np.float32)
    weight_np = np.random.randn(3, 3).astype(np.float32)
    bias_np = np.random.randn(3).astype(np.float32)

    # PyTorch模型设置
    model_torch = SimpleModel()
    model_torch.linear.weight.data = torch.tensor(weight_np)  
    model_torch.linear.bias.data = torch.tensor(bias_np)
    input_torch = torch.tensor(input_data_np, requires_grad=True)

    # MindSpore模型设置
    model_ms = SimpleModelMS()
    model_ms.linear.weight.set_data(Tensor(weight_np))  # 不需要转置
    model_ms.linear.bias.set_data(Tensor(bias_np))
    input_ms = Tensor(input_data_np, dtype=ms.float32)

    # 正向传播
    output_torch = model_torch(input_torch)
    output_ms = model_ms(input_ms).asnumpy()

    # 比较正向传播结果
    diff = np.abs(output_torch.detach().numpy() - output_ms)
    if np.all(diff < 1e-3):
        print("Forward pass results match within tolerance.")
    else:
        print("Forward pass results do not match.")

    # 反向传播
    output_torch.sum().backward()
    grad_input_torch = input_torch.grad.numpy()

    # 反向传播（MindSpore）
    grad_fn = ops.GradOperation(get_all=True)
    grad_input_ms = grad_fn(model_ms)(input_ms)[0].asnumpy()

    # 比较反向传播结果（输入梯度）
    diff_grad = np.abs(grad_input_torch - grad_input_ms)
    if np.all(diff_grad < 1e-3):
        print("Backward pass gradients match within tolerance.")
    else:
        print("Backward pass gradients do not match.")


# 基础测试函数
# test_cos_all_dtypes()
# test_cos_fixed_dtype_random_input()
# test_cos_different_input_types()
# test_cos_random_chaotic_input()

# 神经网络内测试函数
test_forward_backward()