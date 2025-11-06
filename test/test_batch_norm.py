import torch
import pytest
import numpy as np
import mindspore as ms
from mindspore import Tensor, ops, mint
import mindspore.nn as mnn

# 设置随机种子
seed = 42
torch.manual_seed(seed)
ms.set_seed(seed)

# 设置上下文，指定设备为 Ascend
ms.set_context(device_target="Ascend", device_id=0)

'''
1. 对应Pytorch 的相应接口进行测试：
a) 测试random输入不同dtype，对比两个框架的支持度
'''
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_batch_norm_all_dtypes(mode):
    ms.set_context(mode=mode)
    # 定义所有 mindspore.dtype 类型
    ms_dtypes = [
        ms.int8, 
        ms.int16, 
        ms.int32, 
        ms.int64, 
        ms.uint8, 
        ms.float16, 
        ms.float32, 
        ms.float64,
        ms.bfloat16,
        ms.complex64,
        ms.complex128
    ]

    # 定义对应的 PyTorch dtype
    torch_dtypes = [
        torch.int8, 
        torch.int16, 
        torch.int32,
        torch.int64, 
        torch.uint8, 
        torch.float16,
        torch.float32, 
        torch.float64, 
        torch.bfloat16,
        torch.complex64,
        torch.complex128
    ]

    ms_supported_dtypes = []
    ms_unsupported_dtypes = []
    torch_supported_dtypes = []
    torch_unsupported_dtypes = []

    # 遍历所有 dtype
    for ms_dtype, torch_dtype in zip(ms_dtypes, torch_dtypes):
        # 生成随机数据
        input_data = np.random.randn(1, 3, 8, 8).astype(np.float32)  # 生成随机数据
        ms_input = ms.tensor(input_data, ms_dtype)
        torch_input = torch.tensor(input_data, dtype=torch_dtype)
        # 特征均值和方差
        ms_running_mean = ms.tensor(np.zeros(3), dtype=ms.float32)
        ms_running_var = ms.tensor(np.ones(3), dtype=ms.float32)
        torch_running_mean = torch.tensor(np.zeros(3), dtype=torch.float32)
        torch_running_var = torch.tensor(np.ones(3), dtype=torch.float32) 

        try:
            # 调用 batch_norm 函数
            ms_output = mint.nn.functional.batch_norm(ms_input, ms_running_mean, ms_running_var)
            ms_supported_dtypes.append(ms_dtype)
        except Exception as e:
            ms_unsupported_dtypes.append(ms_dtype)

        try:
            # 调用 batch_norm 函数
            torch_output = torch.nn.functional.batch_norm(torch_input, torch_running_mean, torch_running_var)
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
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_batch_norm_fixed_dtype_random_input(mode):
    ms.set_context(mode=mode)
    # 固定dtype为float16、float32
    dtypes = [ms.float16, ms.float32]
    torch_dtypes = [torch.float16, torch.float32]
    # 特征均值和方差
    ms_running_mean = ms.tensor(np.zeros(3), dtype=ms.float32)
    ms_running_var = ms.tensor(np.ones(3), dtype=ms.float32)
    torch_running_mean = torch.tensor(np.zeros(3), dtype=torch.float32)
    torch_running_var = torch.tensor(np.ones(3), dtype=torch.float32) 
    for dtype, torch_dtype in zip(dtypes, torch_dtypes):
        for _ in range(100):
            # 随机生成输入值
            input_data = np.random.randn(1, 3, 8, 8).astype(np.float32)
            ms_input = ms.tensor(input_data, dtype)
            torch_input = torch.tensor(input_data, dtype=torch_dtype)

            # 计算输出
            ms_output = mint.nn.functional.batch_norm(ms_input, ms_running_mean, ms_running_var)
            torch_output = torch.nn.functional.batch_norm(torch_input, torch_running_mean, torch_running_var)

            # 对比输出
            np.testing.assert_allclose(ms_output.asnumpy(), torch_output.detach().numpy(), rtol=1e-3)

    print("MindSpore and PyTorch outputs are equal within tolerance.")


'''
c) 测试固定shape，固定输入值，不同输入参数（string/bool等类型），两个框架的支持度
'''
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_batch_norm_different_input_types(mode):
    ms.set_context(mode=mode)
    # 特征均值和方差
    ms_running_mean = ms.tensor(np.zeros(3), dtype=ms.float32)
    ms_running_var = ms.tensor(np.ones(3), dtype=ms.float32)
    torch_running_mean = torch.tensor(np.zeros(3), dtype=torch.float32)
    torch_running_var = torch.tensor(np.ones(3), dtype=torch.float32) 
    # 定义不同的输入类型
    input_data = np.random.randn(1, 3, 8, 8).astype(np.float32)  # 基础数据
    input_types = {
        "list": input_data.tolist(),  # Python list
        "np.array": np.array(input_data, dtype=np.float32),  # NumPy array
        "mindspore.tensor": ms.tensor(input_data, ms.float32),  # MindSpore mindspore.tensor
        "torch.tensor": torch.tensor(input_data, dtype=torch.float32),  # PyTorch tensor
        "tuple": tuple(input_data),  # Python tuple
        "string": "random_input_data",  # String
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
            ms_output = mint.nn.functional.batch_norm(input_value, ms_running_mean, ms_running_var)
            ms_supported_types.append(input_name)
        except Exception as e:
            ms_unsupported_types.append(input_name)

        try:
            # 测试 PyTorch
            torch_output = torch.nn.functional.batch_norm(input_value, torch_running_mean, torch_running_var)
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
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_batch_norm_random_chaotic_input(mode):
    ms.set_context(mode=mode)
    # 定义随机混乱输入
    chaotic_inputs = [
        np.random.randn(1, 3, 8, 8).astype(np.float32),  # np.array 输入
        torch.randn(1, 3, 8, 8, dtype=torch.float32),  # PyTorch tensor 浮点数
        torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32),  # PyTorch tensor 整型
        ms.tensor(np.random.randn(1, 3, 8, 8).astype(np.float32), dtype=ms.float32),  # MindSpore tensor 浮点数
        ms.tensor(np.array([[1, 2, 3], [4, 5, 6]]), dtype=ms.int32),  # MindSpore tensor 整型
        [1.1, 2.5, -1.5],  # Python list
        (1.1, 2.5, -1.5),  # Python tuple
    ]
    # 特征均值和方差
    ms_running_mean = ms.tensor(np.zeros(3), dtype=ms.float32)
    ms_running_var = ms.tensor(np.ones(3), dtype=ms.float32)
    torch_running_mean = torch.tensor(np.zeros(3), dtype=torch.float32)
    torch_running_var = torch.tensor(np.ones(3), dtype=torch.float32) 
    # 初始化支持和不支持的列表
    ms_supported_inputs = []
    ms_unsupported_inputs = []
    torch_supported_inputs = []
    torch_unsupported_inputs = []

    # 遍历所有混乱输入
    for input_data in chaotic_inputs:
        try:
            ms_output = mint.nn.functional.batch_norm(input_data, ms_running_mean, ms_running_var)
            ms_supported_inputs.append((str(input_data), str(type(input_data))))
        except Exception as e:
            ms_unsupported_inputs.append((str(input_data), str(type(input_data)), str(e)))

        try:
            torch_output = torch.nn.functional.batch_norm(input_data, torch_running_mean, torch_running_var)
            torch_supported_inputs.append((str(input_data), str(type(input_data))))
        except Exception as e:
            torch_unsupported_inputs.append((str(input_data), str(type(input_data)), str(e)))


    # 打印支持和不支持的输入类型及报错信息
    print("MindSpore supported inputs:")
    for input, input_type in ms_supported_inputs:
        print(f"  input: {input}\n  input type:  {input_type}\n")

    print("\nMindSpore unsupported inputs:")
    for input, input_type, error_msg in ms_unsupported_inputs:
        print("*"*80)
        print(f"   input:  {input}")
        print(f"   input type:  {input_type}")
        print(f"   error message:  {error_msg}")
        print("*"*80)

    print("\nPyTorch supported inputs:")
    for input, input_type in torch_supported_inputs:
        print(f"  input: {input}\n  input type:  {input_type}\n")

    print("\nPyTorch unsupported inputs:")
    for input, input_type, error_msg in torch_unsupported_inputs:
        print("*"*80)
        print(f"   input:  {input}")
        print(f"   input type:  {input_type}")
        print(f"   error message:  {error_msg}")
        print("*"*80)


'''
2. 测试使用接口构造函数/神经网络的准确性
b) 使用Pytorch和MindSpore，固定输入和权重，测试正向推理结果（误差范围小于1e-3，若报错则记录报错信息）
c) 测试该神经网络/函数反向，如果是神经网络，则测试Parameter的梯度，如果是函数，则测试函数输入的梯度
'''

@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_batch_norm_forward_backward(mode):
    ms.set_context(mode=mode)
    # 特征均值和方差
    ms_running_mean = ms.tensor(np.zeros(3), dtype=ms.float32)
    ms_running_var = ms.tensor(np.ones(3), dtype=ms.float32)
    torch_running_mean = torch.tensor(np.zeros(3), dtype=torch.float32)
    torch_running_var = torch.tensor(np.ones(3), dtype=torch.float32) 

    # PyTorch模型定义
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()

        def forward(self, x):
            x = torch.nn.functional.batch_norm(x, torch_running_mean, torch_running_var)
            return x

    # MindSpore模型定义
    class SimpleModelMS(mnn.Cell):
        def __init__(self):
            super(SimpleModelMS, self).__init__()

        def construct(self, x):
            x = mint.nn.functional.batch_norm(x, ms_running_mean, ms_running_var)
            return x

    # 创建固定输入数据
    input_data_np = np.random.randn(1, 3, 8, 8).astype(np.float32)

    # PyTorch模型设置
    model_torch = SimpleModel()
    input_torch = torch.tensor(input_data_np, requires_grad=True)

    # MindSpore模型设置
    model_ms = SimpleModelMS()
    input_ms = Tensor(input_data_np, dtype=ms.float32)

    # 正向传播
    output_torch = model_torch(input_torch)
    output_ms = model_ms(input_ms)
    
    # 比较正向传播结果
    np.testing.assert_allclose(output_ms.asnumpy(), output_torch.detach().numpy(), rtol=1e-3)

    # 反向传播
    output_torch.sum().backward()
    grad_input_torch = input_torch.grad.numpy()

    # 反向传播（MindSpore）
    grad_fn = ops.GradOperation(get_all=True)
    grad_input_ms = grad_fn(model_ms)(input_ms)[0]

    # 比较反向传播结果（输入梯度）
    np.testing.assert_allclose(grad_input_ms.asnumpy(), grad_input_torch, rtol=1e-3)
