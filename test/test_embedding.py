import torch
import torch.nn as nn
import pytest
import numpy as np
import mindspore as ms
from mindspore import Tensor, mint, ops
import mindspore.nn as mnn


# 设置随机种子
seed = 42
torch.manual_seed(seed)
ms.set_seed(seed)

# 设置上下文，指定设备为 Ascend
ms.set_context(device_target="Ascend", device_id=0)


'''
1. 测试embedding的不同dtype支持度：
'''
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_embedding_all_dtypes(mode):
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
    ]

    ms_supported_dtypes = []
    ms_unsupported_dtypes = []
    torch_supported_dtypes = []
    torch_unsupported_dtypes = []

    # 遍历所有 dtype
    for ms_dtype, torch_dtype in zip(ms_dtypes, torch_dtypes):
        # 生成随机数据
        input_data = np.array([1, 2, 3, 4], dtype=np.int32)  # 输入的索引
        ms_input = ms.tensor(input_data, ms_dtype)
        torch_input = torch.tensor(input_data, dtype=torch_dtype)

        # 生成相应的 weight
        weight_data = np.random.randn(5, 3).astype(np.float32)
        ms_weight = ms.tensor(weight_data, ms_dtype)
        torch_weight = torch.tensor(weight_data, dtype=torch_dtype)

        try:
            # 调用 embedding 函数
            ms_output = mint.nn.functional.embedding(ms_input, ms_weight)
            ms_supported_dtypes.append(ms_dtype)
        except Exception as e:
            ms_unsupported_dtypes.append(ms_dtype)

        try:
            # 调用 embedding 函数
            torch_output = torch.nn.functional.embedding(torch_input, torch_weight)
            torch_supported_dtypes.append(torch_dtype)
        except Exception as e:
            torch_unsupported_dtypes.append(torch_dtype)

    print(f"MindSpore supported dtypes: {ms_supported_dtypes}")
    print(f"MindSpore unsupported dtypes: {ms_unsupported_dtypes}")
    print(f"PyTorch supported dtypes: {torch_supported_dtypes}")
    print(f"PyTorch unsupported dtypes: {torch_unsupported_dtypes}")


'''
2. 测试固定输入和权重，对比两个框架的输出是否相等
'''
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_embedding_fixed_input_output(mode):
    ms.set_context(mode=mode)

    # 固定输入和权重
    num_embeddings = 5
    embedding_dim = 3
    input_data_np = np.array([1, 2, 3, 4], dtype=np.int32)  # 输入的索引
    weight_np = np.random.randn(num_embeddings, embedding_dim).astype(np.float32)  # embedding的权重

    # PyTorch模型设置
    torch_input = torch.tensor(input_data_np, dtype=torch.long)
    torch_weight = torch.tensor(weight_np, dtype=torch.float32)

    # MindSpore模型设置
    ms_input = Tensor(input_data_np, dtype=ms.int32)
    ms_weight = Tensor(weight_np, dtype=ms.float32)

    # 正向传播
    output_torch = torch.nn.functional.embedding(torch_input, torch_weight)
    output_ms = mint.nn.functional.embedding(ms_input, ms_weight).asnumpy()

    # 比较正向传播结果
    diff = np.abs(output_torch.detach().numpy() - output_ms)
    if np.all(diff < 1e-3):
        print("Forward pass results match within tolerance.")
    else:
        print("Forward pass results do not match.")


'''
3. 测试固定形状和不同输入类型（如字符串、布尔值等）
'''
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_embedding_different_input_types(mode):
    ms.set_context(mode=mode)

    # 定义不同的输入类型
    input_data = [1, 2, 3, 4]  # 基础数据
    input_types = {
        "list": input_data,  # Python list
        "np.array": np.array(input_data, dtype=np.int32),  # NumPy array
        "mindspore.tensor": ms.tensor(input_data, ms.int32),  # MindSpore tensor
        "torch.tensor": torch.tensor(input_data, dtype=torch.long),  # PyTorch tensor
        "tuple": tuple(input_data),  # Python tuple
        "string": "1, 2, 3, 4",  # String
    }

    ms_supported_types = []
    ms_unsupported_types = []
    torch_supported_types = []
    torch_unsupported_types = []

    # 遍历所有输入类型
    for input_name, input_value in input_types.items():
        try:
            # 测试 MindSpore
            ms_output = mint.nn.functional.embedding(input_value, ms.tensor(np.random.randn(5, 3), ms.float32))
            ms_supported_types.append(input_name)
        except Exception as e:
            ms_unsupported_types.append(input_name)

        try:
            # 测试 PyTorch
            torch_output = torch.nn.functional.embedding(input_value, torch.tensor(np.random.randn(5, 3), dtype=torch.float32))
            torch_supported_types.append(input_name)
        except Exception as e:
            torch_unsupported_types.append(input_name)

    print(f"MindSpore supported input types: {ms_supported_types}")
    print(f"MindSpore unsupported input types: {ms_unsupported_types}")
    print(f"PyTorch supported input types: {torch_supported_types}")
    print(f"PyTorch unsupported input types: {torch_unsupported_types}")


'''
4. 测试随机混乱输入，报错信息的准确性
'''
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_embedding_random_chaotic_input(mode):
    ms.set_context(mode=mode)

    # 定义随机混乱输入
    chaotic_inputs = [
        np.array([1, 2, 3, 4], dtype=np.int32),  # np.array 输入
        torch.tensor([1, 2, 3, 4], dtype=torch.long),  # PyTorch tensor 整型
        ms.tensor(np.array([1, 2, 3, 4]), dtype=ms.int32),  # MindSpore tensor 整型
        [1, 2, 3, 4],  # Python list
        (1, 2, 3, 4),  # Python tuple
        "1, 2, 3, 4",  # 字符串
    ]

    ms_supported_inputs = []
    ms_unsupported_inputs = []
    torch_supported_inputs = []
    torch_unsupported_inputs = []

    # 遍历所有混乱输入
    for input_data in chaotic_inputs:
        try:
            ms_output = mint.nn.functional.embedding(input_data, ms.tensor(np.random.randn(5, 3), ms.float32))
            ms_supported_inputs.append((str(input_data), str(type(input_data))))
        except Exception as e:
            ms_unsupported_inputs.append((str(input_data), str(type(input_data)), str(e)))

        try:
            torch_output = torch.nn.functional.embedding(input_data, torch.tensor(np.random.randn(5, 3), dtype=torch.float32))
            torch_supported_inputs.append((str(input_data), str(type(input_data))))
        except Exception as e:
            torch_unsupported_inputs.append((str(input_data), str(type(input_data)), str(e)))

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
a) 使用Pytorch和MindSpore，固定输入和权重，测试正向推理结果（误差范围小于1e-3，若报错则记录报错信息）
b) 测试该神经网络/函数反向，如果是神经网络，则测试Parameter的梯度，如果是函数，则测试函数输入的梯度
'''
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_embedding_forward_backward(mode):
    ms.set_context(mode=mode)

    # Simple PyTorch model with embedding
    class SimpleModelTorch(nn.Module):
        def __init__(self, num_embeddings, embedding_dim):
            super(SimpleModelTorch, self).__init__()
            self.embedding = nn.Embedding(num_embeddings, embedding_dim)  # 初始化embedding层

        def forward(self, x):
            return self.embedding(x)

    # Simple MindSpore model with embedding
    class SimpleModelMS(mnn.Cell):
        def __init__(self, num_embeddings, embedding_dim):
            super(SimpleModelMS, self).__init__()
            self.embedding = mnn.Embedding(num_embeddings, embedding_dim)  # 初始化embedding层

        def construct(self, x):
            return self.embedding(x)

    # 创建固定输入和权重
    num_embeddings = 5
    embedding_dim = 3
    input_data_np = np.array([1, 2, 3, 4], dtype=np.int32)  # 输入的索引
    weight_np = np.random.randn(num_embeddings, embedding_dim).astype(np.float32)  # embedding的权重

    # PyTorch模型设置
    model_torch = SimpleModelTorch(num_embeddings, embedding_dim)
    model_torch.embedding.weight.data = torch.tensor(weight_np)
    input_torch = torch.tensor(input_data_np, dtype=torch.long)

    # MindSpore模型设置
    model_ms = SimpleModelMS(num_embeddings, embedding_dim)
    # MindSpore的embedding层通过初始化时传入权重
    model_ms.embedding.embedding_table.set_data(Tensor(weight_np))  # 设置权重
    input_ms = Tensor(input_data_np, dtype=ms.int32)

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
    grad_input_torch = input_torch.grad

    # 反向传播（MindSpore）
    grad_fn = ops.GradOperation(get_all=True)
    grad_input_ms = grad_fn(model_ms)(input_ms)[0].asnumpy()

    # 比较反向传播结果（输入梯度）
    if grad_input_torch is not None and grad_input_ms is not None:
        diff_grad = np.abs(grad_input_torch - grad_input_ms)
        if np.all(diff_grad < 1e-3):
            print("Backward pass gradients match within tolerance.")
        else:
            print("Backward pass gradients do not match.")
    else:
        print("Skipping gradient comparison because one of the gradients is None.")
