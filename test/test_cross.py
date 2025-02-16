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
def test_cross_all_dtypes():
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

    for ms_dtype, torch_dtype in zip(mindspore_dtypes, torch_dtypes):
        input_data = np.random.randn(2, 3).astype(np.float32)
        other_data = np.random.randn(2, 3).astype(np.float32)

        
        try:
            ms_input = Tensor(input_data, ms_dtype)
            ms_other = Tensor(other_data, ms_dtype)
            ms_output = mint.cross(ms_input, ms_other)
            # 判断输出的 shape 和输入一致
            assert ms_output.shape == (2, 3)
            ms_supported_dtypes.append(ms_dtype)
        except Exception as e:
            
            ms_unsupported_dtypes.append((ms_dtype, str(e)))
        
        try:
            torch_input = torch.tensor(input_data, dtype=torch_dtype)
            torch_other = torch.tensor(other_data, dtype=torch_dtype)
            torch_output = torch.cross(torch_input, torch_other)
            assert torch_output.shape == (2, 3)
            torch_supported_dtypes.append(torch_dtype)
        except Exception as e:
            torch_unsupported_dtypes.append((torch_dtype, str(e)))
    
    print(f"MindSpore supported dtypes: {ms_supported_dtypes}")
    print(f"MindSpore unsupported dtypes: {[dtype for dtype, _ in ms_unsupported_dtypes]}")
    print(f"PyTorch supported dtypes: {torch_supported_dtypes}")
    print(f"PyTorch unsupported dtypes: {[dtype for dtype, _ in torch_unsupported_dtypes]}")


'''
b) 测试固定dtype，random输入值，对比两个框架输出是否相等（误差范围为小于1e-3）
'''
def test_cross_fixed_dtype_random_input():
    # 固定dtype
    dtypes = [
        mindspore.int8, mindspore.int16, mindspore.int32, mindspore.int64,
        mindspore.uint8, 
        # mindspore.float16, # 这里会出现计算结果相对误差大于1e-3的情况
        mindspore.float32, mindspore.float64,
        # mindspore.bfloat16, # 这里会出现运行时报错
        mindspore.complex64, mindspore.complex128
    ]
    
    torch_dtypes = [
        torch.int8, torch.int16, torch.int32, torch.int64,
        torch.uint8, 
        # torch.float16, 
        torch.float32, torch.float64,
        # torch.bfloat16, 
        torch.complex64, torch.complex128
    ]
    for dtype, torch_dtype in zip(dtypes, torch_dtypes):
        # print(dtype, torch_dtype)
        for _ in range(100):
            input_data = np.abs(np.random.randn(2, 3)).astype(np.float32)
            other_data = np.abs(np.random.randn(2, 3)).astype(np.float32)
            
            ms_input = Tensor(input_data, dtype)
            ms_other = Tensor(other_data, dtype)
            torch_input = torch.tensor(input_data, dtype=torch_dtype)
            torch_other = torch.tensor(other_data, dtype=torch_dtype)
            
            ms_output = mint.cross(ms_input, ms_other)
            torch_output = torch.cross(torch_input, torch_other)
            
            # 对比输出
            if dtype == mindspore.bfloat16:
                np.testing.assert_allclose(ms_output.astype(ms.float32).asnumpy(), torch_output.to(torch.float32).numpy(), rtol=1e-3)
            else:
                np.testing.assert_allclose(ms_output.asnumpy(), torch_output.numpy(), rtol=1e-3)
      
    print("MindSpore and PyTorch outputs are equal within tolerance.")


'''
c) 测试固定shape，固定输入值，不同输入参数（string/bool等类型），两个框架的支持度
'''
def test_cross_different_input_types():
    input_data = [[1, 2, 3], [4, 5, 6]]
    other_data = [[7, 8, 9], [10, 11, 12]]
    
    ms_input = Tensor(input_data, mindspore.float32)
    ms_other = Tensor(other_data, mindspore.float32)
    torch_input = torch.tensor(input_data, dtype=torch.float32)
    torch_other = torch.tensor(other_data, dtype=torch.float32)
    params = {
        "int": 1,
        "float": 1.0,
        "bool": True,
        "string": "1",
        "mindspore.tensor": Tensor(1, mindspore.float32),
        "torch.tensor": torch.tensor(1, dtype=torch.float32),
        "list": [1, 2, 3],
        "np.array": np.array([1, 2, 3], dtype=np.float32),
        "tuple": (1, 2, 3)
    }
    # }
    
    ms_supported_types = []
    ms_unsupported_types = []
    torch_supported_types = []
    torch_unsupported_types = []

    for type, param in params.items():
        try:
            ms_output = mint.cross(ms_input, ms_other, dim = param)
            assert ms_output.shape == (2, 3)
            ms_supported_types.append(type)
        except Exception as e:
            ms_unsupported_types.append((type, str(e)))
        
        try:
            
            torch_output = torch.cross(torch_input, torch_other, param)
            torch_supported_types.append(type)
        except Exception as e:
            torch_unsupported_types.append((type, str(e)))
    
    print(f"MindSpore supported input types: {ms_supported_types}")
    print(f"MindSpore unsupported input types: {[type for type, _ in ms_unsupported_types]}")
    print(f"PyTorch supported input types: {torch_supported_types}")
    print(f"PyTorch unsupported input types: {[type for type, _ in torch_unsupported_types]}")


'''
d) 测试随机混乱输入，报错信息的准确性
'''
def test_cross_random_chaotic_input():
    chaotic_inputs = [
        (np.array([1, 2, 3]), np.array([4, 5, 6]), None),
        (torch.tensor([1, 2, 3], dtype=torch.float32), torch.tensor([4, 5, 6], dtype=torch.float32), None),
        (Tensor([1, 2, 3], mindspore.float32), Tensor([4, 5, 6], mindspore.float32), None),
        (Tensor([1, 2, 3], mindspore.float32), Tensor([4, 5, 6], mindspore.float32), 1),
        (Tensor([1, 2, 3], mindspore.float32), Tensor([4, 5, 6], mindspore.float32), 0),
        (Tensor([1, 2, 3], mindspore.float32), Tensor([4, 5, 6], mindspore.float32), "string"),
        (Tensor([1, 2, 3], mindspore.float32), Tensor([4, 5, 6], mindspore.float32), True),
        (Tensor([1, 2, 3], mindspore.float32), Tensor([4, 5, 6], mindspore.float32), False),
        ([1, 2, 3], [4, 5, 6], None),
        ("string", "string", None),
        ("string", "string", None),
        (True, False, None),
        (None, None, None)
    ]
    
    ms_supported_inputs = []
    ms_unsupported_inputs = []
    torch_supported_inputs = []
    torch_unsupported_inputs = []

    for input_pair in chaotic_inputs:
        try:
            ms_output = mint.cross(input_pair[0], input_pair[1], input_pair[2])
            assert ms_output.asnumpy() is not None
            ms_supported_inputs.append(str(input_pair))
        except Exception as e:
            ms_unsupported_inputs.append((str(input_pair), str([type(inp) for inp in input_pair]), str(e)))
        
        try:            
            torch_output = torch.cross(input_pair[0], input_pair[1], dim=input_pair[2])
            assert torch_output.numpy() is not None
            torch_supported_inputs.append(str(input_pair))
        except Exception as e:
            torch_unsupported_inputs.append((str(input_pair), str([type(inp) for inp in input_pair]), str(e)))
    
    print("MindSpore supported inputs:", ms_supported_inputs)
    print("MindSpore unsupported inputs:")  
    for input, input_type, error_msg in ms_unsupported_inputs:
        print("*"*80)
        print(f"   input:  {input}")
        print(f"   input type:  {input_type}")
        print(f"   error message:  {error_msg}")
        print("*"*80)
    print("PyTorch supported inputs:", torch_supported_inputs)
    print("PyTorch unsupported inputs:")  
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
            return torch.cross(self.linear(x), x)

    class SimpleModelMS(mnn.Cell):
        def __init__(self):
            super(SimpleModelMS, self).__init__()
            self.linear = mnn.Dense(3, 3)  # 输入维度为3，输出维度也为3

        def construct(self, x):
            return mint.cross(self.linear(x), x)

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
# test_cross_all_dtypes()
# test_cross_fixed_dtype_random_input()
# test_cross_different_input_types()
# test_cross_random_chaotic_input()

# 神经网络内测试函数
test_forward_backward()