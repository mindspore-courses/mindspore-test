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
def test_clamp_all_dtypes():
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
        input_data = np.random.randn(5).astype(np.float32)
        ms_input = Tensor(input_data, ms_dtype)
        torch_input = torch.tensor(input_data, dtype=torch_dtype)
        
        min_val = -1.0
        max_val = 1.0

        try:
            ms_output = mint.clamp(ms_input, min=min_val, max=max_val)
            # 对比结果shape
            assert ms_output.shape == ms_input.shape

            ms_supported_dtypes.append(ms_dtype)
        except Exception as e:
            ms_unsupported_dtypes.append((ms_dtype, str(e)))
        
        try:
            
            torch_output = torch.clamp(torch_input, min=min_val, max=max_val)
            # 对比结果shape
            assert torch_output.shape == torch_input.shape

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
def test_clamp_fixed_dtype_random_input():
    # 固定dtype
    dtypes = [
        mindspore.int8, mindspore.int16, mindspore.int32, mindspore.int64,
        mindspore.uint8, mindspore.float16, mindspore.float32, mindspore.float64,
        mindspore.bfloat16, 
    ]
    
    torch_dtypes = [
        torch.int8, torch.int16, torch.int32, torch.int64,
        torch.uint8, torch.float16, torch.float32, torch.float64,
        torch.bfloat16, 
    ]
    for dtype, torch_dtype in zip(dtypes, torch_dtypes):
        for _ in range(100):
            input_data = np.abs(np.random.randn(10)).astype(np.float32)
            ms_input = Tensor(input_data, dtype)
            torch_input = torch.tensor(input_data, dtype=torch_dtype)
            
            min_val = 1.0
            max_val = 2.0
            
            ms_output = mint.clamp(ms_input, min=min_val, max=max_val)
            torch_output = torch.clamp(torch_input, min=min_val, max=max_val)

            # 对比输出
            if dtype == mindspore.bfloat16:
                np.testing.assert_allclose(ms_output.astype(ms.float32).asnumpy(), torch_output.to(torch.float32).numpy(), rtol=1e-3)
            else:
                np.testing.assert_allclose(ms_output.asnumpy(), torch_output.numpy(), rtol=1e-3)
    
    print("MindSpore and PyTorch outputs are equal within tolerance.")



'''
c) 测试固定shape，固定输入值，不同输入参数（string/bool等类型），两个框架的支持度
'''
def test_clamp_different_input_types():
    shape = (3,)
    input_data = np.random.randn(*shape).astype(np.float32)
    ms_input = Tensor(input_data, mindspore.float32)
    torch_input = torch.tensor(input_data, dtype=torch.float32)
    
    params = {
        "mindspore.tensor": {"min": Tensor(-1.0, mindspore.float32), "max": Tensor(1.0, mindspore.float32)},
        "torch.tensor": {"min": torch.tensor(-1.0, dtype=torch.float32), "max": torch.tensor(1.0, dtype=torch.float32)},
        "int": {"min": -1, "max": 1},
        "float": {"min": -1.0, "max": 1.0},
        "bool": {"min": False, "max": True},
        "string": {"min": "1.0", "max": "2.0"},
        "list": {"min": [1.0, 2.0], "max": [3.0, 4.0]},
        "tuple": {"min": (1.0, 2.0), "max": (3.0, 4.0)},
        "None": {"min": None, "max": None},
    }
    
    ms_supported_types = []
    ms_unsupported_types = []
    torch_supported_types = []
    torch_unsupported_types = []


    for type, param in params.items():
        try:
            ms_output = mint.clamp(ms_input, **param)
            assert ms_output.shape == ms_input.shape
            ms_supported_types.append(type)
        except Exception as e:
            ms_unsupported_types.append((type, str(e)))
        
        try:
            torch_output = torch.clamp(torch_input, **param)
            assert torch_output.shape == torch_input.shape
            torch_supported_types.append(type)
        except Exception as e:
            torch_unsupported_types.append((type, str(e)))
    
    print(f"MindSpore supported types: {ms_supported_types}")
    print(f"MindSpore unsupported types: {[type for type, _ in ms_unsupported_types]}")
    print(f"PyTorch supported types: {torch_supported_types}")
    print(f"PyTorch unsupported types: {[type for type, _ in  torch_unsupported_types]}")


'''
d) 测试随机混乱输入，报错信息的准确性
'''
def test_clamp_random_chaotic_input():
    chaotic_inputs = [
        np.array([1.1, 2.5, -1.5], dtype=np.float32),  # np.array 输入
        torch.tensor([1.1, 2.5, -1.5], dtype=torch.float32),  # PyTorch tensor 浮点数
        torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32),  # PyTorch tensor 整型
        mindspore.tensor(np.array([1.1, 2.5, -1.5]), dtype=mindspore.float32),  # MindSpore mindspore.tensor 浮点数
        mindspore.tensor(np.array([[1, 2, 3], [4, 5, 6]]), dtype=mindspore.int32),  # MindSpore mindspore.tensor 整型
        [1.1, 2.5, -1.5],  # Python list
        None
    ]

    chaotic_params = [
        {"min": None, "max": Tensor(1.0, mindspore.float32)},
        {"min": Tensor(-1.0, mindspore.float32), "max": None},
        {"min": Tensor(-1.0, mindspore.float32), "max": Tensor(1.0, mindspore.float32)},
        {"min": None, "max": torch.tensor(1.0, dtype=torch.float32)},
        {"min": torch.tensor(-1.0, dtype=torch.float32), "max": None},
        {"min": torch.tensor(-1.0, dtype=torch.float32), "max": torch.tensor(1.0, dtype=torch.float32)},
        {"min": 1.0, "max": -1.0},  # Invalid case
        {"min": "string", "max": None},  # Invalid type
        {"min": True, "max": None},  # Invalid type
        {"min": None, "max": None},  # Invalid case
    ]
    
    ms_supported_inputs = []
    ms_unsupported_inputs = []
    torch_supported_inputs = []
    torch_unsupported_inputs = []

    for input_data in chaotic_inputs:
        for param in chaotic_params:
            try:
                ms_output = mint.clamp(input_data, **param)
                assert ms_output.asnumpy() is not None 
                ms_supported_inputs.append(str([input_data, param["min"], param["max"]]))
            except Exception as e:
                ms_unsupported_inputs.append((str([input_data, param["min"], param["max"]]), 
                                              str([type(input_data), type(param["min"]), type(param["max"])]), 
                                              str(e)))
            
            try:
                torch_output = torch.clamp(input_data, **param)
                assert torch_output.numpy() is not None
                torch_supported_inputs.append(str([input_data, param["min"], param["max"]]))
            except Exception as e:
                torch_unsupported_inputs.append((str([input_data, param["min"], param["max"]]), 
                                                 str([type(input_data), type(param["min"]), type(param["max"])]), 
                                                 str(e)))
    
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
            return torch.clamp(self.linear(x), -1, 1)

    class SimpleModelMS(mnn.Cell):
        def __init__(self):
            super(SimpleModelMS, self).__init__()
            self.linear = mnn.Dense(3, 3)  # 输入维度为3，输出维度也为3

        def construct(self, x):
            return mint.clamp(self.linear(x), -1, 1)

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
# test_clamp_all_dtypes()
# test_clamp_fixed_dtype_random_input()
# test_clamp_different_input_types()
# test_clamp_random_chaotic_input()

# 神经网络内测试函数
test_forward_backward()