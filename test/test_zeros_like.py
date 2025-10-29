import pytest
import random
import numpy as np
import mindspore as ms
from mindspore import mint, Tensor, value_and_grad
import torch

"""
mindspore.mint.zeros_like(input, *, dtype=None)
创建一个数值全为0的Tensor，shape和 input 相同，dtype由 dtype 决定。
如果 dtype = None，输出Tensor的数据类型会和 input 一致。

参数：input (Tensor) - 用来描述所创建的Tensor的shape。
关键字参数：dtype (mindspore.dtype, 可选) - 用来描述所创建的Tensor的 dtype。如果为 None ，那么将会使用 input 的dtype。默认值： None 。
返回：返回一个用0填充的Tensor。
异常：TypeError - 如果 dtype 不是MindSpore的dtype。
"""

"""
torch.zeros_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format) → Tensor

返回一个填充了标量值 0 的张量，其大小与输入张量相同。torch.zeros_like(input) 等效于 torch.zeros(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)。

参数
    input (Tensor): 输入张量的大小将决定输出张量的大小。
关键字参数
    dtype (torch.dtype, 可选): 返回张量的期望数据类型。默认值：如果为 None，则默认为输入张量的 dtype。
    layout (torch.layout, 可选): 返回张量的期望布局。默认值：如果为 None，则默认为输入张量的布局。
    device (torch.device, 可选): 返回张量的期望设备。默认值：如果为 None，则默认为输入张量的设备。
    requires_grad (bool, 可选): 如果需要自动求导记录在返回张量上的操作。默认值：False。
    memory_format (torch.memory_format, 可选): 返回张量的期望内存格式。默认值：torch.preserve_format。
"""

mindspore_dtype_list = [
    ms.int8,
    ms.int16, 
    ms.int32, 
    ms.int64, 
    ms.uint8, 
    ms.uint16,
    ms.uint32, 
    ms.uint64,
    ms.float16, 
    ms.float32, 
    ms.float64, 
    ms.bfloat16,
    ms.complex64,
    ms.complex128,
    ms.bool_
]

pytorch_dtype_list = [
    torch.int8,    
    torch.int16,    
    torch.int32,   
    torch.int64,  
    torch.uint8,    
    torch.uint16,
    torch.uint32,
    torch.uint64,
    torch.float16, 
    torch.float32,  
    torch.float64,  
    torch.bfloat16, 
    torch.complex64, 
    torch.complex128,
    torch.bool
]


gradient_dtype_list = {
    ms.float16: torch.float16,
    ms.float32: torch.float32,
    ms.float64: torch.float64
}



def is_same(input_data=[[1,0],[0,0],[1,1]], shape=None, dtype_ms=ms.float32, dtype_torch=torch.float32):
    if shape != None:
        input_data = np.random.randn(*shape)

    ms_tensor = Tensor(input_data, dtype=dtype_ms)
    torch_tensor = torch.tensor(input_data, dtype=dtype_torch)

    ms_result = mint.zeros_like(ms_tensor, dtype=dtype_ms).numpy()
    torch_result = torch.zeros_like(torch_tensor, dtype=dtype_torch).numpy()

    return np.allclose(ms_result, torch_result, atol=1e-3)

def generate_input_shapes(min_dims=1, max_dims=7, min_size=1, max_size=10):
    """
    生成随机的输入形状，如 [2, 3], [3, 5, 6], [7, 8, 9]。

    参数:
    - min_dims: 最小维度
    - max_dims: 最大维度
    - min_size: 每个维度的最小大小
    - max_size: 每个维度的最大大小

    返回:
    - shapes: 生成的输入形状列表
    """
    shapes = []
    for num_dims in range(min_dims, max_dims + 1):
        shape = [random.randint(min_size, max_size) for _ in range(num_dims)]
        shapes.append(shape)
    return shapes

@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_any_different_dtypes(mode):
    """测试random输入不同dtype，对比两个框架的支持度"""
    ms.set_context(mode=mode)
    
    input_data = np.random.randn(2, 3)
    for i in range(len(mindspore_dtype_list)):
        dtype_ms = mindspore_dtype_list[i]
        dtype_torch = pytorch_dtype_list[i]

        ms_tensor = Tensor(input_data, dtype_ms)
        torch_tensor = torch.tensor(input_data, dtype=dtype_torch)

        err = False
        try:
            ms_result = mint.zeros_like(ms_tensor, dtype=dtype_ms).asnumpy()
        except Exception as e:
            err = True
            print(f"mint.zeros_like not supported for {dtype_ms}")

        try:
            torch_result = torch.zeros_like(torch_tensor, dtype=dtype_torch).numpy()
        except Exception as e:
            err = True
            print(f"torch.zeros_like not supported for {dtype_torch}")

        if not err:
            assert np.allclose(ms_result, torch_result, atol=1e-3)


@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_any_random_input_fixed_dtype(mode):
    """测试固定数据类型下的随机输入值，对比输出误差"""
    ms.set_context(mode=mode)

    shape_list = generate_input_shapes()
    for i in range(len(mindspore_dtype_list)):
        dtype_ms = mindspore_dtype_list[i]
        dtype_torch = pytorch_dtype_list[i]
        for i in range(len(shape_list)):
            shape = shape_list[i]
            result = is_same(shape=shape, dtype_ms=dtype_ms, dtype_torch=dtype_torch)
            assert result


@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_any_wrong_input(mode):
    """测试随机混乱输入，报错信息的准确性"""
    ms.set_context(mode=mode)
    
    # 异常：TypeError - input 不是Tensor。
    try:
        input_tensor = "wrong_input"
        ms_tensor = mint.zeros_like(input_tensor, dtype=ms.float32)
    except Exception as e:
        print(f"Error with wrong input: {e}")


@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_any_forward_back(mode):
    """使用Pytorch和MindSpore, 固定输入和权重, 测试正向推理结果和反向梯度"""
    ms.set_context(mode=mode)

    def forward(x, y):
        return x * y
    
    shapes_list = generate_input_shapes()
    for ms_dtype, torch_dtype in gradient_dtype_list.items():
        for shape in shapes_list:
            x_data = np.random.randn(*shape)
            weight = np.random.rand(*shape)

            ms_tensor = Tensor(x_data, dtype=ms_dtype)
            x_ms = mint.zeros_like(ms_tensor, dtype=ms_dtype)
            y_ms = Tensor(weight, ms_dtype)
            grad_fn = value_and_grad(forward)
            z_ms, gradient_ms= grad_fn(x_ms, y_ms)

            torch_tensor = torch.tensor(x_data, dtype=torch_dtype)
            x_torch = torch.zeros_like(torch_tensor, dtype=torch_dtype, requires_grad=True)
            y_torch = torch.tensor(weight, dtype=torch_dtype, requires_grad=True)
            z_torch = forward(x_torch, y_torch)
            loss = z_torch.sum()
            loss.backward()
            gradient_torch = x_torch.grad

            assert np.allclose(z_ms.asnumpy(), z_torch.detach().numpy(), atol=1e-3)
            assert np.allclose(gradient_ms.asnumpy(), gradient_torch.numpy(), atol=1e-3)


if __name__ == '__main__':
    test_any_different_dtypes(ms.GRAPH_MODE)
    test_any_different_dtypes(ms.PYNATIVE_MODE)
    test_any_random_input_fixed_dtype(ms.GRAPH_MODE)
    test_any_random_input_fixed_dtype(ms.PYNATIVE_MODE)
    test_any_wrong_input(ms.GRAPH_MODE)
    test_any_wrong_input(ms.PYNATIVE_MODE)
    test_any_forward_back(ms.GRAPH_MODE)
    test_any_forward_back(ms.PYNATIVE_MODE)

        