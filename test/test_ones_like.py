import pytest
import random
import numpy as np
import mindspore as ms
from mindspore import mint, Tensor, value_and_grad
import torch

"""
mindspore.mint.ones_like(input, *, dtype=None)
创建一个数值全为1的Tensor，shape和 input 相同，dtype由 dtype 决定。
如果 dtype = None，输出Tensor的数据类型会和 input 一致。

参数：input (Tensor) - 任意维度的Tensor。
关键字参数：dtype (mindspore.dtype, 可选) - 用来描述所创建的Tensor的 dtype。如果为 None ，那么将会使用 input 的dtype。默认值： None 。
返回：Tensor，具有与 input 相同的shape并填充了1。
异常：TypeError - input 不是Tensor。
"""

"""
torch.ones_like()是 PyTorch 中用于创建与给定张量具有相同形状和数据类型的全为 1 的张量的函数。
用法
    torch.ones_like(input_tensor, dtype=None, layout=None, device=None, requires_grad=False)
参数
    input_tensor: 输入张量，ones_like 将根据此张量的形状和数据类型创建新的张量。
    dtype (可选): 指定返回张量的数据类型。如果未指定，默认使用 input_tensor 的数据类型。
    layout (可选): 指定张量的布局（例如，稀疏或稠密）。默认值为 torch.strided。
    device (可选): 指定张量的设备（如 CPU 或 GPU）。默认与 input_tensor 相同。
    requires_grad (可选): 如果设置为 True，则在计算图中跟踪该张量的操作。默认值为 False。
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

    ms_result = mint.ones_like(ms_tensor, dtype=dtype_ms).numpy()
    torch_result = torch.ones_like(torch_tensor, dtype=dtype_torch).numpy()

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

        ms_tensor = Tensor(input_data, dtype=dtype_ms)
        torch_tensor = torch.tensor(input_data, dtype=dtype_torch)

        err = False
        try:
            ms_result = mint.ones_like(ms_tensor, dtype=dtype_ms).asnumpy()
        except Exception as e:
            err = True
            print(f"mint.ones_like not supported for {dtype_ms}")

        try:
            torch_result = torch.ones_like(torch_tensor, dtype=dtype_torch).numpy()
        except Exception as e:
            err = True
            print(f"torch.ones_like not supported for {dtype_torch}")

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
        ms_tensor = mint.ones_like(input_tensor, dtype=ms.float32)
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
            x_ms = mint.ones_like(ms_tensor, dtype=ms_dtype)
            y_ms = Tensor(weight, ms_dtype)
            grad_fn = value_and_grad(forward)
            z_ms, gradient_ms= grad_fn(x_ms, y_ms)

            torch_tensor = torch.tensor(x_data, dtype=torch_dtype)
            x_torch = torch.ones_like(torch_tensor, dtype=torch_dtype, requires_grad=True)
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

        