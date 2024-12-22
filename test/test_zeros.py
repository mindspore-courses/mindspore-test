import pytest
import random
import numpy as np
import mindspore as ms
from mindspore import mint, Tensor, value_and_grad
import torch

"""
mindspore.mint.zeros(size, *, dtype=None)
创建一个值全为0的Tensor。第一个参数 size 指定Tensor的shape，第二个参数 dtype 指定填充值的数据类型。

参数：size (Union[tuple[int], list[int], int, Tensor]) - 用来描述所创建的Tensor的shape，只允许正整数或者包含正整数的tuple、list、Tensor。 
如果是一个Tensor，必须是一个数据类型为int32或者int64的0-D或1-D Tensor。
关键字参数：dtype (mindspore.dtype, 可选) - 用来描述所创建的Tensor的dtype。如果为 None ，那么将会使用mindspore.float32。默认值： None 。
返回：Tensor，dtype和shape由入参决定。
异常：TypeError - 如果 size 不是一个int，或元素为int的元组、列表、Tensor。
"""

"""
torch.zeros(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
    返回一个填充了标量值 0 的张量，形状由变量参数 size 定义。

参数
    size (int...)：一个整数序列，定义输出张量的形状。可以是可变数量的参数，也可以是列表或元组等集合。
关键字参数
    out (Tensor, 可选)：输出张量。
    dtype (torch.dtype, 可选)：返回张量的期望数据类型。默认值：如果为 None，则使用全局默认值（见 torch.set_default_dtype()）。
    layout (torch.layout, 可选)：返回张量的期望布局。默认值：torch.strided。
    device (torch.device, 可选)：返回张量的期望设备。默认值：如果为 None，则使用默认张量类型的当前设备（见 torch.set_default_device()）。对于 CPU 张量类型，设备将是 CPU；对于 CUDA 张量类型，设备将是当前的 CUDA 设备。
    requires_grad (bool, 可选)：如果自动求导应记录对返回张量的操作。默认值：False。
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
    torch.float32,  
    torch.float64, 
    torch.float16,  
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



def is_same(shape, dtype_ms=ms.float32, dtype_torch=torch.float32):

    ms_result = mint.zeros(shape, dtype=dtype_ms).numpy()
    torch_result = torch.zeros(shape, dtype=dtype_torch).numpy()

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

def generate_tuple_input_shapes(min_dims=1, max_dims=7, min_size=1, max_size=10):
    """
    生成随机的输入形状，如 (2, 3), (3, 5, 6), (7, 8, 9)。

    参数:
    - min_dims: 最小维度
    - max_dims: 最大维度
    - min_size: 每个维度的最小大小
    - max_size: 每个维度的最大大小

    返回:
    - shapes: 生成的输入形状元组列表
    """
    shapes = []
    for num_dims in range(min_dims, max_dims + 1):
        shape = tuple(random.randint(min_size, max_size) for _ in range(num_dims))
        shapes.append(shape)
    return shapes

@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_any_different_dtypes(mode):
    """测试random输入不同dtype，对比两个框架的支持度"""
    ms.set_context(mode=mode)
    
    shape = generate_input_shapes()[3]
    for i in range(len(mindspore_dtype_list)):
        dtype_ms = mindspore_dtype_list[i]
        dtype_torch = pytorch_dtype_list[i]

        err = False
        try:
            ms_result = mint.zeros(shape, dtype=dtype_ms).asnumpy()
        except Exception as e:
            err = True
            print(f"mint.zeros not supported for {dtype_ms}")

        try:
            torch_result = torch.zeros(shape, dtype=dtype_torch).numpy()
        except Exception as e:
            err = True
            print(f"torch.zeros not supported for {dtype_torch}")

        if not err:
            assert np.allclose(ms_result, torch_result, atol=1e-3)

@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_any_random_input_fixed_dtype(mode):
    """测试固定数据类型下的随机输入值，对比输出误差"""
    ms.set_context(mode=mode)

    # size (list[int])
    shape_list = generate_input_shapes()
    for i in range(len(mindspore_dtype_list)):
        dtype_ms = mindspore_dtype_list[i]
        dtype_torch = pytorch_dtype_list[i]
        for i in range(len(shape_list)):
            shape = shape_list[i]
            result = is_same(shape=shape, dtype_ms=dtype_ms, dtype_torch=dtype_torch)
            assert result
    
    # size (tuple[int])
    tuple_shape_list = generate_tuple_input_shapes()
    for i in range(len(tuple_shape_list)):
        dtype_ms = mindspore_dtype_list[i]
        dtype_torch = pytorch_dtype_list[i]
        for i in range(len(shape_list)):
            shape = tuple_shape_list[i]
            result = is_same(shape=shape, dtype_ms=dtype_ms, dtype_torch=dtype_torch)
            assert result
    
    # size (int)
    for i in range(len(mindspore_dtype_list)):
        dtype_ms = mindspore_dtype_list[i]
        dtype_torch = pytorch_dtype_list[i]
        shape = np.random.randint(1, 1000)
        result = is_same(shape=shape, dtype_ms=dtype_ms, dtype_torch=dtype_torch)
        assert result
    
    # size (Tensor) 
    for i in range(len(mindspore_dtype_list)):
        if mindspore_dtype_list[i] != ms.int32 or mindspore_dtype_list[i] != ms.int64:
            continue
        dtype_ms = mindspore_dtype_list[i]
        dtype_torch = pytorch_dtype_list[i]
        shape = np.random.randint(1, 32)
        shape_tensor = np.random.rand(shape)
        ms_tensor = Tensor(shape_tensor, dtype=dtype_ms)
        result = is_same(shape=ms_tensor, dtype_ms=dtype_ms, dtype_torch=dtype_torch)
        assert result



@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_any_wrong_input(mode):
    """测试随机混乱输入，报错信息的准确性"""
    ms.set_context(mode=mode)
    
    # 异常：TypeError - input 不是Tensor。
    try:
        input_tensor = "wrong_input"
        ms_tensor = mint.zeros(input_tensor, dtype=ms.float32)
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
            weight = np.random.rand(*shape)

            x_ms = mint.zeros(shape, dtype=ms_dtype)
            y_ms = Tensor(weight, ms_dtype)
            grad_fn = value_and_grad(forward)
            z_ms, gradient_ms= grad_fn(x_ms, y_ms)

            x_torch = torch.zeros(shape, dtype=torch_dtype, requires_grad=True)
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

        