import pytest
import random
import numpy as np
import mindspore as ms
from mindspore import mint, Tensor, value_and_grad
import torch

"""
mindspore.mint.cat(tensors, dim=0)
    在指定维度上拼接输入Tensor。

参数：
    tensors (Union[tuple, list]) - 输入为Tensor组成的tuple或list。元素秩相同, 即 R 。 
    dim (int) - 表示指定的维度，取值范围是 [-R,R)。默认值： 0 。

返回：Tensor，数据类型与 tensors 相同。

异常：
    TypeError - dim 不是int。
    ValueError - tensors 是不同维度的Tensor。
    ValueError - dim 的值不在区间 [-R,R) 内。
    ValueError - 除了 dim 之外， tensors 的shape不相同。
    ValueError - tensors 为空tuple或list。
支持平台：Ascend
"""

"""
torch.cat(tensors, dim=0, *, out=None) → Tensor
    该函数用于在给定维度上连接一系列张量。所有张量必须具有相同的形状（在连接维度上除外），或者是大小为 (0,) 的一维空张量。
    torch.cat() 可以视为 torch.split() 和 torch.chunk() 的逆操作。
    通过示例可以更好地理解 torch.cat()。

参数
    tensors (张量序列) – 任何 Python 张量序列，必须是相同类型的张量。提供的非空张量必须在连接维度上具有相同的形状。
    dim (int, 可选) – 连接张量的维度。
关键字参数
    out (Tensor, 可选) – 输出张量。
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



def is_same(shape, dtype_ms=ms.float32, dtype_torch=torch.float32, dim=0):
    input_x1 = np.random.randn(*shape)
    input_x2 = np.random.randn(*shape)

    ms_tensor_x1 = Tensor(input_x1, dtype=dtype_ms)
    ms_tensor_x2 = Tensor(input_x2, dtype=dtype_ms)
    ms_result = mint.cat([ms_tensor_x1, ms_tensor_x2], dim=dim).numpy()

    torch_tensor_x1 = torch.tensor(input_x1, dtype=dtype_torch)
    torch_tensor_x2 = torch.tensor(input_x2, dtype=dtype_torch)
    torch_result = torch.cat([torch_tensor_x1, torch_tensor_x2], dim=dim).numpy()

    return np.allclose(ms_result, torch_result)

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
    
    input_x1 = np.random.randn(2, 3)
    input_x2 = np.random.randn(2, 3)
    for i in range(len(mindspore_dtype_list)):
        dtype_ms = mindspore_dtype_list[i]
        dtype_torch = pytorch_dtype_list[i]

        err = False
        try:
            ms_tensor_x1 = Tensor(input_x1, dtype=dtype_ms)
            ms_tensor_x2 = Tensor(input_x2, dtype=dtype_ms)
            ms_result = mint.cat([ms_tensor_x1, ms_tensor_x2], dim=0).numpy()
        except Exception as e:
            err = True
            print(f"mint.cat not supported for {dtype_ms}")

        try:
            torch_tensor_x1 = torch.tensor(input_x1, dtype=dtype_torch)
            torch_tensor_x2 = torch.tensor(input_x2, dtype=dtype_torch)
            torch_result = torch.cat([torch_tensor_x1, torch_tensor_x2], dim=0).numpy()
        except Exception as e:
            err = True
            print(f"torch.cat not supported for {dtype_torch}")

        if not err:
            assert np.allclose(ms_result, torch_result)


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
            for dim in range(-len(shape),len(shape)):
                result = is_same(shape=shape, dtype_ms=dtype_ms, dtype_torch=dtype_torch, dim=dim)
                assert result


@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_any_wrong_input(mode):
    """测试随机混乱输入，报错信息的准确性"""
    # TypeError - dim 不是int。
    try:
        input_x1 = Tensor([[0, 1], [2, 1]], dtype = ms.float32)
        input_x2 = Tensor([[0, 1], [2, 1]], dtype = ms.float32)
        output = mint.cat([input_x1, input_x2], dim='a')
        print(output)
    except Exception as e:
        print(f'TypeError - dim 不是int : {e}')
    # ValueError - tensors 是不同维度的Tensor。
    try:
        input_x1 = Tensor([[0, 1], [2, 1]], dtype = ms.float32)
        input_x2 = Tensor([[0, 1], [2, 1], [2,1]], dtype = ms.float32)
        output = mint.cat([input_x1, input_x2], dim=1)
        print(output)
    except Exception as e:
        print(f'ValueError - tensors 是不同维度的Tensor : {e}')
    # ValueError - dim 的值不在区间 [-R,R) 内。
    try:
        input_x1 = Tensor([[0, 1], [2, 1]], dtype = ms.float32)
        input_x2 = Tensor([[0, 1], [2, 1]], dtype = ms.float32)
        output = mint.cat([input_x1, input_x2], dim=2)
        print(output)
    except Exception as e:
        print(f'ValueError - dim 的值不在区间 [-R,R) 内 : {e}')
    # ValueError - 除了 dim 之外， tensors 的shape不相同。
    try:
        input_x1 = Tensor([[0, 1], [2, 1]], dtype = ms.float32)
        input_x2 = Tensor([[0, 1, 2], [2, 1, 0]], dtype = ms.float32)
        output = mint.cat([input_x1, input_x2], dim=0)
        print(output)
    except Exception as e:
        print(f'ValueError - 除了 dim 之外， tensors 的shape不相同 : {e}')
    # ValueError - tensors 为空tuple或list。
    try:
        output = mint.cat([], dim=0)
        print(output)
    except Exception as e:
        print(f'ValueError - tensors 为空tuple或list : {e}')


@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_any_forward_back(mode):
    """使用Pytorch和MindSpore, 固定输入和权重, 测试正向推理结果和反向梯度"""
    ms.set_context(mode=mode)

    def forward(x, y):
        return x * y
    
    shapes_list = generate_input_shapes()
    for ms_dtype, torch_dtype in gradient_dtype_list.items():
        for shape in shapes_list:
            for dim in range(-len(shape),len(shape)):
                weight_shape = shape.copy()
                weight_shape[dim] = weight_shape[dim] * 2
                weight = np.random.rand(*weight_shape)

                input_x1 = np.random.rand(*shape)
                input_x2 = np.random.rand(*shape)

                ms_tensor_x1 = Tensor(input_x1, dtype=ms_dtype)
                ms_tensor_x2 = Tensor(input_x2, dtype=ms_dtype)
                ms_weight = Tensor(weight, dtype=ms_dtype)
                ms_result = mint.cat([ms_tensor_x1, ms_tensor_x2], dim=dim)
                grad_fn = value_and_grad(forward)
                z_ms, gradient_ms= grad_fn(ms_result, ms_weight)

                torch_tensor_x1 = torch.tensor(input_x1, dtype=torch_dtype, requires_grad=True)
                torch_tensor_x2 = torch.tensor(input_x2, dtype=torch_dtype, requires_grad=True)
                torch_weight = torch.tensor(weight, dtype=torch_dtype, requires_grad=True)
                torch_result = torch.cat([torch_tensor_x1, torch_tensor_x2], dim=dim)
                torch_result.retain_grad()
                z_torch = forward(torch_result, torch_weight)
                loss = z_torch.sum()
                loss.backward()
                gradient_torch = torch_result.grad

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