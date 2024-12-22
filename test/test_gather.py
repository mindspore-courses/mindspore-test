import pytest
import random
import numpy as np
import mindspore as ms
from mindspore import mint, Tensor, value_and_grad
import torch

"""
mindspore.mint.gather(input, dim, index)
    返回输入Tensor在指定 index 索引对应的元素组成的切片。

警告
    在Ascend后端，以下场景将导致不可预测的行为：
    正向执行流程中，当 index 的取值不在范围 [-input.shape[dim], input.shape[dim]) 内；
    反向执行流程中，当 index 的取值不在范围 [0, input.shape[dim]) 内。

参数：
    input (Tensor) - 待索引切片取值的原始Tensor。
    dim (int) - 指定要切片的维度索引。取值范围 [-input.rank, input.rank)。
    index (Tensor) - 指定原始Tensor中要切片的索引。数据类型必须是int32或int64。需要同时满足以下条件：
    index.rank == input.rank；
    对于 axis != dim ， index.shape[axis] <= input.shape[axis] ；
    index 的取值在有效区间 [-input.shape[dim], input.shape[dim]) ；
返回：
    Tensor，数据类型与 input 保持一致，shape与 index 保持一致。

异常：
    ValueError - input 的shape取值非法。
    ValueError - dim 取值不在有效范围 [-input.rank, input.rank)。
    ValueError - index 的值不在有效范围。
    TypeError - index 的数据类型非法。

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
    length = shape[dim]
    np_index = np.random.randint(0, length, size=shape)
    input_x = np.random.randn(*shape)

    ms_tensor_x = Tensor(input_x, dtype=dtype_ms)
    ms_index = Tensor(np_index, dtype=ms.int32)
    ms_result = mint.gather(ms_tensor_x, dim, ms_index).numpy()

    torch_tensor_x = torch.tensor(input_x, dtype=dtype_torch)
    torch_index = torch.tensor(np_index, dtype=torch.int64)
    torch_result = torch.gather(torch_tensor_x, dim, torch_index).numpy()

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
    
    input_x = np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]])
    index = np.array([[0, 1], [1, 1]])
    for i in range(len(mindspore_dtype_list)):
        dtype_ms = mindspore_dtype_list[i]
        dtype_torch = pytorch_dtype_list[i]

        err = False
        try:
            ms_tensor = Tensor(input_x, dtype=dtype_ms)
            index_ms = Tensor(index, dtype=ms.int32)
            ms_result = mint.gather(ms_tensor, 1, index_ms).numpy()
        except Exception as e:
            err = True
            print(f"mint.gather not supported for {dtype_ms}")

        try:
            torch_tensor = torch.tensor(input_x, dtype=dtype_torch)
            index_torch = torch.tensor(index, dtype=torch.int32)
            torch_result = torch.gather(torch_tensor, 1, index_torch).numpy()
        except Exception as e:
            err = True
            print(f"torch.gather not supported for {dtype_torch}")

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
            for dim in range(-len(shape),len(shape)):
                result = is_same(shape=shape, dtype_ms=dtype_ms, dtype_torch=dtype_torch, dim=dim)
                assert result


@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_any_wrong_input(mode):
    """测试随机混乱输入，报错信息的准确性"""
    ms_dtype = ms.float32
    # ValueError - input 的shape取值非法。
    try:
        input_tensor = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), dtype = ms_dtype)
        index = Tensor(np.array([[0, 0], [1, 1], [2, 1]]), dtype = mindspore.int32)
        output = mint.gather(input_tensor, 1, index)
    except Exception as e:
        print(f"ValueError - input 的shape取值非法: {e}.")
    # ValueError - dim 取值不在有效范围 [-input.rank, input.rank)。
    try:
        input_tensor = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), dtype = ms_dtype)
        index = Tensor(np.array([[0, 0], [1, 1]]), dtype = mindspore.int32)
        output = mint.gather(input_tensor, 2, index)
    except Exception as e:
        print(f"ValueError - dim 取值不在有效范围 [-input.rank, input.rank): {e}.")
    # ValueError - index 的值不在有效范围。
    try:
        input_tensor = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), dtype = ms_dtype)
        index = Tensor(np.array([[0, 0], [1, 4]]), dtype = mindspore.int32)
        output = mint.gather(input_tensor, 1, index)
        print(output)
    except Exception as e:
        print(f"ValueError - index 的值不在有效范围: {e}.")
    # TypeError - index 的数据类型非法。
    try:
        input_tensor = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), dtype = ms_dtype)
        index = Tensor(np.array([[0, 0], [1, 1]]), dtype = mindspore.float32)
        output = mint.gather(input_tensor, 1, index)
    except Exception as e:
        print(f"TypeError - index 的数据类型非法: {e}.")


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
                length = shape[dim]
                np_index = np.random.randint(0, length, size=shape)
                input_x = np.random.rand(*shape)
                weight = np.random.rand(*shape)

                ms_tensor_x1 = Tensor(input_x, dtype=ms_dtype)
                ms_index = Tensor(np_index, dtype=ms.int32)
                ms_weight = Tensor(weight, dtype=ms_dtype)
                ms_result = mint.gather(ms_tensor_x1, dim, ms_index)
                grad_fn = value_and_grad(forward)
                z_ms, gradient_ms= grad_fn(ms_result, ms_weight)

                torch_tensor_x = torch.tensor(input_x, dtype=torch_dtype, requires_grad=True)
                torch_index = torch.tensor(np_index, dtype=torch.int64)
                torch_weight = torch.tensor(weight, dtype=torch_dtype, requires_grad=True)
                torch_result = torch.gather(torch_tensor_x, dim, torch_index)
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