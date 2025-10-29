import pytest
import numpy as np
import mindspore as ms
from mindspore import Tensor, value_and_grad, mint
import torch


def create_tensors(
    input_data, src_data, ms_dtype, torch_dtype, index=None, requires_grad=False
):
    ms_tensor = Tensor(input_data, ms_dtype)
    torch_tensor = torch.tensor(
        input_data, dtype=torch_dtype, requires_grad=requires_grad
    )
    ms_src = Tensor(src_data, ms_dtype)
    torch_src = torch.tensor(src_data, dtype=torch_dtype)
    if index is not None:
        ms_index = Tensor(index, ms.int64)
        torch_index = torch.tensor(index, dtype=torch.long)
    else:
        ms_index = None
        torch_index = None
    return ms_tensor, torch_tensor, ms_src, torch_src, ms_index, torch_index


def perform_scatter_add(
    ms_tensor, torch_tensor, ms_src, torch_src, dim, ms_index, torch_index
):
    ms_result = mint.scatter_add(ms_tensor, dim, ms_index, ms_src)
    torch_result = torch.scatter_add(torch_tensor, dim, torch_index, torch_src)
    return ms_result, torch_result


def compare_results(ms_result, torch_result, atol=1e-3):
    assert np.allclose(ms_result.asnumpy(), torch_result.numpy(), atol=atol)


@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize(
    "dtype_pair",
    [
        (ms.int8, torch.int8),
        (ms.int16, torch.int16),
        (ms.int32, torch.int32),
        (ms.int64, torch.int64),
        (ms.uint8, torch.uint8),
        # (ms.uint16, torch.uint16),
        # (ms.uint32, torch.uint32),
        # (ms.uint64, torch.uint64),
        (ms.float16, torch.float16),
        (ms.float32, torch.float32),
        (ms.float64, torch.float64),
        # (ms.complex64, torch.complex64),
        # (ms.complex128, torch.complex128),
    ],
)
def test_scatter_add_different_dtypes(mode, dtype_pair):
    """测试 random 输入不同 dtype，使用 scatter_add，对比两个框架的支持度"""
    ms.set_context(mode=mode)

    ms_dtype, torch_dtype = dtype_pair
    input_data = np.zeros((5, 5))
    src_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    index = np.array([[0, 0, 0], [2, 2, 2], [4, 4, 4]])

    ms_tensor, torch_tensor, ms_src, torch_src, ms_index, torch_index = create_tensors(
        input_data, src_data, ms_dtype, torch_dtype, index=index
    )
    try:
        ms_result, torch_result = perform_scatter_add(
            ms_tensor,
            torch_tensor,
            ms_src,
            torch_src,
            dim=1,
            ms_index=ms_index,
            torch_index=torch_index,
        )
        compare_results(ms_result, torch_result)
    except Exception as e:
        print(f"Error with dtype {ms_dtype} and torch dtype {torch_dtype}: {e}")
        raise


@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize(
    "input_data, src_data",
    [
        (np.random.randn(5), np.random.randn(5)),
        (np.random.randn(3, 4), np.random.randn(3, 4)),
        (np.random.randn(2, 3, 4), np.random.randn(2, 3, 4)),
        (np.random.randn(2, 3, 4, 5), np.random.randn(2, 3, 4, 5)),
        (np.random.randn(2, 3, 4, 5, 6), np.random.randn(2, 3, 4, 5, 6)),
        (np.random.randn(2, 3, 4, 5, 6, 7), np.random.randn(2, 3, 4, 5, 6, 7)),
        (np.random.randn(2, 3, 4, 5, 6, 7, 8), np.random.randn(2, 3, 4, 5, 6, 7, 8)),
        (
            np.random.randn(2, 3, 4, 5, 6, 7, 8, 9),
            np.random.randn(2, 3, 4, 5, 6, 7, 8, 9),
        ),
    ],
)
def test_scatter_add_random_input_fixed_dtype(mode, input_data, src_data):
    """测试固定数据类型下的随机输入值，使用 scatter_add，对比输出误差"""
    ms.set_context(mode=mode)

    index = np.random.randint(0, input_data.shape[0], size=input_data.shape)

    ms_tensor, torch_tensor, ms_src, torch_src, ms_index, torch_index = create_tensors(
        input_data, src_data, ms.float32, torch.float32, index=index
    )

    try:
        ms_result, torch_result = perform_scatter_add(
            ms_tensor,
            torch_tensor,
            ms_src,
            torch_src,
            dim=0,
            ms_index=ms_index,
            torch_index=torch_index,
        )
        compare_results(ms_result, torch_result)
    except Exception as e:
        raise e


@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize(
    "dtype_pair",
    [
        (ms.int8, torch.int8),
        (ms.int16, torch.int16),
        (ms.int32, torch.int32),
        (ms.int64, torch.int64),
        (ms.uint8, torch.uint8),
        # (ms.uint16, torch.uint16),
        # (ms.uint32, torch.uint32),
        # (ms.uint64, torch.uint64),
        (ms.float16, torch.float16),
        (ms.float32, torch.float32),
        (ms.float64, torch.float64),
        # (ms.complex64, torch.complex64),
        # (ms.complex128, torch.complex128),
    ],
)
@pytest.mark.parametrize("dim", [0, 1])
def test_scatter_add_different_para(mode, dtype_pair, dim):
    """测试固定 shape，固定输入值，使用 scatter_add，不同输入参数（dim），对比两个框架的支持度"""
    ms.set_context(mode=mode)

    ms_dtype, torch_dtype = dtype_pair
    input_data = np.zeros((5, 5))
    src_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    index = np.array([[0, 0, 0], [2, 2, 2], [4, 4, 4]])

    ms_tensor, torch_tensor, ms_src, torch_src, ms_index, torch_index = create_tensors(
        input_data, src_data, ms_dtype, torch_dtype, index=index
    )

    try:
        ms_result, torch_result = perform_scatter_add(
            ms_tensor,
            torch_tensor,
            ms_src,
            torch_src,
            dim,
            ms_index=ms_index,
            torch_index=torch_index,
        )
        compare_results(ms_result, torch_result)
    except Exception as e:
        raise e


@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize(
    "input_data, src_data, index",
    [
        (
            np.random.randn(3, 4),
            np.random.randn(5, 4),
            [0, 1, 2, 3, 4],
        ),  # index 数量超过 input_data 的维度
        (
            np.random.randn(3, 4),
            np.random.randn(4, 4),
            [0, 0, 1, 1],
        ),  # index 大小匹配，但内容不合法（重复索引）
        (np.random.randn(3, 4), np.random.randn(0, 4), []),  # index 为空
        (
            np.random.randn(3, 4),
            np.random.randn(3, 4),
            [[0, 1], [2, 3]],
        ),  # index 形状不匹配
        (
            np.random.randn(3, 4),
            np.random.randn(3, 4),
            "invalid_index",
        ),  # index 非法类型（字符串）
        (
            np.random.randn(3, 4),
            np.random.randn(3, 4),
            [[0], [1], [2]],
        ),  # index 形状不匹配
    ],
)
def test_scatter_add_wrong_input(mode, input_data, src_data, index):
    """测试随机混乱输入，使用 scatter_add，报错信息的准确性"""
    ms.set_context(mode=mode)

    try:
        ms_tensor, torch_tensor, ms_src, torch_src, ms_index, torch_index = (
            create_tensors(input_data, src_data, ms.float32, torch.float32, index=index)
        )
        ms_result, torch_result = perform_scatter_add(
            ms_tensor,
            torch_tensor,
            ms_src,
            torch_src,
            dim=0,
            ms_index=ms_index,
            torch_index=torch_index,
        )
    except Exception as e:
        print(e)


@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_scatter_add_forward_back(mode):
    """测试前向和反向传播，使用 scatter_add，对比梯度"""
    ms.set_context(mode=mode)

    def forward_ms(x, src, idx):
        return mint.scatter_add(x, 0, idx, src).sum()

    def forward_torch(x, src, idx):
        return torch.scatter_add(x, 0, idx, src).sum()

    try:
        input_data = np.zeros((5, 5))
        src_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        index = np.array([[0, 0, 0], [2, 2, 2], [4, 4, 4]])
        ms_tensor, torch_tensor, ms_src, torch_src, ms_index, torch_index = (
            create_tensors(
                input_data,
                src_data,
                ms.float32,
                torch.float32,
                index=index,
                requires_grad=True,
            )
        )

        grad_fn_ms = value_and_grad(lambda x: forward_ms(x, ms_src, ms_index))
        output_ms, gradient_ms = grad_fn_ms(ms_tensor)

        output_torch = forward_torch(torch_tensor, torch_src, torch_index)
        output_torch.backward()

        compare_results(output_ms, output_torch.detach())
        compare_results(gradient_ms, torch_tensor.grad)
    except Exception as e:
        raise e
