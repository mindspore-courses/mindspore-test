import pytest
import numpy as np
import mindspore as ms
from mindspore import Tensor, value_and_grad, mint
import torch


def create_tensors(
    input_data, ms_dtype, torch_dtype, permute_axes=None, requires_grad=False
):
    ms_tensor = Tensor(input_data, ms_dtype)
    torch_tensor = torch.tensor(
        input_data, dtype=torch_dtype, requires_grad=requires_grad
    )
    if permute_axes is not None:
        ms_permute_axes = tuple(permute_axes)
        torch_permute_axes = tuple(permute_axes)
    else:
        ms_permute_axes = None
        torch_permute_axes = None
    return ms_tensor, torch_tensor, ms_permute_axes, torch_permute_axes


def perform_permute(ms_tensor, torch_tensor, permute_axes):
    if permute_axes is not None:
        ms_result = mint.permute(ms_tensor, permute_axes)
        torch_result = torch_tensor.permute(permute_axes)
    else:
        ms_result = ms_tensor
        torch_result = torch_tensor
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
        (ms.uint16, torch.uint16),
        (ms.uint32, torch.uint32),
        (ms.uint64, torch.uint64),
        (ms.float16, torch.float16),
        (ms.float32, torch.float32),
        (ms.float64, torch.float64),
        (ms.complex64, torch.complex64),
        (ms.complex128, torch.complex128),
    ],
)
def test_permute_different_dtypes(mode, dtype_pair):
    """测试random输入不同dtype，对比两个框架的支持度"""
    ms.set_context(mode=mode)

    ms_dtype, torch_dtype = dtype_pair
    input_data = [[1, 6, 2, 4], [7, 3, 8, 2], [2, 9, 11, 5]]
    permute_axes = [1, 0]  # 示例置换

    ms_tensor, torch_tensor, ms_permute_axes, torch_permute_axes = create_tensors(
        input_data, ms_dtype, torch_dtype, permute_axes=permute_axes
    )
    try:
        ms_result, torch_result = perform_permute(
            ms_tensor, torch_tensor, permute_axes=ms_permute_axes
        )
        compare_results(ms_result, torch_result)
    except Exception as e:
        print(f"Error with dtype {ms_dtype} and torch dtype {torch_dtype}: {e}")
        raise


@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize(
    "input_data",
    [
        np.random.randn(5),
        np.random.randn(3, 4),
        np.random.randn(2, 3, 4),
        np.random.randn(2, 3, 4, 5),
        np.random.randn(2, 3, 4, 5, 6),
        np.random.randn(2, 3, 4, 5, 6, 7),
        np.random.randn(2, 3, 4, 5, 6, 7, 8),
        np.random.randn(2, 3, 4, 5, 6, 7, 8, 9),
    ],
)
def test_permute_random_input_fixed_dtype(mode, input_data):
    """测试固定数据类型下的随机输入值，对比输出误差"""
    ms.set_context(mode=mode)

    # 根据维度生成随机置换
    ndim = input_data.ndim
    permute_axes = np.random.permutation(ndim).tolist()

    ms_tensor, torch_tensor, ms_permute_axes, torch_permute_axes = create_tensors(
        input_data, ms.float32, torch.float32, permute_axes=permute_axes
    )

    try:
        ms_result, torch_result = perform_permute(
            ms_tensor, torch_tensor, permute_axes=ms_permute_axes
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
        (ms.uint16, torch.uint16),
        (ms.uint32, torch.uint32),
        (ms.uint64, torch.uint64),
        (ms.float16, torch.float16),
        (ms.float32, torch.float32),
        (ms.float64, torch.float64),
        (ms.complex64, torch.complex64),
        (ms.complex128, torch.complex128),
    ],
)
@pytest.mark.parametrize("permute_axes", [[0, 1], [1, 0]])
def test_permute_different_para(mode, dtype_pair, permute_axes):
    """测试固定shape，固定输入值，不同输入参数（string\bool等类型），两个框架的支持度"""
    ms.set_context(mode=mode)

    ms_dtype, torch_dtype = dtype_pair
    input_data = [[1, 6, 2, 4], [7, 3, 8, 2], [2, 9, 11, 5]]

    ms_tensor, torch_tensor, ms_permute_axes, torch_permute_axes = create_tensors(
        input_data, ms_dtype, torch_dtype, permute_axes=permute_axes
    )

    try:
        ms_result, torch_result = perform_permute(
            ms_tensor, torch_tensor, permute_axes=ms_permute_axes
        )
        compare_results(ms_result, torch_result)
    except Exception as e:
        raise e


@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize(
    "input_data, permute_axes",
    [
        (np.random.randn(3, 4), [1, 2]),  # 置换轴超过输入维度
        (np.random.randn(3, 4), [0, 0]),  # 置换轴有重复
        (np.random.randn(3, 4), []),  # 置换轴为空
        (np.random.randn(3, 4), [0, 1, 2]),  # 置换轴数量不匹配
        (np.random.randn(3, 4), "invalid_axes"),  # 置换轴类型非法（字符串）
        (np.random.randn(3, 4), [0, 1, 0]),  # 置换轴有重复
    ],
)
def test_permute_wrong_input(mode, input_data, permute_axes):
    """测试随机混乱输入，报错信息的准确性"""
    ms.set_context(mode=mode)

    try:
        ms_tensor, torch_tensor, ms_permute_axes, torch_permute_axes = create_tensors(
            input_data, ms.float32, torch.float32, permute_axes=permute_axes
        )
        ms_result, torch_result = perform_permute(
            ms_tensor, torch_tensor, permute_axes=ms_permute_axes
        )
    except Exception as e:
        print(e)


@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_permute_forward_back(mode):
    """测试前向和反向传播，对比梯度"""
    ms.set_context(mode=mode)

    def forward_ms(x, permute_axes):
        return mint.permute(x, permute_axes).sum()

    def forward_torch(x, permute_axes):
        return x.permute(permute_axes).sum()

    try:
        input_data = [[1, 6, 2, 4], [7, 3, 8, 2], [2, 9, 11, 5]]
        permute_axes = [1, 0]
        ms_tensor, torch_tensor, ms_permute_axes, torch_permute_axes = create_tensors(
            input_data,
            ms.float32,
            torch.float32,
            permute_axes=permute_axes,
            requires_grad=True,
        )
        grad_fn_ms = value_and_grad(lambda x: forward_ms(x, ms_permute_axes))
        output_ms, gradient_ms = grad_fn_ms(ms_tensor)

        output_torch = forward_torch(torch_tensor, torch_permute_axes)
        output_torch.backward()
        compare_results(output_ms, output_torch.detach())
        compare_results(gradient_ms, torch_tensor.grad)
    except Exception as e:
        raise e
