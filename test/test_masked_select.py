import pytest
import numpy as np
import mindspore as ms
from mindspore import Tensor, value_and_grad, mint
import torch


def create_tensors(input_data, ms_dtype, torch_dtype, mask=None, requires_grad=False):
    ms_tensor = Tensor(input_data, ms_dtype)
    torch_tensor = torch.tensor(
        input_data, dtype=torch_dtype, requires_grad=requires_grad
    )
    if mask is not None:
        mask = np.array(mask, dtype=bool)
        ms_mask = Tensor(mask, ms.bool_)
        torch_mask = torch.tensor(mask, dtype=torch.bool)
    else:
        ms_mask = None
        torch_mask = None
    return ms_tensor, torch_tensor, ms_mask, torch_mask


def perform_masked_select(ms_tensor, torch_tensor, ms_mask, torch_mask):
    ms_result = mint.masked_select(ms_tensor, ms_mask)
    torch_result = torch.masked_select(torch_tensor, torch_mask)
    return ms_result, torch_result


def compare_results(ms_result, torch_result, atol=1e-3):
    assert np.allclose(ms_result.asnumpy(), torch_result.numpy(), atol=atol)


@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize(
    "dtype_pair",
    [
        (ms.int32, torch.int32),
        (ms.int64, torch.int64),
        (ms.uint8, torch.uint8),
        (ms.float16, torch.float16),
        (ms.float32, torch.float32),
        (ms.int8, torch.int8),
        (ms.int16, torch.int16),
        (ms.float64, torch.float64),
    ],
)
def test_masked_select_different_dtypes(mode, dtype_pair):
    """测试random输入不同dtype，对比两个框架的支持度"""
    ms.set_context(mode=mode)

    ms_dtype, torch_dtype = dtype_pair
    input_data = [[1, 6, 2, 4], [7, 3, 8, 2], [2, 9, 11, 5]]
    mask = [
        [True, True, True, True],
        [False, False, False, False],
        [True, True, True, True],
    ]

    ms_tensor, torch_tensor, ms_mask, torch_mask = create_tensors(
        input_data, ms_dtype, torch_dtype, mask=mask
    )
    try:
        ms_result, torch_result = perform_masked_select(
            ms_tensor, torch_tensor, ms_mask, torch_mask
        )
        compare_results(ms_result, torch_result)
    except Exception as e:
        print(f"Error with dtype {ms_dtype} and torch dtype {torch_dtype}: {e}")
        raise


@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize(
    "dtype_pair",
    [
        (ms.int32, torch.int32),
        (ms.int64, torch.int64),
        (ms.uint8, torch.uint8),
        (ms.float16, torch.float16),
        (ms.float32, torch.float32),
        (ms.int8, torch.int8),
        (ms.int16, torch.int16),
        (ms.float64, torch.float64),
    ],
)
def test_masked_select_different_dtypes_fixed_input(mode, dtype_pair):
    """测试固定shape，固定输入值，不同dtype，对比两个框架的支持度"""
    ms.set_context(mode=mode)

    ms_dtype, torch_dtype = dtype_pair
    input_data = [[1, 6, 2, 4], [7, 3, 8, 2], [2, 9, 11, 5]]
    mask = [
        [True, True, True, True],
        [False, False, False, False],
        [True, True, True, True],
    ]

    ms_tensor, torch_tensor, ms_mask, torch_mask = create_tensors(
        input_data, ms_dtype, torch_dtype, mask=mask
    )
    try:
        ms_result, torch_result = perform_masked_select(
            ms_tensor, torch_tensor, ms_mask, torch_mask
        )
        compare_results(ms_result, torch_result)
    except Exception as e:
        raise e


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
def test_masked_select_random_input_fixed_dtype(mode, input_data):
    """测试固定数据类型下的随机输入值，对比输出误差"""
    ms.set_context(mode=mode)

    # Generate a random mask with approximately half True values
    mask = np.random.choice(a=[False, True], size=input_data.shape)

    ms_tensor, torch_tensor, ms_mask, torch_mask = create_tensors(
        input_data, ms.float32, torch.float32, mask=mask
    )

    try:
        ms_result, torch_result = perform_masked_select(
            ms_tensor, torch_tensor, ms_mask, torch_mask
        )
        compare_results(ms_result, torch_result)
    except Exception as e:
        raise e


@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize(
    "dtype_pair",
    [
        (ms.int32, torch.int32),
        (ms.int64, torch.int64),
        (ms.uint8, torch.uint8),
        (ms.float16, torch.float16),
        (ms.float32, torch.float32),
        (ms.int8, torch.int8),
        (ms.int16, torch.int16),
        (ms.float64, torch.float64),
    ],
)
@pytest.mark.parametrize(
    "mask",
    [
        # Select specific elements
        [
            [True, True, True, True],
            [False, False, False, False],
            [True, True, True, True],
        ],
        # Select diagonal elements
        [
            [True, False, False, False],
            [False, True, False, False],
            [False, False, True, False],
        ],
        # Random mask
        [
            [True, False, True, False],
            [False, True, False, True],
            [True, False, True, False],
        ],
        # All True
        [[True, True, True, True], [True, True, True, True], [True, True, True, True]],
        # All False
        [
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
        ],
    ],
)
def test_masked_select_different_para(mode, dtype_pair, mask):
    """测试固定shape，固定输入值，不同输入参数（string\bool等类型），两个框架的支持度"""
    ms.set_context(mode=mode)

    ms_dtype, torch_dtype = dtype_pair
    input_data = [[1, 6, 2, 4], [7, 3, 8, 2], [2, 9, 11, 5]]

    ms_tensor, torch_tensor, ms_mask, torch_mask = create_tensors(
        input_data, ms_dtype, torch_dtype, mask=mask
    )

    try:
        ms_result, torch_result = perform_masked_select(
            ms_tensor, torch_tensor, ms_mask, torch_mask
        )
        compare_results(ms_result, torch_result)
    except Exception as e:
        raise e


@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize(
    "input_data, mask",
    [
        (
            np.random.randn(3, 4),
            np.random.choice(a=[False, True], size=(3, 4)),
        ),  # Valid mask
        (
            np.random.randn(3, 4),
            np.random.randint(0, 2, size=(4, 3)).astype(bool),
        ),  # Mask shape mismatch
        (
            np.random.randn(3, 4),
            np.random.randint(0, 2, size=0).astype(bool),
        ),  # Empty mask
        (np.random.randn(3, 4), "invalid_mask"),  # Invalid mask type
        (
            np.random.randn(3, 4),
            np.random.randint(0, 2, size=(3, 1)).astype(bool),
        ),  # Mask shape mismatch
    ],
)
def test_masked_select_wrong_input(mode, input_data, mask):
    """测试随机混乱输入，报错信息的准确性"""
    ms.set_context(mode=mode)

    try:
        ms_tensor, torch_tensor, ms_mask, torch_mask = create_tensors(
            input_data, ms.float32, torch.float32, mask=mask
        )
        ms_result, torch_result = perform_masked_select(
            ms_tensor, torch_tensor, ms_mask, torch_mask
        )
    except Exception as e:
        print(e)


@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_masked_select_forward_back(mode):
    """测试前向和反向传播，对比梯度"""
    ms.set_context(mode=mode)

    def forward_ms(x, mask):
        return mint.masked_select(x, mask).sum()

    def forward_torch(x, mask):
        return torch.masked_select(x, mask).sum()

    try:
        input_data = [[1.0, 6.0, 2.0, 4.0], [7.0, 3.0, 8.0, 2.0], [2.0, 9.0, 11.0, 5.0]]
        mask = [
            [True, False, True, False],
            [False, True, False, True],
            [True, False, True, False],
        ]
        ms_tensor, torch_tensor, ms_mask, torch_mask = create_tensors(
            input_data, ms.float32, torch.float32, mask=mask, requires_grad=True
        )

        grad_fn_ms = value_and_grad(lambda x: forward_ms(x, ms_mask))
        output_ms, gradient_ms = grad_fn_ms(ms_tensor)

        output_torch = forward_torch(torch_tensor, torch_mask)
        output_torch.backward()

        compare_results(output_ms, output_torch.detach())
        compare_results(gradient_ms, torch_tensor.grad)
    except Exception as e:
        raise e
