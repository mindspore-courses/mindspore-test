import pytest
import numpy as np
import mindspore as ms
from mindspore import mint, Tensor, value_and_grad
import torch


def is_same_baddbmm(shape_input=(2, 3, 3), shape_batch1=(2, 3, 4), shape_batch2=(2, 4, 3),
                    beta=1, alpha=1, dtype_ms=ms.float32, dtype_torch=torch.float32):
    # 生成合法三维张量
    np.random.seed(42)
    input_np = np.random.randn(*shape_input).astype(np.float32)
    batch1_np = np.random.randn(*shape_batch1).astype(np.float32)
    batch2_np = np.random.randn(*shape_batch2).astype(np.float32)

    # 类型转换
    ms_input = Tensor(input_np, dtype_ms)
    ms_batch1 = Tensor(batch1_np, dtype_ms)
    ms_batch2 = Tensor(batch2_np, dtype_ms)
    torch_input = torch.tensor(input_np, dtype=dtype_torch)
    torch_batch1 = torch.tensor(batch1_np, dtype=dtype_torch)
    torch_batch2 = torch.tensor(batch2_np, dtype=dtype_torch)

    # 执行计算
    try:
        ms_result = mint.baddbmm(ms_input, ms_batch1, ms_batch2, beta=beta, alpha=alpha).asnumpy()
    except Exception as e:
        print(f"MindSpore error: {str(e)}")
        return False

    try:
        torch_result = torch.baddbmm(torch_input, torch_batch1, torch_batch2, beta=beta, alpha=alpha).numpy()
    except Exception as e:
        print(f"PyTorch error: {str(e)}")
        return False

    # 动态设置容差
    if dtype_ms in [ms.float16, ms.bfloat16]:
        atol = 1e-3
    else:
        atol = 1e-5
    return np.allclose(ms_result, torch_result, atol=atol)


@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_baddbmm_different_dtypes(mode):
    """测试全数据类型支持情况"""
    ms.set_context(mode=mode)
    ms_dtypes = [ms.int8, ms.int16, ms.int32, ms.int64, ms.uint8, ms.uint16, ms.uint32, ms.uint64,
                 ms.float16, ms.float32, ms.float64, ms.bfloat16, ms.bool_]
    torch_dtypes = [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.uint16, torch.uint32,
                    torch.uint64, torch.float16, torch.float32, torch.float64, torch.bfloat16, torch.bool]

    for i in range(len(ms_dtypes)):
        dtype_ms = ms_dtypes[i]
        dtype_torch = torch_dtypes[i]
        success = True

        # 布尔型特殊处理
        if dtype_ms == ms.bool_:
            input_np = np.random.choice([True, False], size=(2, 3, 3))
            batch1_np = np.random.choice([True, False], size=(2, 3, 4))
            batch2_np = np.random.choice([True, False], size=(2, 4, 3))
        else:
            input_np = np.random.randn(2, 3, 3).astype(np.float32)
            batch1_np = np.random.randn(2, 3, 4).astype(np.float32)
            batch2_np = np.random.randn(2, 4, 3).astype(np.float32)

        try:
            ms_input = Tensor(input_np, dtype_ms)
            ms_batch1 = Tensor(batch1_np, dtype_ms)
            ms_batch2 = Tensor(batch2_np, dtype_ms)
            ms_result = mint.baddbmm(ms_input, ms_batch1, ms_batch2).asnumpy()
        except Exception as e:
            success = False
            print(f"MindSpore {dtype_ms} failed: {e}")

        try:
            torch_input = torch.tensor(input_np, dtype=dtype_torch)
            torch_batch1 = torch.tensor(batch1_np, dtype=dtype_torch)
            torch_batch2 = torch.tensor(batch2_np, dtype=dtype_torch)
            torch_result = torch.baddbmm(torch_input, torch_batch1, torch_batch2).numpy()
        except Exception as e:
            success = False
            print(f"PyTorch {dtype_torch} failed: {e}")

        # 严格校验框架支持一致性
        assert (ms_result is not None) == (torch_result is not None), \
            f"Framework mismatch for {dtype_ms}: MS({ms_result is not None}) vs Torch({torch_result is not None})"

        if success:
            if dtype_ms == ms.bool_:
                # 布尔型校验逻辑运算结果
                assert np.array_equal(ms_result, torch_result), "Boolean result mismatch"
            else:
                assert np.allclose(ms_result, torch_result,
                                   atol=1e-3 if dtype_ms in [ms.float16, ms.bfloat16] else 1e-5)


@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_baddbmm_parameters(mode):
    """测试不同参数组合"""
    ms.set_context(mode=mode)
    params = [
        (0, 1), (1, 0), (2, 3), (-1, 2), (0.5, 1.5)
    ]
    for beta, alpha in params:
        assert is_same_baddbmm(beta=beta, alpha=alpha), f"Failed with beta={beta}, alpha={alpha}"


@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_baddbmm_broadcast(mode):
    """测试广播机制"""
    ms.set_context(mode=mode)
    # 输入可广播的shape组合
    shape_combos = [
        ((1, 3, 3), (2, 3, 4), (2, 4, 3)),  # 广播input
        ((2, 1, 3), (2, 3, 4), (2, 4, 3)),  # 广播中间维度
        ((5, 1, 1), (5, 3, 4), (5, 4, 3))  # 多维度广播
    ]
    for shapes in shape_combos:
        assert is_same_baddbmm(*shapes), f"Broadcast failed for {shapes}"


@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_baddbmm_invalid_input(mode):
    """测试非法输入异常"""
    ms.set_context(mode=mode)
    valid_tensor = Tensor(np.ones((2, 3, 3)), ms.float32)

    # 测试非三维输入
    with pytest.raises(ValueError):
        mint.baddbmm(Tensor(np.ones((3, 3))), valid_tensor, valid_tensor)

    # 测试类型不匹配
    with pytest.raises(TypeError):
        mint.baddbmm(Tensor(np.ones((2, 3, 3)), ms.float32),
                         Tensor(np.ones((2, 3, 4)), ms.int32),
                         valid_tensor)


@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_baddbmm_gradient(mode):
    """测试梯度计算正确性"""
    ms.set_context(mode=mode)
    np.random.seed(42)

    # 生成可微分数据
    input_np = np.random.randn(2, 3, 3).astype(np.float32) * 0.1
    batch1_np = np.random.randn(2, 3, 4).astype(np.float32) * 0.1
    batch2_np = np.random.randn(2, 4, 3).astype(np.float32) * 0.1

    # MindSpore设置
    ms_input = Tensor(input_np, ms.float32)
    ms_batch1 = Tensor(batch1_np, ms.float32)
    ms_batch2 = Tensor(batch2_np, ms.float32)
    ms_input.requires_grad = True
    ms_batch1.requires_grad = True
    ms_batch2.requires_grad = True

    # PyTorch设置
    torch_input = torch.tensor(input_np, dtype=torch.float32, requires_grad=True)
    torch_batch1 = torch.tensor(batch1_np, dtype=torch.float32, requires_grad=True)
    torch_batch2 = torch.tensor(batch2_np, dtype=torch.float32, requires_grad=True)

    # 前向计算
    def forward_ms(i, b1, b2):
        return mint.baddbmm(i, b1, b2).sum()

    def forward_pt(i, b1, b2):
        return torch.baddbmm(i, b1, b2).sum()

    # 反向计算
    grad_fn = value_and_grad(forward_ms, (0, 1, 2))
    ms_output, (ms_grad_i, ms_grad_b1, ms_grad_b2) = grad_fn(ms_input, ms_batch1, ms_batch2)

    pt_output = forward_pt(torch_input, torch_batch1, torch_batch2)
    pt_output.backward()

    # 结果比较
    assert np.allclose(ms_output.asnumpy(), pt_output.detach().numpy(), atol=1e-5)
    assert np.allclose(ms_grad_i.asnumpy(), torch_input.grad.numpy(), atol=1e-5)
    assert np.allclose(ms_grad_b1.asnumpy(), torch_batch1.grad.numpy(), atol=1e-5)
    assert np.allclose(ms_grad_b2.asnumpy(), torch_batch2.grad.numpy(), atol=1e-5)