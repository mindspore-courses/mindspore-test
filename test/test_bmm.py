import pytest
import numpy as np
import mindspore as ms
from mindspore import mint, Tensor, value_and_grad
import torch


def is_same_bmm(shape1=(2, 3, 4), shape2=(2, 4, 5), dtype_ms=ms.float32, dtype_torch=torch.float32):
    # 生成合法三维张量对
    np.random.seed(42)
    input1 = np.random.randn(*shape1).astype(np.float32)
    input2 = np.random.randn(*shape2).astype(np.float32)

    ms_tensor1 = Tensor(input1, dtype_ms)
    ms_tensor2 = Tensor(input2, dtype_ms)
    torch_tensor1 = torch.tensor(input1, dtype=dtype_torch)
    torch_tensor2 = torch.tensor(input2, dtype=dtype_torch)

    try:
        ms_result = mint.bmm(ms_tensor1, ms_tensor2).asnumpy()
    except Exception as e:
        print(f"MindSpore bmm error: {e}")
        return False

    try:
        torch_result = torch.bmm(torch_tensor1, torch_tensor2).numpy()
    except Exception as e:
        print(f"PyTorch bmm error: {e}")
        return False

    return np.allclose(ms_result, torch_result, atol=1e-4)


@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_bmm_different_dtypes(mode):
    """严格测试所有数据类型组合"""
    ms.set_context(mode=mode)

    # 完整数据类型列表
    ms_dtypes = [ms.int8, ms.int16, ms.int32, ms.int64, ms.uint8,
                 ms.uint16, ms.uint32, ms.uint64, ms.float16, ms.float32,
                 ms.float64, ms.bfloat16, ms.bool_]
    torch_dtypes = [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8,
                    torch.uint16, torch.uint32, torch.uint64, torch.float16,
                    torch.float32, torch.float64, torch.bfloat16, torch.bool]

    for i in range(len(ms_dtypes)):
        ms_dtype = ms_dtypes[i]
        torch_dtype = torch_dtypes[i]

        # 生成适配数据类型的安全数据
        np.random.seed(42)
        if ms_dtype in [ms.bool_, ms.int8, ms.uint8]:
            # 小范围整型数据 (避免溢出)
            input1 = np.random.randint(0, 2 if ms_dtype == ms.bool_ else 3, size=(2, 3, 4))
            input2 = np.random.randint(0, 2 if ms_dtype == ms.bool_ else 3, size=(2, 4, 5))
        else:
            # 浮点和较大整型使用缩放后的随机数据
            scale = 1e3 if ms_dtype in [ms.float16, ms.bfloat16] else 1e6
            input1 = (np.random.randn(2, 3, 4) * scale).astype(np.float32)
            input2 = (np.random.randn(2, 4, 5) * scale).astype(np.float32)

        # 类型转换
        ms_tensor1 = Tensor(input1, ms_dtype)
        ms_tensor2 = Tensor(input2, ms_dtype)
        torch_tensor1 = torch.tensor(input1, dtype=torch_dtype)
        torch_tensor2 = torch.tensor(input2, dtype=torch_dtype)

        # 执行状态跟踪
        ms_success, torch_success = False, False
        ms_result, torch_result = None, None

        # MindSpore执行
        try:
            ms_result = mint.bmm(ms_tensor1, ms_tensor2).asnumpy()
            ms_success = True
        except Exception as e:
            print(f"MindSpore {ms_dtype} failed: {str(e)}")

        # PyTorch执行
        try:
            torch_result = torch.bmm(torch_tensor1, torch_tensor2).numpy()
            torch_success = True
        except Exception as e:
            print(f"PyTorch {torch_dtype} failed: {str(e)}")

        # 严格断言逻辑
        assert ms_success == torch_success, \
            f"Framework mismatch: MS({ms_success}) vs Torch({torch_success}) for {ms_dtype}"

        if ms_success and torch_success:
            # 动态精度控制
            if ms_dtype in [ms.float16, ms.bfloat16]:
                atol = 1e-3
            elif ms_dtype in [ms.int8, ms.uint8, ms.bool_]:
                atol = 0  # 离散值必须严格相等
            else:
                atol = 1e-5

            # 结果对比
            same = np.allclose(ms_result, torch_result, atol=atol)
            if not same:
                print(f"Mismatch details for {ms_dtype}:")
                print("Input1:\n", input1)
                print("Input2:\n", input2)
                print("MS result:\n", ms_result)
                print("Torch result:\n", torch_result)
                print("Difference:\n", ms_result - torch_result)
            assert same, f"Numerical mismatch for {ms_dtype}"


@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_bmm_random_shapes(mode):
    """测试不同合法形状组合"""
    ms.set_context(mode=mode)
    shape_pairs = [
        ((3, 2, 4), (3, 4, 5)),
        ((5, 1, 6), (5, 6, 3)),
        ((4, 10, 20), (4, 20, 15)),
        ((1, 100, 200), (1, 200, 50)),
    ]

    for s1, s2 in shape_pairs:
        assert is_same_bmm(s1, s2), f"Shape pair {s1}, {s2} failed"


@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_bmm_invalid_inputs(mode):
    """测试非法输入异常捕获"""
    ms.set_context(mode=mode)

    # 非三维输入
    with pytest.raises(ValueError):
        mint.bmm(Tensor(np.random.randn(2, 3), ms.float32),
                     Tensor(np.random.randn(2, 3, 4), ms.float32))

    # batch大小不匹配
    with pytest.raises(ValueError):
        mint.bmm(Tensor(np.random.randn(2, 3, 4), ms.float32),
                     Tensor(np.random.randn(3, 4, 5), ms.float32))

    # 维度不匹配
    with pytest.raises(ValueError):
        mint.bmm(Tensor(np.random.randn(3, 5, 6), ms.float32),
                     Tensor(np.random.randn(3, 7, 8), ms.float32))


@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_bmm_forward_backward(mode):
    """测试可微分类型的前向和反向"""
    ms.set_context(mode=mode)

    # 过滤不可微分类型
    differentiable_types = [
        (ms.float16, torch.float16),
        (ms.float32, torch.float32),
        (ms.float64, torch.float64),
        (ms.bfloat16, torch.bfloat16)
    ]

    for ms_dtype, torch_dtype in differentiable_types:
        np.random.seed(42)
        x1_np = np.random.randn(3, 2, 4).astype(np.float32) * 0.1  # 缩小数值范围
        x2_np = np.random.randn(3, 4, 5).astype(np.float32) * 0.1

        # MindSpore设置
        ms_x1 = Tensor(x1_np, ms_dtype)
        ms_x2 = Tensor(x2_np, ms_dtype)
        ms_x1.requires_grad = True
        ms_x2.requires_grad = True

        # PyTorch设置
        torch_x1 = torch.tensor(x1_np, dtype=torch_dtype, requires_grad=True)
        torch_x2 = torch.tensor(x2_np, dtype=torch_dtype, requires_grad=True)

        # 前向计算
        def forward_ms(a, b):
            return mint.bmm(a, b).sum()  # sum用于产生标量梯度

        def forward_pt(a, b):
            return torch.bmm(a, b).sum()

        # 反向计算
        try:
            ms_grad_fn = value_and_grad(forward_ms, (0, 1))
            ms_output, (ms_grad1, ms_grad2) = ms_grad_fn(ms_x1, ms_x2)
        except Exception as e:
            pytest.fail(f"MindSpore backward failed for {ms_dtype}: {e}")

        try:
            pt_output = forward_pt(torch_x1, torch_x2)
            pt_output.backward()
        except Exception as e:
            pytest.fail(f"PyTorch backward failed for {torch_dtype}: {e}")

        # 结果比较
        assert np.allclose(ms_output.asnumpy(), pt_output.detach().numpy(), atol=1e-5), "Forward mismatch"
        assert np.allclose(ms_grad1.asnumpy(), torch_x1.grad.numpy(), atol=1e-5), "Gradient1 mismatch"
        assert np.allclose(ms_grad2.asnumpy(), torch_x2.grad.numpy(), atol=1e-5), "Gradient2 mismatch"