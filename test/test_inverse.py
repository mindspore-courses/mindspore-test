import numpy as np
import mindspore
from mindspore import Tensor, dtype as mstype, ops
import mindspore.mint as mint
import torch

def test_inverse_random_dtype_support():
    """
    (1a) 测试random输入不同dtype (float16, float32)
    """
    print("===== Inverse random dtype support test =====")
    for dt in [mstype.float16, mstype.float32, mstype.int32]:
        # 构造可逆矩阵
        A = np.random.randn(2,2)
        A += np.eye(2)*0.1  # 保证可逆
        if dt == mstype.int32:
            # int32 矩阵不适合做 inverse, 预期报错
            A = A.astype(np.int32)
        else:
            A = A.astype(mindspore.dtype_to_nptype(dt))

        print(f"MindSpore dtype={dt}")
        try:
            A_inv_ms = mint.inverse(Tensor(A, dt))
            print("  MS inverse shape:", A_inv_ms.shape)
        except Exception as e:
            print("  MS error:", e)

        # PyTorch
        torch_dt = torch.float32 if dt != mstype.float16 else torch.float16
        A_torch = torch.tensor(A, dtype=torch_dt)
        print(f"PyTorch dtype={torch_dt}")
        try:
            A_inv_pt = torch.inverse(A_torch)
            print("  PT inverse shape:", A_inv_pt.shape)
        except Exception as e:
            print("  PT error:", e)
        print("-----------------------------------------")


def test_inverse_fixed_dtype_output_equality():
    """
    (1b) 固定dtype=float32, 随机输入, 对比两个框架输出 (误差<1e-3)
    """
    print("===== Inverse fixed dtype output equality test =====")
    A = np.random.rand(3,3).astype(np.float32)
    A += np.eye(3)*0.1
    A_ms = Tensor(A, mstype.float32)
    A_inv_ms = mint.inverse(A_ms).asnumpy()

    A_torch = torch.tensor(A, dtype=torch.float32)
    A_inv_pt = torch.inverse(A_torch).numpy()

    diff = np.abs(A_inv_ms - A_inv_pt).max()
    print("Max diff:", diff)
    assert diff < 1e-3


def test_inverse_fixed_shape_diff_params():
    """
    (1c) inverse只有一个输入参数，这里只能测试不同形状
    """
    print("===== Inverse fixed shape diff params test =====")
    # batch of matrices
    A = np.random.randn(2,2,2).astype(np.float32)
    # 保证可逆
    for i in range(2):
        A[i] += np.eye(2)*0.1
    A_ms = Tensor(A, mstype.float32)
    A_inv_ms = mint.inverse(A_ms)
    print("MindSpore batch inverse shape:", A_inv_ms.shape)

    A_torch = torch.tensor(A, dtype=torch.float32)
    A_inv_pt = torch.inverse(A_torch)
    diff = np.abs(A_inv_ms.asnumpy() - A_inv_pt.numpy()).max()
    print("Max diff in batch inverse:", diff)


def test_inverse_error_messages():
    """
    (1d) 测试随机混乱输入, 报错信息准确性
    """
    print("===== Inverse error messages test =====")
    # 非方阵
    try:
        mint.inverse(Tensor(np.ones((2,3)), mstype.float32))
    except Exception as e:
        print("  Non-square error:", e)

    # 不可逆
    singular = np.array([[1,2],[2,4]], np.float32)
    try:
        mint.inverse(Tensor(singular, mstype.float32))
    except Exception as e:
        print("  Singular error:", e)


def test_inverse_network_forward_backward():
    """
    (2b,2c) 测试 inverse 用于函数: trace(inv(A)) => 求梯度
    """
    print("===== Inverse forward/backward test =====")
    # PyTorch
    A_pt = torch.tensor([[2.0,0.5],[0.5,1.0]], requires_grad=True)
    trace_inv_pt = torch.inverse(A_pt).trace()
    trace_inv_pt.backward()
    grad_pt = A_pt.grad.numpy()
    print("PyTorch grad:\n", grad_pt)

    # MindSpore
    A_ms = Tensor(np.array([[2.0,0.5],[0.5,1.0]], np.float32))
    A_ms.requires_grad = True
    def fn(mat):
        return ops.Trace()(mint.inverse(mat))
    grad_fn = ops.grad(fn, grad_position=0)
    grad_ms = grad_fn(A_ms).asnumpy()
    print("MindSpore grad:\n", grad_ms)
