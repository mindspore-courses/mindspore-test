import numpy as np
import mindspore
from mindspore import Tensor, dtype as mstype, ops
import mindspore.mint as mint
import torch

def test_matmul_random_dtype_support():
    """
    (1a) 测试不同 dtype
    """
    print("===== MatMul random dtype support test =====")
    for dt in [mstype.float16, mstype.float32, mstype.int32]:
        A = np.random.randn(2,3)
        B = np.random.randn(3,4)
        if dt == mstype.int32:
            A = A.astype(np.int32)
            B = B.astype(np.int32)
        else:
            A = A.astype(mindspore.dtype_to_nptype(dt))
            B = B.astype(mindspore.dtype_to_nptype(dt))

        try:
            out_ms = mint.matmul(Tensor(A, dt), Tensor(B, dt))
            print(f"MindSpore dtype={dt}, shape={out_ms.shape}")
        except Exception as e:
            print("MindSpore error:", e)

        torch_dt = torch.float16 if dt == mstype.float16 else torch.float32
        try:
            out_pt = torch.matmul(torch.tensor(A, dtype=torch_dt), torch.tensor(B, dtype=torch_dt))
            print(f"PyTorch dtype={torch_dt}, shape={out_pt.shape}")
        except Exception as e:
            print("PyTorch error:", e)
        print("-------------------------------------")


def test_matmul_fixed_dtype_output_equality():
    """
    (1b) 固定 dtype=float32，随机输入，对比输出
    """
    print("===== MatMul fixed dtype output equality test =====")
    A = np.random.randn(2,3).astype(np.float32)
    B = np.random.randn(3,4).astype(np.float32)

    out_ms = mint.matmul(Tensor(A, mstype.float32), Tensor(B, mstype.float32)).asnumpy()
    out_pt = torch.matmul(torch.tensor(A), torch.tensor(B)).numpy()

    diff = np.abs(out_ms - out_pt).max()
    print("Max diff:", diff)
    assert diff < 1e-3


def test_matmul_fixed_shape_diff_params():
    """
    (1c) MatMul 只有2个输入，不同形状/维度
    """
    print("===== MatMul fixed shape diff params test =====")
    # 1D @ 1D => scalar
    a = Tensor(np.array([1.,2.,3.], np.float32))
    b = Tensor(np.array([4.,5.,6.], np.float32))
    out = mint.matmul(a, b)
    print("1D@1D => shape:", out.shape, "value:", out.asnumpy())

    # 2D @ 1D => 1D
    A = Tensor(np.ones((2,3), np.float32))
    b = Tensor(np.ones((3,), np.float32))
    out2 = mint.matmul(A, b)
    print("2D@1D => shape:", out2.shape, "value:", out2.asnumpy())


def test_matmul_error_messages():
    """
    (1d) 尺寸不匹配, 非法输入
    """
    print("===== MatMul error messages test =====")
    # mismatch shape
    try:
        mint.matmul(Tensor(np.ones((2,3)), mstype.float32), Tensor(np.ones((2,4)), mstype.float32))
    except Exception as e:
        print("shape mismatch error:", e)

    # non-tensor
    try:
        mint.matmul([1,2], [3,4])
    except Exception as e:
        print("non-tensor error:", e)


def test_matmul_network_forward_backward():
    """
    (2b, 2c) 前向和反向梯度
    """
    print("===== MatMul forward/backward test =====")
    # PyTorch
    X = torch.tensor([[1.,2.],[3.,4.]], requires_grad=True)
    Y = torch.tensor([[5.,6.],[7.,8.]], requires_grad=True)
    Z = torch.matmul(X, Y)
    loss = Z.sum()
    loss.backward()
    gradX_pt = X.grad.numpy()
    gradY_pt = Y.grad.numpy()

    # MindSpore
    X_ms = Tensor(np.array([[1.,2.],[3.,4.]], np.float32))
    Y_ms = Tensor(np.array([[5.,6.],[7.,8.]], np.float32))
    X_ms.requires_grad = True
    Y_ms.requires_grad = True

    def forward_fn(x, y):
        return mint.matmul(x,y).sum()

    grad_fn = ops.GradOperation(get_all=True)(forward_fn)
    gradX_ms, gradY_ms = grad_fn(X_ms, Y_ms)
    gradX_ms = gradX_ms.asnumpy()
    gradY_ms = gradY_ms.asnumpy()

    print("PyTorch gradX:\n", gradX_pt, "\nMindSpore gradX:\n", gradX_ms)
    print("PyTorch gradY:\n", gradY_pt, "\nMindSpore gradY:\n", gradY_ms)
