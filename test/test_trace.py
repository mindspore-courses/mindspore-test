import numpy as np
import mindspore
from mindspore import Tensor, dtype as mstype, ops
import mindspore.mint as mint
import torch

def test_trace_random_dtype_support():
    """
    (1a) 不同 dtype
    """
    print("===== Trace random dtype support test =====")
    for dt in [mstype.float16, mstype.float32, mstype.int32, mstype.bool_]:
        # 构造2D矩阵
        if dt == mstype.bool_:
            mat = np.random.randint(0,2,size=(3,3)).astype(np.bool_)
        elif dt == mstype.int32:
            mat = np.random.randint(0,10,size=(3,3)).astype(np.int32)
        else:
            mat = np.random.randn(3,3).astype(mindspore.dtype_to_nptype(dt))
        try:
            out_ms = mint.trace(Tensor(mat, dt))
            print(f"MindSpore dtype={dt}, value=", out_ms.asnumpy(), " shape=", out_ms.shape)
        except Exception as e:
            print("MindSpore error:", e)

        # PyTorch
        if dt == mstype.float16:
            torch_dt = torch.float16
        elif dt == mstype.float32:
            torch_dt = torch.float32
        elif dt == mstype.int32:
            torch_dt = torch.int32
        elif dt == mstype.bool_:
            # PyTorch 1.9+  trace(bool tensor) => int64?
            torch_dt = torch.bool
        mat_torch = torch.tensor(mat, dtype=torch_dt)
        try:
            out_pt = torch.trace(mat_torch)
            print(f"PyTorch dtype={torch_dt}, value={out_pt.item()}")
        except Exception as e:
            print("PyTorch error:", e)
        print("----------------------------------------")


def test_trace_fixed_dtype_output_equality():
    """
    (1b) 固定 float32, 随机输入
    """
    print("===== Trace fixed dtype output equality test =====")
    mat = np.random.randn(4,4).astype(np.float32)
    ms_val = mint.trace(Tensor(mat, mstype.float32)).asnumpy()
    pt_val = torch.trace(torch.tensor(mat, dtype=torch.float32)).item()
    diff = abs(ms_val - pt_val)
    print("diff:", diff)
    assert diff < 1e-3


def test_trace_fixed_shape_diff_params():
    """
    (1c) Trace无多余参数, 只测试不同形状
    """
    print("===== Trace fixed shape diff params test =====")
    mat1 = Tensor(np.arange(9).reshape(3,3), mstype.int32)
    val1 = mint.trace(mat1)
    print("trace of 3x3 int:", val1.asnumpy())

    mat2 = Tensor(np.array([[True,False],[False,True]]), mstype.bool_)
    val2 = mint.trace(mat2)
    print("trace of bool:", val2.asnumpy())

    # 传入非2D => error
    try:
        mint.trace(Tensor(np.ones((2,2,2)), mstype.float32))
    except Exception as e:
        print("non-2D error:", e)


def test_trace_error_messages():
    """
    (1d) 错误输入测试
    """
    print("===== Trace error messages test =====")
    # 1D tensor
    try:
        mint.trace(Tensor([1,2,3], mstype.int32))
    except Exception as e:
        print("1D error:", e)


def test_trace_network_forward_backward():
    """
    (2b,2c) 对输入的梯度测试
    """
    print("===== Trace forward/backward test =====")
    # PyTorch
    x_pt = torch.tensor([[1.,2.],[3.,4.]], requires_grad=True)
    val_pt = torch.trace(x_pt)
    val_pt.backward()
    grad_pt = x_pt.grad.numpy()
    print("PyTorch grad:\n", grad_pt)

    # MindSpore
    x_ms = Tensor(np.array([[1.,2.],[3.,4.]], np.float32))
    x_ms.requires_grad = True
    def fn(mat):
        return mint.trace(mat)
    grad_fn = ops.grad(fn, grad_position=0)
    grad_ms = grad_fn(x_ms).asnumpy()
    print("MindSpore grad:\n", grad_ms)

