import numpy as np
import mindspore
from mindspore import Tensor, dtype as mstype, ops
import mindspore.mint.nn.functional as F_ms
import torch
import torch.nn.functional as F_torch


def test_hardsigmoid_random_dtype_support():
    """
    (1a) 测试随机输入、不同 dtype 支持度
    """
    print("===== Hardsigmoid random dtype support test =====")
    for dt in [mstype.float16, mstype.float32, mstype.int32]:
        x_np = np.random.randn(3,3)
        if dt == mstype.int32:
            x_np = x_np.astype(np.int32)
        else:
            x_np = x_np.astype(mindspore.dtype_to_nptype(dt))

        print(f"Testing MindSpore Hardsigmoid with dtype={dt}")
        try:
            out_ms = F_ms.hardsigmoid(Tensor(x_np, dt))
            print("  MindSpore output shape:", out_ms.shape)
        except Exception as e:
            print("  MindSpore error:", e)

        # PyTorch
        torch_dt = torch.float16 if dt == mstype.float16 else torch.float32
        x_torch = torch.tensor(x_np, dtype=torch_dt)
        print(f"Testing PyTorch Hardsigmoid with dtype={torch_dt}")
        try:
            out_torch = F_torch.hardsigmoid(x_torch)
            print("  PyTorch output shape:", out_torch.shape)
        except Exception as e:
            print("  PyTorch error:", e)
        print("------------------------------------------------")


def test_hardsigmoid_fixed_dtype_output_equality():
    """
    (1b) 固定dtype=float32，随机输入，对比两个框架输出（误差<1e-3）
    """
    print("===== Hardsigmoid fixed dtype output equality test =====")
    x_np = np.random.uniform(-5,5,size=(4,4)).astype(np.float32)
    ms_in = Tensor(x_np, mstype.float32)
    torch_in = torch.tensor(x_np, dtype=torch.float32)

    out_ms = F_ms.hardsigmoid(ms_in).asnumpy()
    out_torch = F_torch.hardsigmoid(torch_in).numpy()

    diff = np.abs(out_ms - out_torch).max()
    print("Max diff:", diff)
    assert diff < 1e-3, f"Hardsigmoid output mismatch, diff={diff}"


def test_hardsigmoid_fixed_shape_diff_params():
    """
    (1c) Hardsigmoid 只有输入，没有其他参数，可以测试形状不同
    """
    print("===== Hardsigmoid fixed shape diff params test =====")
    # 不同形状
    arr1 = Tensor(np.array([-4.0,0.0,4.0], np.float32))
    arr2 = Tensor(np.array([[-2.0,0.0],[2.0,4.0]], np.float32))

    out1 = F_ms.hardsigmoid(arr1)
    out2 = F_ms.hardsigmoid(arr2)

    print("arr1 hardsigmoid:", out1.asnumpy())
    print("arr2 hardsigmoid:", out2.asnumpy())

    # 传入非 tensor/非支持dtype
    try:
        F_ms.hardsigmoid("not a tensor")
    except Exception as e:
        print("Error with str input:", e)


def test_hardsigmoid_error_messages():
    """
    (1d) 测试随机无效输入，报错信息
    """
    print("===== Hardsigmoid error messages test =====")
    try:
        F_ms.hardsigmoid(Tensor(np.ones((2,2), np.int64)))  # MindSpore不一定支持int64
    except Exception as e:
        print("int64 input error:", e)


def test_hardsigmoid_network_forward_backward():
    """
    (2b,2c) 使用Hardsigmoid前向和梯度对比
    """
    print("===== Hardsigmoid forward/backward test =====")
    x_np = np.array([-4., -2., 0., 2., 4.], np.float32)

    # PyTorch
    x_pt = torch.tensor(x_np, requires_grad=True)
    y_pt = F_torch.hardsigmoid(x_pt)
    y_pt.sum().backward()
    grad_pt = x_pt.grad.numpy()

    # MindSpore
    x_ms = Tensor(x_np, mstype.float32)
    x_ms.requires_grad = True
    def ms_forward(x):
        return F_ms.hardsigmoid(x).sum()
    grad_fn = ops.grad(ms_forward, grad_position=0)
    grad_ms = grad_fn(x_ms).asnumpy()

    print("PyTorch grad:", grad_pt)
    print("MindSpore grad:", grad_ms)


