import numpy as np
import mindspore
from mindspore import Tensor, dtype as mstype, ops
import mindspore.mint.nn.functional as F_ms
import torch
import torch.nn.functional as F_torch


def test_hardswish_random_dtype_support():
    """
    (1a) 测试不同 dtype
    """
    print("===== Hardswish random dtype support test =====")
    for dt in [mstype.float16, mstype.float32, mstype.int32]:
        x_np = np.random.uniform(-5,5,size=(3,3))
        if dt == mstype.int32:
            x_np = x_np.astype(np.int32)
        else:
            x_np = x_np.astype(mindspore.dtype_to_nptype(dt))
        try:
            out_ms = F_ms.hardswish(Tensor(x_np, dt))
            print(f"MindSpore dtype={dt}, output shape={out_ms.shape}")
        except Exception as e:
            print(f"MindSpore error with dtype={dt}:", e)

        torch_dt = torch.float16 if dt == mstype.float16 else torch.float32
        try:
            out_torch = F_torch.hardswish(torch.tensor(x_np, dtype=torch_dt))
            print(f"PyTorch dtype={torch_dt}, output shape={out_torch.shape}")
        except Exception as e:
            print(f"PyTorch error with dtype={torch_dt}:", e)
        print("--------------------------------------------")


def test_hardswish_fixed_dtype_output_equality():
    """
    (1b) 固定dtype float32，对比输出
    """
    print("===== Hardswish fixed dtype output equality test =====")
    x_np = np.random.uniform(-5,5,(4,4)).astype(np.float32)
    ms_in = Tensor(x_np, mstype.float32)
    torch_in = torch.tensor(x_np, dtype=torch.float32)

    out_ms = F_ms.hardswish(ms_in).asnumpy()
    out_torch = F_torch.hardswish(torch_in).numpy()

    diff = np.abs(out_ms - out_torch).max()
    print("Max diff:", diff)
    assert diff < 1e-3


def test_hardswish_fixed_shape_diff_params():
    """
    (1c) Hardswish无额外参数, 仅测试不同形状
    """
    print("===== Hardswish fixed shape diff params test =====")
    arr1 = Tensor(np.array([-2., -1., 0., 1., 2.], np.float32))
    out1 = F_ms.hardswish(arr1)
    print("arr1 shape:", arr1.shape, "out:", out1.asnumpy())

    arr2 = Tensor(np.array([[-3,3],[4,-5]], np.float32))
    out2 = F_ms.hardswish(arr2)
    print("arr2 shape:", arr2.shape, "out:\n", out2.asnumpy())


def test_hardswish_error_messages():
    """
    (1d) 非tensor或不支持dtype
    """
    print("===== Hardswish error messages test =====")
    try:
        F_ms.hardswish("not a tensor")
    except Exception as e:
        print("String input error:", e)

    try:
        F_ms.hardswish(Tensor([0,1,2], mstype.int64))
    except Exception as e:
        print("Int64 input error:", e)


def test_hardswish_network_forward_backward():
    """
    (2b,2c) 前向与梯度
    """
    print("===== Hardswish network forward/backward test =====")
    # PyTorch
    x_torch = torch.tensor([-4.,-2.,0.,2.,4.], requires_grad=True)
    y_torch = F_torch.hardswish(x_torch)
    y_torch.sum().backward()
    grad_torch = x_torch.grad.numpy()
    print("PyTorch grad:", grad_torch)

    # MindSpore
    x_ms = Tensor(np.array([-4.,-2.,0.,2.,4.], np.float32))
    x_ms.requires_grad = True
    def net(inp):
        return F_ms.hardswish(inp).sum()
    grad_fn = ops.grad(net, grad_position=0)
    grad_ms = grad_fn(x_ms).asnumpy()
    print("MindSpore grad:", grad_ms)
