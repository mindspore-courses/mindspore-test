import numpy as np
import mindspore
from mindspore import Tensor, dtype as mstype, ops
import mindspore.mint.nn.functional as F_ms
import torch
import torch.nn.functional as F_torch

def test_hardshrink_random_dtype_support():
    """
    (1a) 测试随机输入、不同 dtype 支持度
    """
    print("===== Hardshrink random dtype support test =====")
    dtypes_to_test = [mstype.float16, mstype.float32, mstype.int32]
    for dt in dtypes_to_test:
        x_np = np.random.randn(3, 3).astype(mindspore.dtype_to_nptype(dt) if dt != mstype.int32 else np.int32)
        print(f"Testing MindSpore Hardshrink with dtype={dt}")
        try:
            out_ms = F_ms.hardshrink(Tensor(x_np, dt), lambd=0.5)
            print("  MindSpore output shape:", out_ms.shape)
        except Exception as e:
            print("  MindSpore error:", e)

        # 对应 PyTorch
        torch_dt = torch.float16 if dt == mstype.float16 else torch.float32
        # int32在torch里对应 int32，但F.hardshrink需要浮点 => 这里可能会报错或自动提升
        x_torch = torch.tensor(x_np, dtype=torch_dt)
        print(f"Testing PyTorch Hardshrink with dtype={torch_dt}")
        try:
            out_torch = F_torch.hardshrink(x_torch, lambd=0.5)
            print("  PyTorch output shape:", out_torch.shape)
        except Exception as e:
            print("  PyTorch error:", e)
        print("------------------------------------------------")


def test_hardshrink_fixed_dtype_output_equality():
    """
    (1b) 固定dtype=float32，随机输入值，对比两个框架输出是否相等（误差<1e-3）
    """
    print("===== Hardshrink fixed dtype output equality test =====")
    x_np = np.random.randn(4, 4).astype(np.float32)
    ms_in = Tensor(x_np, mstype.float32)
    torch_in = torch.tensor(x_np, dtype=torch.float32)

    out_ms = F_ms.hardshrink(ms_in, lambd=0.5).asnumpy()
    out_torch = F_torch.hardshrink(torch_in, lambd=0.5).detach().numpy()

    diff = np.abs(out_ms - out_torch).max()
    print("Max diff:", diff)
    assert diff < 1e-3, f"Hardshrink outputs differ too much: {diff}"


def test_hardshrink_fixed_shape_diff_params():
    """
    (1c) 固定 shape + 固定输入值，不同输入参数类型
    """
    print("===== Hardshrink fixed shape diff params test =====")
    x = Tensor(np.array([-1.0, -0.3, 0.3, 1.0]), mstype.float32)

    # 1) lambda default=0.5
    out_default = F_ms.hardshrink(x)
    # 2) lambda float
    out_float = F_ms.hardshrink(x, lambd=1.0)
    # 3) lambda int
    out_int = F_ms.hardshrink(x, lambd=1)
    # 4) lambda bool => bool会被转成 float(1.0) or 0.0
    out_bool = F_ms.hardshrink(x, lambd=True)

    print("  out_default:", out_default.asnumpy())
    print("  out_float:", out_float.asnumpy())
    print("  out_int:", out_int.asnumpy())
    print("  out_bool:", out_bool.asnumpy())


def test_hardshrink_error_messages():
    """
    (1d) 测试随机混乱输入，报错信息的准确性
    """
    print("===== Hardshrink error messages test =====")
    # 非Tensor输入
    try:
        F_ms.hardshrink([1, -1, 2])
    except Exception as e:
        print("  Non-tensor input error:", e)

    # 不支持的dtype (e.g. int32)
    try:
        F_ms.hardshrink(Tensor(np.array([1, -1], np.int32)))
    except Exception as e:
        print("  Int32 input error:", e)


def test_hardshrink_network_forward_backward():
    """
    (2b, 2c) 使用 Hardshrink 在网络中测试前向输出和梯度
    """
    print("===== Hardshrink network forward/backward test =====")
    x_np = np.random.randn(3,).astype(np.float32)

    # PyTorch
    x_torch = torch.tensor(x_np, requires_grad=True)
    y_torch = F_torch.hardshrink(x_torch, lambd=0.5)
    y_torch_sum = y_torch.sum()
    y_torch_sum.backward()
    grad_torch = x_torch.grad.detach().numpy()

    # MindSpore
    x_ms = Tensor(x_np, mstype.float32)
    x_ms.requires_grad = True

    def ms_forward(inp):
        return F_ms.hardshrink(inp, lambd=0.5).sum()

    grad_fn = ops.grad(ms_forward, grad_position=0)
    grad_ms = grad_fn(x_ms).asnumpy()

    print("PyTorch grad:", grad_torch)
    print("MindSpore grad:", grad_ms)
