import numpy as np
import mindspore
from mindspore import Tensor, dtype as mstype, ops
import mindspore.mint as mint
import torch

def test_broadcast_to_random_dtype_support():
    """
    (1a) 随机输入，不同dtype
    """
    print("===== broadcast_to random dtype support test =====")
    shape_target = (2,3)
    for dt in [mstype.float16, mstype.float32, mstype.int32]:
        x_np = np.random.randn(1,3)
        if dt == mstype.int32:
            x_np = x_np.astype(np.int32)
        else:
            x_np = x_np.astype(mindspore.dtype_to_nptype(dt))
        try:
            out_ms = mint.broadcast_to(Tensor(x_np, dt), shape_target)
            print(f"MindSpore dtype={dt}, shape={out_ms.shape}")
        except Exception as e:
            print("MindSpore error:", e)

        # PyTorch
        torch_dt = torch.float16 if dt == mstype.float16 else torch.float32
        t_in = torch.tensor(x_np, dtype=torch_dt)
        # broadcast_to is new in PyTorch >=1.10, or use expand
        try:
            out_pt = torch.broadcast_to(t_in, shape_target)
            print(f"PyTorch dtype={torch_dt}, shape={out_pt.shape}")
        except Exception as e:
            print("PyTorch error:", e)
        print("-------------------------------------------")


def test_broadcast_to_fixed_dtype_output_equality():
    """
    (1b) broadcast_to 不改变数值，主要比较形状
    """
    print("===== broadcast_to fixed dtype output test =====")
    x_np = np.array([1,2,3], np.float32)
    shape_target = (3,3)
    out_ms = mint.broadcast_to(Tensor(x_np), shape_target).asnumpy()
    out_pt = torch.broadcast_to(torch.tensor(x_np), shape_target).numpy()

    # 检查值
    diff = np.abs(out_ms - out_pt).sum()
    print("Sum diff:", diff)
    assert diff < 1e-3
    # 检查形状
    assert out_ms.shape == shape_target


def test_broadcast_to_fixed_shape_diff_params():
    """
    (1c) 测试传入-1
    """
    print("===== broadcast_to fixed shape diff params test =====")
    x = Tensor(np.ones((2,1)), mstype.float32)
    out = mint.broadcast_to(x, (-1, 3))
    print("shape from (2,1) to (-1,3) =>", out.shape)
    print(out.asnumpy())

    # 非tuple => error
    try:
        mint.broadcast_to(x, [-1,3])
    except Exception as e:
        print("non-tuple shape error:", e)

    # 多个-1 => error
    try:
        mint.broadcast_to(x, (-1,-1))
    except Exception as e:
        print("multiple -1 error:", e)


def test_broadcast_to_error_messages():
    """
    (1d) 测试随机混乱输入
    """
    print("===== broadcast_to error messages test =====")
    # 维度不匹配
    x = Tensor(np.ones((2,2)), mstype.float32)
    try:
        mint.broadcast_to(x, (3,2))  # can't broadcast dim=2 -> 3
    except Exception as e:
        print("dim mismatch error:", e)

    # 非Tensor输入
    try:
        mint.broadcast_to([1,2], (2,))
    except Exception as e:
        print("non-tensor error:", e)


def test_broadcast_to_network_forward_backward():
    """
    (2b,2c) 测试广播在网络中 + 反向梯度
    """
    print("===== broadcast_to network forward/backward test =====")
    # PyTorch
    a_pt = torch.tensor([[1.],[2.]], requires_grad=True)  # shape (2,1)
    b_pt = a_pt.expand(2,3)  # broadcast to (2,3)
    loss_pt = b_pt.sum()
    loss_pt.backward()
    grad_a_pt = a_pt.grad.numpy()
    print("PyTorch grad a:", grad_a_pt)

    # MindSpore
    a_ms = Tensor(np.array([[1.],[2.]], np.float32))
    a_ms.requires_grad = True
    def forward_fn(x):
        return mint.broadcast_to(x, (2,3)).sum()
    grad_fn = ops.grad(forward_fn, grad_position=0)
    grad_ms = grad_fn(a_ms).asnumpy()
    print("MindSpore grad a:", grad_ms)
