import numpy as np
import mindspore
from mindspore import Tensor, dtype as mstype, ops
import mindspore.mint.nn.functional as F_ms
import torch
import torch.nn.functional as F_torch

def test_layernorm_random_dtype_support():
    """
    (1a) 测试 random 输入 不同 dtype
    MindSpore LayerNorm 一般支持 float16, float32
    """
    print("===== LayerNorm random dtype support test =====")
    dtypes_to_test = [mstype.float16, mstype.float32, mstype.int32]
    for dt in dtypes_to_test:
        x_np = np.random.randn(2,4).astype(np.float32 if dt != mstype.int32 else np.int32)
        try:
            out_ms = F_ms.layer_norm(Tensor(x_np, dt), normalized_shape=(4,))
            print(f"MindSpore dtype={dt}, shape={out_ms.shape}")
        except Exception as e:
            print(f"MindSpore error with dtype={dt}:", e)

        # PyTorch
        torch_dt = torch.float16 if dt == mstype.float16 else torch.float32
        x_torch = torch.tensor(x_np, dtype=torch_dt)
        try:
            out_pt = F_torch.layer_norm(x_torch, normalized_shape=(4,))
            print(f"PyTorch dtype={torch_dt}, shape={out_pt.shape}")
        except Exception as e:
            print(f"PyTorch error with dtype={torch_dt}:", e)
        print("--------------------------------------")


def test_layernorm_fixed_dtype_output_equality():
    """
    (1b) 固定dtype=float32, 随机输入, 对比输出
    """
    print("===== LayerNorm fixed dtype output equality test =====")
    x_np = np.random.randn(2,3,4).astype(np.float32)
    w_np = np.random.randn(4).astype(np.float32)
    b_np = np.random.randn(4).astype(np.float32)
    ms_in = Tensor(x_np, mstype.float32)
    ms_w = Tensor(w_np, mstype.float32)
    ms_b = Tensor(b_np, mstype.float32)

    out_ms = F_ms.layer_norm(ms_in, normalized_shape=(4,), weight=ms_w, bias=ms_b, eps=1e-5).asnumpy()

    x_torch = torch.tensor(x_np, dtype=torch.float32)
    w_torch = torch.tensor(w_np, dtype=torch.float32)
    b_torch = torch.tensor(b_np, dtype=torch.float32)
    out_pt = F_torch.layer_norm(x_torch, normalized_shape=(4,), weight=w_torch, bias=b_torch, eps=1e-5).numpy()

    diff = np.abs(out_ms - out_pt).max()
    print("Max diff:", diff)
    assert diff < 1e-3, f"LayerNorm diff too large: {diff}"


def test_layernorm_fixed_shape_diff_params():
    """
    (1c) 测试 normalized_shape 是 int or tuple, weight/bias 可省略
    """
    print("===== LayerNorm fixed shape diff params test =====")
    x = Tensor(np.random.randn(2,4).astype(np.float32))
    # normalized_shape int vs tuple
    out1 = F_ms.layer_norm(x, 4)  # int
    out2 = F_ms.layer_norm(x, (4,))  # tuple
    diff = np.abs(out1.asnumpy() - out2.asnumpy()).max()
    print("normalized_shape int vs tuple diff:", diff)

    # weight/bias省略
    out3 = F_ms.layer_norm(x, normalized_shape=4)
    print("No weight/bias out shape:", out3.shape)


def test_layernorm_error_messages():
    """
    (1d) 非法输入
    """
    print("===== LayerNorm error messages test =====")
    # not matching shape
    try:
        F_ms.layer_norm(Tensor(np.random.randn(2,3), mstype.float32), normalized_shape=(4,))
    except Exception as e:
        print("shape not match error:", e)

    # weight dimension mismatch
    try:
        F_ms.layer_norm(Tensor(np.random.randn(2,4), mstype.float32), normalized_shape=(4,),
                        weight=Tensor(np.random.randn(5).astype(np.float32)))
    except Exception as e:
        print("weight mismatch error:", e)

    # invalid eps
    try:
        F_ms.layer_norm(Tensor(np.random.randn(2,4), mstype.float32), normalized_shape=(4,), eps="1e-5")
    except Exception as e:
        print("invalid eps error:", e)


def test_layernorm_network_forward_backward():
    """
    (2b,2c) 测试LayerNorm在网络的正向推理和反向梯度
    """
    print("===== LayerNorm forward/backward test =====")
    # PyTorch
    x_pt = torch.randn(2,3,4, requires_grad=True)
    out_pt = F_torch.layer_norm(x_pt, normalized_shape=(4,))
    loss_pt = out_pt.sum()
    loss_pt.backward()
    grad_pt = x_pt.grad.numpy()

    # MindSpore
    x_ms = Tensor(x_pt.detach().numpy(), mstype.float32)
    x_ms.requires_grad = True
    def forward_fn(x):
        return F_ms.layer_norm(x, normalized_shape=(4,)).sum()
    grad_fn = ops.grad(forward_fn, grad_position=0)
    grad_ms = grad_fn(x_ms).asnumpy()

    diff = np.abs(grad_pt - grad_ms).max()
    print("Max grad diff:", diff)
    assert diff < 1e-3
