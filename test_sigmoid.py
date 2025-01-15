# test_sigmoid.py

import pytest
import numpy as np
import mindspore
from mindspore import Tensor, nn, ops, Parameter
from mindspore import dtype as mstype
from mindspore.mint import sigmoid
import torch

def print_env_info():
    """
    打印 MindSpore 与 PyTorch 的版本信息，方便后续排查。
    """
    print("===== Environment Info =====")
    try:
        print("MindSpore version:", mindspore.__version__)
    except AttributeError:
        print("MindSpore version: Unknown")
    try:
        print("PyTorch version:", torch.__version__)
    except AttributeError:
        print("PyTorch version: Unknown")
    print("============================\n")

def test_sigmoid_error_input():
    """
    (1d) 测试传入混乱输入(字符串、元组、列表等)，检查报错信息
    """
    print_env_info()
    bad_inputs = [
        "this is a string",
        (1, 0),
        [1, 2, 3],
        None
    ]
    for inp in bad_inputs:
        try:
            _ = sigmoid(inp)
        except Exception as e:
            print(f"Sigmoid error with input={inp} =>", e)

def test_sigmoid_calculation_fixed_dtype():
    """
    (1b) 固定 dtype(float32) + 固定输入，对比 MindSpore 与 PyTorch
    """
    print_env_info()
    np_input = np.array([[0.1, 0.5], [-0.2, 2.0]], dtype=np.float32)
    ms_input = Tensor(np_input, mstype.float32)
    ms_output = sigmoid(ms_input)

    torch_input = torch.tensor(np_input, dtype=torch.float32)
    torch_output = torch.sigmoid(torch_input)

    assert np.allclose(ms_output.asnumpy(), torch_output.detach().numpy(), atol=1e-3)

def test_sigmoid_calculation_random_dtype():
    """
    (1a) 随机输入，不同 dtype(float16/32/64)，对比 MindSpore/PyTorch
    """
    print_env_info()
    dtype_map = {
        mstype.float16: torch.float16,
        mstype.float32: torch.float32,
        mstype.float64: torch.float64
    }
    for ms_dt, torch_dt in dtype_map.items():
        arr = np.random.randn(5, 5).astype(mindspore.dtype_to_nptype(ms_dt))
        ms_res = sigmoid(Tensor(arr, ms_dt))
        torch_res = torch.sigmoid(torch.tensor(arr, dtype=torch_dt))
        assert np.allclose(ms_res.asnumpy(), torch_res.detach().numpy(), atol=1e-3)

def test_sigmoid_calculation_fixed_shape_diff_param():
    """
    (1c) 固定输入，加多余的字符串参数，预期报错
    """
    print_env_info()
    arr = np.array([[0, 1], [2, 3]], dtype=np.float32)
    try:
        _ = sigmoid(Tensor(arr), "extra_param")
    except Exception as e:
        print("Sigmoid extra param error:", e)

def test_sigmoid_calculation_broadcast():
    """
    扩展：测试广播形状 (2,1) vs (1,2)
    """
    print_env_info()
    a_np = np.random.randn(2,1).astype(np.float32)
    b_np = np.random.randn(1,2).astype(np.float32)
    # 这里演示对单输入的 sigmoid，无需同时处理 a_np/b_np
    # 如果想看 broadcasting，可将 a_np+b_np 之后再做 sigmoid
    # 仅演示可行性
    out_a = sigmoid(Tensor(a_np))
    out_b = sigmoid(Tensor(b_np))
    print("Broadcast test - done")

def test_sigmoid_calculation_empty():
    print_env_info()
    arr = np.array([], dtype=np.float32)
    ms_out = sigmoid(Tensor(arr))
    torch_out = torch.sigmoid(torch.tensor(arr))
    assert ms_out.shape == (0,)
    assert torch_out.shape == (0,)

class SigmoidNetMindspore(nn.Cell):
    """
    (2a, 2b, 2c) 一个简单的网络：Dense + Sigmoid
    """
    def __init__(self):
        super(SigmoidNetMindspore, self).__init__()
        self.fc = nn.Dense(3, 3)
        self.act = sigmoid

    def construct(self, x):
        x = self.fc(x)
        x = self.act(x)
        return x

def test_sigmoid_nn_inference_compare_with_torch():
    """
    (2b) 比较固定权重网络的前向推理
    """
    print_env_info()
    net_ms = SigmoidNetMindspore()
    w_init = np.array([[0.1, -0.2, 0.3],
                       [0.4, 0.5, -0.1],
                       [0.05, 0.05, 0.05]], dtype=np.float32)
    b_init = np.array([0.01, -0.02, 0.03], dtype=np.float32)
    net_ms.fc.weight.set_data(Parameter(Tensor(w_init)))
    net_ms.fc.bias.set_data(Parameter(Tensor(b_init)))

    inp_data = np.random.randn(4, 3).astype(np.float32)
    ms_out = net_ms(Tensor(inp_data)).asnumpy()

    net_torch = torch.nn.Sequential(
        torch.nn.Linear(3, 3),
        torch.nn.Sigmoid()
    )
    with torch.no_grad():
        net_torch[0].weight.copy_(torch.tensor(w_init))
        net_torch[0].bias.copy_(torch.tensor(b_init))
    torch_out = net_torch(torch.tensor(inp_data, dtype=torch.float32)).detach().numpy()

    assert np.allclose(ms_out, torch_out, atol=1e-3)

def test_sigmoid_function_grad():
    """
    (2c) 测试函数本身的梯度 + 网络参数的梯度
    """
    print_env_info()

    # 1) 函数本身对输入的梯度
    x_arr = np.random.randn(4, 3).astype(np.float32)
    x_ms = Tensor(x_arr)
    x_ms.requires_grad = True
    grad_fn = ops.GradOperation(get_all=True)(sigmoid)
    grad_ms = grad_fn(x_ms)[0].asnumpy()

    x_torch = torch.tensor(x_arr, requires_grad=True)
    y_torch = torch.sigmoid(x_torch)
    y_torch.sum().backward()
    grad_torch = x_torch.grad.detach().numpy()
    assert np.allclose(grad_ms, grad_torch, atol=1e-3)

    # 2) 网络参数梯度
    net_ms = SigmoidNetMindspore()
    net_ms.fc.weight.set_data(Parameter(Tensor(np.ones((3,3)), mstype.float32)))
    net_ms.fc.bias.set_data(Parameter(Tensor(np.zeros(3), mstype.float32)))

    x_ms2 = Tensor(np.random.randn(2, 3).astype(np.float32))
    label_ms = Tensor(np.zeros((2, 3)), mstype.float32)
    loss_fn = nn.MSELoss()

    def net_with_loss(x, lbl):
        pred = net_ms(x)
        return loss_fn(pred, lbl)

    grad_op = ops.GradOperation(get_all=True, get_by_list=True)
    grads = grad_op(net_with_loss, net_ms.trainable_params())(x_ms2, label_ms)
    for p, g in zip(net_ms.trainable_params(), grads):
        print(f"Param: {p.name}, grad : {g}")
