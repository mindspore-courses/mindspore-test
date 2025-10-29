import mindspore as ms
import mindspore.mint.nn.functional as F_mint
import numpy as np
import pytest
import torch
import torch.nn.functional as F_torch
from mindspore import Tensor

np.random.seed(42)
ms.set_seed(42)
torch.manual_seed(42)

@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_prelu_dtypes_support(mode):
    """测试不同dtype输入的支持情况"""
    ms.set_context(mode=mode, device_target='Ascend')
    ms_dtypes = [ms.int8, ms.int16, ms.int32, ms.int64, ms.uint8, ms.uint16, ms.uint32, ms.uint64, ms.float16, ms.float32, ms.float64, ms.bfloat16, ms.bool_]
    torch_dtypes = [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.uint16, torch.uint32, torch.uint64, torch.float16, torch.float32, torch.float64, torch.bfloat16, torch.bool]

    for i in range(len(ms_dtypes)):
        ms_dtype = ms_dtypes[i]
        torch_dtype = torch_dtypes[i]
        
        x_np = np.random.randn(2, 3)
        weight_np = np.array([0.25], dtype=np.float32)

        # PyTorch
        torch_support = True
        try:
            x_torch = torch.tensor(x_np, dtype=torch_dtype)
            _ = F_torch.prelu(x_torch, torch.tensor(weight_np, dtype=torch_dtype))
        except Exception as e:
            print(e)
            torch_support = False
        
        # MindSpore
        ms_support = True
        try:
            x_ms = Tensor(x_np, ms_dtype)
            _ = F_mint.prelu(x_ms, Tensor(weight_np, ms_dtype)).asnumpy()
        except Exception as e:
            print(e)
            ms_support = False

        assert ms_support == torch_support, f"支持情况不同：ms_dtype: {ms_dtype} ({ms_support}), torch_dtype: {torch_dtype} ({torch_support})"''

@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_prelu_fixed_dtype_accuracy(mode):
    """固定dtype测试输出一致性"""
    ms.set_context(mode=mode, device_target='Ascend')
    x_np = np.random.randn(3, 4).astype(np.float32)
    weight_np = np.array([-0.5, 0, 0.25, 1.5], dtype=np.float32)
    
    # PyTorch
    x_torch = torch.tensor(x_np)
    out_torch = F_torch.prelu(x_torch, weight=torch.tensor(weight_np))
    
    # MindSpore
    x_ms = Tensor(x_np, ms.float32)
    out_ms = F_mint.prelu(x_ms, weight=Tensor(weight_np))
    
    assert np.allclose(out_ms.asnumpy(), out_torch.detach().numpy(), atol=1e-3), "输出不一致"

@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_prelu_invalid_input(mode):
    ms.set_context(mode=mode, device_target='Ascend')
    """测试随机混乱输入，报错信息的准确性"""
    x_np = np.random.randn(2, 3).astype(np.float32)
    x_ms = Tensor(x_np, ms.float32)
    
    # 测试非Tensor输入
    with pytest.raises(TypeError):
        F_mint.prelu(np.random.randn(2,3), weight=Tensor(np.random.randn(2, 3)))
        
    # 测试weight不是Tensor
    with pytest.raises(TypeError):
        F_mint.prelu(x_ms, weight=np.random.randn(2,3))
    with pytest.raises(TypeError):
        F_mint.prelu(x_ms, weight=False)

class TorchPReLUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)
        self.linear.weight.data = torch.ones_like(self.linear.weight) * 0.5
        self.linear.bias.data.fill_(0.1)
    
    def forward(self, x):
        return F_torch.prelu(self.linear(x), weight=torch.tensor([0.25]))

class MsPReLUNet(ms.nn.Cell):
    def __init__(self):
        super().__init__()
        self.linear = ms.nn.Dense(3, 2)
        self.linear.weight.set_data(Tensor(np.full((2,3), 0.5), ms.float32))
        self.linear.bias.set_data(Tensor([0.1, 0.1], ms.float32))
    
    def construct(self, x):
        return F_mint.prelu(self.linear(x), Tensor([0.25]))

@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_prelu_network_forward(mode):
    """测试网络正向传播一致性"""
    ms.set_context(mode=mode, device_target='Ascend')
    x_np = np.random.randn(2, 3).astype(np.float32)
    
    # PyTorch
    torch_net = TorchPReLUNet()
    out_torch = torch_net(torch.tensor(x_np))
    
    # MindSpore
    ms_net = MsPReLUNet()
    out_ms = ms_net(Tensor(x_np))
    
    assert np.allclose(out_ms.asnumpy(), out_torch.detach().numpy(), atol=1e-3), "网络输出不一致"

@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_prelu_network_backward(mode):
    """测试网络反向传播梯度"""
    ms.set_context(mode=mode, device_target='Ascend')
    x_np = np.random.randn(2, 3).astype(np.float32)
    y_np = np.array([0, 1])
    
    # PyTorch
    torch_net = TorchPReLUNet()
    loss_fn = torch.nn.NLLLoss()
    
    x_torch = torch.tensor(x_np, requires_grad=True)
    out_torch = torch_net(x_torch)
    loss_torch = loss_fn(out_torch, torch.tensor(y_np))
    loss_torch.backward()
    grad_torch = torch_net.linear.weight.grad.detach().numpy()
    
    # MindSpore
    ms_net = MsPReLUNet()
    loss_fn = ms.nn.NLLLoss()
    
    def forward_fn(x, y):
        logits = ms_net(x)
        return loss_fn(logits, y)
    
    grad_fn = ms.ops.value_and_grad(forward_fn, None, ms_net.trainable_params())
    x_ms = Tensor(x_np)
    y_ms = Tensor(y_np, ms.int32)
    _, grads = grad_fn(x_ms, y_ms)
    grad_ms = grads[0].asnumpy()
    
    assert np.allclose(grad_ms, grad_torch, atol=1e-3), "权重梯度不一致"

