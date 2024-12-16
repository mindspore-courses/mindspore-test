import pytest
import torch
import numpy as np
import mindspore as ms
from mindspore import mint, Tensor

input_data = np.random.rand(10, 20, 30) 
dropout_rate = 0.5 

ms_tensor = Tensor(input_data, ms.float32)
torch_tensor = torch.tensor(input_data, dtype=torch.float32)


@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_dropout_train_eval(mode):
    """测试训练模式和评估模式下的Dropout表现"""
    ms.set_context(mode=mode)

    # MindSpore Dropout
    dropout_ms = mint.nn.Dropout(p=dropout_rate)

    dropout_ms.set_train()  
    ms_output_train = dropout_ms(ms_tensor)
    dropout_ms.set_train(False)
    ms_output_eval = dropout_ms(ms_tensor)
    np_array = np.array(input_data)
    # 检查训练模式和评估模式下的输出差异
    # print("Dropout之前的数据是：")
    # print(ms_tensor)
    # print("Dropout在train模式下的结果如下：")
    # print(ms_output_train)

    assert np.allclose(np_array, ms_output_eval.asnumpy(),
                           atol=1e-3), "Mindspore Dropout behavior in training mode is not as expected, eval also dropout"


@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_dropout_sparse_in_training(mode):
    """测试训练模式下的Dropout稀疏性"""
    ms.set_context(mode=mode)

    # MindSpore Dropout
    dropout_ms = mint.nn.Dropout(p=dropout_rate)
    dropout_ms.set_train()
    # PyTorch Dropout
    dropout_torch = torch.nn.Dropout(p=dropout_rate)

    ms_output_train = dropout_ms(ms_tensor)
    print(ms_output_train)
    print("((*(*(")
    torch_output_train = dropout_torch(torch_tensor)

    ms_nonzero = np.count_nonzero(ms_output_train.asnumpy())
    torch_nonzero = torch.count_nonzero(torch_output_train).item()

    # 对比是否丢弃了接近50%的神经元（考虑到一定的随机性）
    ms_nonzero_ratio = ms_nonzero / ms_output_train.size
    torch_nonzero_ratio = torch_nonzero / torch_output_train.numel()

    assert not abs(1 - ms_nonzero_ratio - dropout_rate) < 0.1, f"MindSpore的Dropout之后剩余的比例: {ms_nonzero_ratio}"



@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_dropout_reproducibility(mode):
    """测试Dropout的可重复性（相同输入是否得到相同的输出）"""
    ms.set_context(mode=mode)

    # MindSpore Dropout
    dropout_ms = mint.nn.Dropout(p=dropout_rate)
    dropout_ms.set_train()
    # PyTorch Dropout
    dropout_torch = torch.nn.Dropout(p=dropout_rate)

    ms_output_1 = dropout_ms(ms_tensor)
    ms_output_2 = dropout_ms(ms_tensor)

    torch_output_1 = dropout_torch(torch_tensor)
    torch_output_2 = dropout_torch(torch_tensor)

    assert not np.allclose(ms_output_1.asnumpy(), ms_output_2.asnumpy(),
                           atol=1e-3), "MindSpore Dropout results should differ between different calls"
    assert not np.allclose(torch_output_1.detach().numpy(), torch_output_2.detach().numpy(),
                           atol=1e-3), "PyTorch Dropout results should differ between different calls"

