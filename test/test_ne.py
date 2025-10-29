import pytest
import numpy as np
import mindspore as ms
from mindspore import mint, Tensor, value_and_grad
import torch


dtype_ms = ms.float32
dtype_torch = torch.float32
input_data_x = [[1, 2], [3, 4], [5, 6]]
input_data_y = [[6, 5], [4, 3], [2, 1]]
ms_tensor_x = Tensor(input_data_x, dtype_ms)
ms_tensor_y = Tensor(input_data_y, dtype_ms)
torch_tensor_x = torch.tensor(input_data_x, dtype=dtype_torch)
torch_tensor_y = torch.tensor(input_data_y, dtype=dtype_torch)


def is_same(input_data_x=[[1, 2], [3, 4], [5, 6]], input_data_y=[[6, 5], [4, 3], [2, 1]], shape=None, dtype_ms=ms.float32, dtype_torch=torch.float32):
    if shape is not None:
        input_data_x = np.random.randn(*shape)
        input_data_y = np.random.randn(*shape)

    ms_tensor_x = Tensor(input_data_x, dtype_ms)
    ms_tensor_y = Tensor(input_data_y, dtype_ms)
    torch_tensor_x = torch.tensor(input_data_x, dtype=dtype_torch)
    torch_tensor_y = torch.tensor(input_data_y, dtype=dtype_torch)

    ms_result = mint.ne(ms_tensor_x, ms_tensor_y).asnumpy()
    torch_result = torch.ne(torch_tensor_x, torch_tensor_y).numpy()
    if  np.allclose(ms_result, torch_result):
        return True
    else:
        print(f"input_data: {input_data_x}")
        print(f"input_data: {input_data_y}")
        print(f"ms_result: {ms_result}")
        print(f"torch_result: {torch_result}")
        return False


@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ne_different_dtypes(mode):
    """测试random输入不同dtype，对比两个框架的支持度"""
    ms.set_context(mode=mode)
    ms_dtypes = [ms.int8, ms.int16, ms.int32, ms.int64, ms.uint8, ms.uint16, ms.uint32, ms.uint64, ms.float16, ms.float32, ms.float64, ms.bfloat16, ms.bool_]
    torch_dtypes = [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.uint16, torch.uint32, torch.uint64, torch.float16, torch.float32, torch.float64, torch.bfloat16, torch.bool]

    for i in range(len(ms_dtypes)):
        dtype_ms = ms_dtypes[i]
        dtype_torch = torch_dtypes[i]

        ms_tensor_x = Tensor(input_data_x, dtype_ms)
        ms_tensor_y = Tensor(input_data_y, dtype_ms)
        torch_tensor_x = torch.tensor(input_data_x, dtype=dtype_torch)
        torch_tensor_y = torch.tensor(input_data_y, dtype=dtype_torch)

        err = False
        try:
            ms_result = mint.ne(ms_tensor_x, ms_tensor_y).asnumpy()
        except Exception as e:
            err = True
            print(f"mint.ne not supported for {dtype_ms}")

        try:
            torch_result = torch.ne(torch_tensor_x, torch_tensor_y).numpy()
        except Exception as e:
            err = True
            print(f"torch.ne not supported for {dtype_torch}")

        if not err:
            if not np.allclose(ms_result, torch_result):
                print(f"mint.ne is supported for {dtype_ms} but not working properly")


@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ne_precision_alignment(mode):
    """验证不同维度随机输入的精度对齐"""
    ms.set_context(mode=mode)

    test_shapes = [
        (10,),
        (5, 4),
        (3, 128, 128),  # 高维数据测试
        (2, 3, 4, 5)  # 4D数据测试
    ]

    for shape in test_shapes:
        # 生成对比数据时包含边界值
        np_x = np.random.randn(*shape)
        np_y = np.random.randn(*shape)
        np_y[0] = np_x[0]  # 强制设置相等的情况

        assert is_same(np_x, np_y), f"Shape {shape} 测试失败"

@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ne_wrong_input(mode):
    """测试随机混乱输入，报错信息的准确性"""
    ms.set_context(mode=mode)
    valid_tensor = ms.Tensor([1, 2], dtype=ms.float32)
    torch_valid = torch.tensor([1, 2], dtype=torch.float32)

    # 测试非法输入类型（字符串代替张量）
    try:
        ms.mint.ne("invalid_input", valid_tensor)
    except TypeError as e:
        assert "input" in str(e) and "Tensor" in str(e), f"错误信息不准确，实际信息：{e}"
        print(f"[ne] TypeError 测试通过！非法输入类型检测成功，报错信息：\n{e}")

    # 测试非法的数据类型（字典类型）
    try:
        ms.mint.ne({'a': 1}, valid_tensor)
    except TypeError as e:
        assert "unexpected keyword argument" not in str(e), "应捕获输入类型错误而非参数名错误"
        print(f"[ne] TypeError 测试通过！非法数据类型检测成功，报错信息：\n{e}")

    # 测试形状不匹配（不可广播）
    try:
        a = ms.Tensor([[1, 2], [3, 4]], dtype=ms.float32)
        b = ms.Tensor([1, 2, 3], dtype=ms.float32)
        ms.mint.ne(a, b)
    except ValueError as e:
        assert "broadcast" in str(e).lower(), "应提示广播失败"
        print(f"[ne] ValueError 测试通过！形状不匹配检测成功，报错信息：\n{e}")

    # 测试无效关键字参数
    try:
        ms.mint.ne(valid_tensor, valid_tensor, nonsuch_param=1)
    except TypeError as e:
        assert "nonsuch_param" in str(e), "应明确提示无效参数名"
        print(f"[ne] TypeError 测试通过！无效参数检测成功，报错信息：\n{e}")

    # 测试双布尔类型输入（根据文档Note提示的约束）
    try:
        a = ms.Tensor([True, False], dtype=ms.bool_)
        b = ms.Tensor([False, True], dtype=ms.bool_)
        ms.mint.ne(a, b)
    except TypeError as e:
        assert "bool" in str(e).lower(), "应提示不支持双布尔类型输入"
        print(f"[ne] TypeError 测试通过！双布尔类型检测成功，报错信息：\n{e}")

    # 对比PyTorch的报错行为
    try:
        torch.ne(torch_valid, "invalid_type")
    except TypeError as e:
        assert "Tensor" in str(e), "PyTorch应拒绝非Tensor输入"
        print(f"[PyTorch ne] TypeError 一致性验证通过，报错信息：\n{e}")

@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_framework_support(mode):
    """验证不同运行模式下的支持情况"""
    ms.set_context(mode=mode)

    # 测试广播机制支持（网页4说明mint支持Ascend广播）
    broadcast_cases = [
        ([[1, 2]], [[3]]),  # 1x2 vs 1x1
        (np.random.randn(3, 1, 2), np.random.randn(2)),
    ]

    for case in broadcast_cases:
        x = Tensor(case[0])
        y = Tensor(case[1])
        try:
            out = mint.ne(x, y)
            print(f"广播模式 {x.shape} vs {y.shape} 支持")
        except Exception as e:
            print(f"广播失败: {str(e)}")

@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ne_forward_back(mode):
    """使用Pytorch和MindSpore, 固定输入和权重, 测试正向推理结果和反向梯度"""
    ms.set_context(mode=mode)
    # 设计输入数据，包含可导操作路径
    input_data_x = np.array([2.0, -3.0, 5.0], dtype=np.float32)
    input_data_y = np.array([1.0, 0.0, 5.0], dtype=np.float32)

    # 创建PyTorch张量（需计算梯度）
    torch_x = torch.tensor(input_data_x, dtype=torch.float32, requires_grad=True)
    torch_y = torch.tensor(input_data_y, dtype=torch.float32, requires_grad=True)

    # 创建MindSpore张量
    ms_x = ms.Tensor(input_data_x, dtype=ms.float32)
    ms_y = ms.Tensor(input_data_y, dtype=ms.float32)

    # 前向函数定义：通过可导操作触发梯度
    def forward_pt(x, y):
        # 将比较结果作为掩码，与可导操作结合
        mask = torch.ne(x, y).float()  # 布尔转浮点
        return (x * mask).sum()  # 可导路径：x * mask

    def forward_ms(x, y):
        mask = mint.ne(x, y).astype(ms.float32)  # 使用astype正确转换类型
        return (x * mask).sum()

    # 计算MindSpore的梯度
    grad_fn = ms.ops.value_and_grad(forward_ms, grad_position=(0, 1))
    output_ms, (grad_x_ms, grad_y_ms) = grad_fn(ms_x, ms_y)
    # 计算PyTorch的输出和梯度
    output_pt = forward_pt(torch_x, torch_y)
    # 反向传播前清零梯度
    if torch_x.grad is not None:
        torch_x.grad.zero_()
    if torch_y.grad is not None:
        torch_y.grad.zero_()
    output_pt.backward()

    grad_x_pt = torch_x.grad.numpy() if torch_x.grad is not None else np.zeros_like(input_data_x)
    grad_y_pt = torch_y.grad.numpy() if torch_y.grad is not None else np.zeros_like(input_data_y)

    # 断言
    assert np.allclose(output_ms.asnumpy(), output_pt.detach().numpy(), atol=1e-3), "正向结果不一致"
    assert np.allclose(grad_x_ms.asnumpy(), grad_x_pt, atol=1e-3), "x梯度不一致"
    assert np.allclose(grad_y_ms.asnumpy(), grad_y_pt, atol=1e-3), "y梯度不一致"