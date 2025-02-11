import mindspore as ms
import numpy as np
from mindspore import Tensor, mint
import torch
from mindspore import value_and_grad

# 设置随机种子
seed = 42
torch.manual_seed(seed)
ms.set_seed(seed)


# 生成随机输入数据
def generate_random_input(dtype, shape=(10,)):
    if dtype in [torch.float32, torch.float64, torch.int32, torch.int64]:
        return torch.randn(shape).to(dtype) if dtype in [torch.float32, torch.float64] else torch.randint(0, 10,
                                                                                                          shape).to(
            dtype)
    else:
        return ms.Tensor(np.random.randn(*shape).astype(dtype))


'''
1.对应Pytorch 的相应接口进行测试：
a) 测试random输入不同dtype，对比两个框架的支持度
'''
def test_random_vary_detype():
    dtypes = [torch.float32, torch.float64, torch.int32, torch.int64,
              ms.float32, ms.float64, ms.int32, ms.int64]

    for dtype in ['torch.float32', 'torch.float64', 'torch.int32', 'torch.int64',
                  'ms.float32', 'ms.float64', 'ms.int32', 'ms.int64']:
        try:
            if 'torch' in dtype:
                dt = getattr(torch, dtype.split('.')[1])
                x = torch.randn(10).to(dt) if 'float' in dtype else torch.randint(0, 10, (10,)).to(dt)
                arccos_x = torch.acos(x)
                arccosh_x = torch.arccosh(x)
                arcsin_x = torch.arcsin(x)
                arcsinh_x = torch.arcsinh(x)
                arctan_x = torch.arctan(x)
            else:
                dt = getattr(ms, dtype.split('.')[1])
                x = ms.Tensor(np.random.randn(10), dt)
                arccos_x = mint.arccos(x)
                arccosh_x = mint.arccosh(x)
                arcsin_x = mint.arcsin(x)
                arcsinh_x = mint.arcsinh(x)
                arctan_x = mint.arctan(x)
                # print(arctan_x)
            print(f"{dtype} 支持")
        except Exception as e:
            print(f"{dtype} 不支持: {e}")


test_random_vary_detype()


'''
b) 测试固定dtype，random输入值，对比两个框架输出是否相等（误差范围为小于1e-3）
c) 测试固定shape，固定输入值，不同输入参数（string/bool等类型），两个框架的支持度(测试的函数没有参数，因此这一项不需要做)
'''
def test_random_fix_detype():
    # 定义测试参数
    num_tests = 100
    shape = (100,)  # 每次生成100个元素
    dtype = np.float32
    threshold = 1e-3

    # 记录通过测试的次数
    pass_counts = {func: 0 for func in ["acos", "arccosh", "arcsin", "arcsinh", "arctan"]}

    for test_num in range(1, num_tests + 1):
        test_results = {}

        # 生成不同函数的输入数据
        input_data = {
            "acos": np.random.uniform(-1, 1, shape).astype(dtype),  # [-1,1] 适用于 acos
            "arccosh": np.random.uniform(1, 1e10, shape).astype(dtype),  # [1,∞) 适用于 arccosh
            "arcsin": np.random.uniform(-1, 1, shape).astype(dtype),  # [-1,1] 适用于 arcsin
            "arcsinh": np.random.uniform(-1e10, 1e10, shape).astype(dtype),  # 全体实数 适用于 arcsinh
            "arctan": np.random.uniform(-1e10, 1e10, shape).astype(dtype),  # 全体实数 适用于 arctan
        }

        for func_name, input_np in input_data.items():
            input_torch = torch.tensor(input_np)
            input_ms = ms.Tensor(input_np)

            try:
                # 计算 PyTorch 和 MindSpore 结果
                output_torch = getattr(torch, func_name)(input_torch)
                output_ms = getattr(mint, func_name)(input_ms)

                # 计算误差
                difference = np.abs(output_torch.detach().numpy() - output_ms.asnumpy())

                if np.all(difference < threshold):
                    pass_counts[func_name] += 1
                    test_results[func_name] = "通过"
                else:
                    max_diff = np.max(difference)
                    test_results[func_name] = f"不通过，最大误差为: {max_diff}"

            except Exception as e:
                test_results[func_name] = f"抛出异常 - {e}"

        # 打印当前测试轮次的结果
        print(f"测试 {test_num}: {test_results}")

    # 总结结果
    print("\n在 100 次测试中，各个函数的通过次数:")
    for func, count in pass_counts.items():
        print(f"{func}: {count} 次")


# 运行测试
test_random_fix_detype()


# %%
def create_tensors(input_data, ms_dtype, torch_dtype, requires_grad=True):
    ms_tensor = ms.Tensor(input_data, dtype=ms_dtype)
    torch_tensor = torch.tensor(input_data, dtype=torch_dtype, requires_grad=requires_grad)
    return ms_tensor, torch_tensor


'''
d) 测试随机混乱输入，报错信息的准确性
'''
def test_extreme_invalid_inputs():
    # 定义各种类型的非法输入
    invalid_inputs = {
        "acos": np.array([-2, 2, np.nan, np.inf, -np.inf, "invalid", b"bytes_data", True, False, None], dtype=object),
        "arccosh": np.array([-1, 0.5, np.nan, -np.inf, "test", [], {}, None], dtype=object),
        "arcsin": np.array([-2, 2, np.nan, np.inf, -np.inf, "sin_error", True, None], dtype=object),
        "arcsinh": np.array([np.nan, "NaN_test", None, []], dtype=object),
        "arctan": np.array([np.nan, "tan_test", None, {}], dtype=object),
    }

    print("=== 开始混乱输入测试 ===\n")

    for func_name, inputs in invalid_inputs.items():
        print(f"Testing {func_name} with invalid inputs...\n")

        for i, input_val in enumerate(inputs):
            print(f"Test {i + 1}/{len(inputs)}: Input value: {input_val}")

            # 尝试MindSpore
            try:
                ms_tensor, _ = create_tensors(np.array([input_val]), ms.float32, torch.float32, requires_grad=True)
                result_ms = None
                if func_name == "acos":
                    result_ms = mint.arccos(ms_tensor)
                elif func_name == "arccosh":
                    result_ms = mint.arccosh(ms_tensor)
                elif func_name == "arcsin":
                    result_ms = mint.arcsin(ms_tensor)
                elif func_name == "arcsinh":
                    result_ms = mint.arcsinh(ms_tensor)
                elif func_name == "arctan":
                    result_ms = mint.arctan(ms_tensor)
                print(f"MindSpore result: {result_ms}")
            except Exception as e:
                print(f"MindSpore failed with error: {e}")

            # 尝试PyTorch
            try:
                _, torch_tensor = create_tensors(np.array([input_val]), ms.float32, torch.float32, requires_grad=True)
                result_torch = None
                if func_name == "acos":
                    result_torch = torch.acos(torch_tensor)
                elif func_name == "arccosh":
                    result_torch = torch.acosh(torch_tensor)
                elif func_name == "arcsin":
                    result_torch = torch.asin(torch_tensor)
                elif func_name == "arcsinh":
                    result_torch = torch.asinh(torch_tensor)
                elif func_name == "arctan":
                    result_torch = torch.atan(torch_tensor)
                print(f"PyTorch result: {result_torch}")
            except Exception as e:
                print(f"PyTorch failed with error: {e}")
            print("-" * 50)  # 分隔符，便于查看

    print("\n=== 测试完成 ===")


test_extreme_invalid_inputs()


# %%
def compare_results(ms_result, torch_result):
    """对比 MindSpore 和 PyTorch 计算结果是否一致"""
    ms_numpy = ms_result.asnumpy()
    torch_numpy = torch_result.detach().numpy()
    assert np.allclose(ms_numpy, torch_numpy, atol=1e-3), f"Mismatch found:\nMS: {ms_numpy}\nTorch: {torch_numpy}"


'''
2. 测试使用接口构造函数/神经网络的准确性
a) Github搜索带有该接口的代码片段/神经网络
b) 使用Pytorch和MindSpore，固定输入和权重，测试正向推理结果（误差范围小于1e-3，若报错则记录报错信息）
c) 测试该神经网络/函数反向，如果是神经网络，则测试Parameter的梯度，如果是函数，则测试函数输入的梯度
'''
def test_exp_forward_backward():
    """测试 MindSpore 和 PyTorch 在 5 个反三角函数上的前向和反向计算一致性"""

    # 适用于 arccos(x) 和 arcsin(x) 的输入范围 [-1, 1]
    input_data_1 = np.random.uniform(-1, 1, (3, 3)).astype(np.float32)

    # 适用于 arccosh(x) 的输入范围 [1.01, 10]，避免 x=1 时梯度爆炸
    input_data_2 = np.random.uniform(1.01, 10, (3, 3)).astype(np.float32)

    # 适用于 arcsinh(x) 和 arctan(x) 的输入范围 (-∞, ∞)
    input_data_3 = np.random.uniform(-5, 5, (3, 3)).astype(np.float32)

    test_cases = [
        ("arccos", mint.arccos, torch.arccos, input_data_1),
        ("arccosh", mint.arccosh, torch.arccosh, input_data_2),
        ("arcsin", mint.arcsin, torch.arcsin, input_data_1),
        ("arcsinh", mint.arcsinh, torch.arcsinh, input_data_3),
        ("arctan", mint.arctan, torch.arctan, input_data_3),
    ]

    for name, ms_func, torch_func, input_data in test_cases:
        print(f"Testing {name}...")

        # 创建张量
        ms_tensor, torch_tensor = create_tensors(input_data, ms.float32, torch.float32, requires_grad=True)

        # 前向传播
        ms_result = ms_func(ms_tensor)
        torch_result = torch_func(torch_tensor)
        compare_results(ms_result, torch_result)

        # 反向传播
        grad_fn = value_and_grad(lambda x: ms_func(x).sum())  # 计算梯度
        _, ms_grad = grad_fn(ms_tensor)

        torch_result.sum().backward()
        torch_grad = torch_tensor.grad

        compare_results(ms_grad, torch_grad)

    print("All inverse trigonometric function tests passed!")


# 运行测试
test_exp_forward_backward()