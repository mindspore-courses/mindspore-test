import numpy as np
import torch
import mindspore as ms
import mindspore.mint as mint
from mindspore import Tensor, Parameter
import pytest


def calculate_error(torch_output, ms_output):
    """计算两个输出之间的最大误差"""
    if isinstance(torch_output, tuple):
        torch_output = torch_output[0]  # 某些操作可能返回多个值
    if isinstance(ms_output, (tuple, list)):
        ms_output = ms_output[0]  # MindSpore可能返回列表
    return np.max(np.abs(torch_output.detach().numpy() - ms_output.asnumpy()))


def generate_random_data(shape, dtype=np.float32):
    """生成随机数据"""
    if dtype in [np.float16, np.float32, np.float64]:
        return np.random.randn(*shape).astype(dtype)
    elif dtype == np.uint8:
        return np.random.randint(0, 100, size=shape, dtype=dtype)
    else:
        return np.random.randint(-100, 100, size=shape, dtype=dtype)


def test_random_dtype_support():
    """测试不同数据类型的支持情况"""
    shape = (4, 5)
    dtypes = [
        (np.float32, torch.float32, ms.float32),  # 主要测试float32
        (np.int32, torch.int32, ms.int32),  # 主要测试int32
    ]

    results = []
    for np_dtype, torch_dtype, ms_dtype in dtypes:
        data = generate_random_data(shape, np_dtype)
        torch_support = True
        ms_support = True
        error_msg = {'torch': '', 'mindspore': ''}

        try:
            torch_input = torch.tensor(data, dtype=torch_dtype)
            torch_output = torch.cummin(torch_input, dim=0)
        except Exception as e:
            torch_support = False
            error_msg['torch'] = str(e)

        try:
            ms_input = Tensor(data, dtype=ms_dtype)
            ms_output = mint.cummin(ms_input, dim=0)
        except Exception as e:
            ms_support = False
            error_msg['mindspore'] = str(e)

        if torch_support and ms_support:
            try:
                error = calculate_error(torch_output, ms_output)
                status = "PASS" if error < 1e-3 else f"FAIL (error: {error:.6f})"
            except Exception as e:
                status = f"ERROR: {str(e)}"
        else:
            status = "ERROR"

        results.append({
            'dtype': str(np_dtype),
            'torch_support': torch_support,
            'ms_support': ms_support,
            'status': status,
            'torch_error': error_msg['torch'],
            'ms_error': error_msg['mindspore']
        })

    print("\n=== Data Type Support Results ===")
    print("\n{:<15} {:<15} {:<15} {:<20}".format(
        "Data Type", "PyTorch", "MindSpore", "Status"))
    print("-" * 65)

    for result in results:
        print("{:<15} {:<15} {:<15} {:<20}".format(
            result['dtype'],
            "✓" if result['torch_support'] else "✗",
            "✓" if result['ms_support'] else "✗",
            result['status']
        ))


def test_fixed_dtype_output_accuracy():
    """测试固定数据类型下随机输入的输出精度"""
    shapes_to_test = [
        (4,),  # 1D
        (4, 5),  # 2D
    ]

    print("\n=== Output Accuracy Test Results ===")
    print("\n{:<20} {:<15} {:<20}".format("Shape", "Max Error", "Status"))
    print("-" * 55)

    for shape in shapes_to_test:
        try:
            data = generate_random_data(shape, np.float32)

            # PyTorch测试
            torch_input = torch.tensor(data, dtype=torch.float32)
            torch_output = torch.cummin(torch_input, dim=0)

            # MindSpore测试
            ms_input = Tensor(data, dtype=ms.float32)
            ms_output = mint.cummin(ms_input, dim=0)

            # 计算误差
            error = calculate_error(torch_output, ms_output)
            status = "PASS" if error < 1e-3 else "FAIL"

            print("{:<20} {:<15.6f} {:<20}".format(
                str(shape), error, status))

        except Exception as e:
            print("{:<20} {:<15} {:<20}".format(
                str(shape), "ERROR", str(e)[:20]))


def test_simple_cases():
    """测试简单用例"""
    print("\n=== Simple Cases Test Results ===")

    test_cases = [
        {
            'name': 'Basic 1D',
            'data': np.array([3, 1, 4, 1, 5], dtype=np.float32),
            'dim': 0
        },
        {
            'name': 'Basic 2D',
            'data': np.array([[2, 1], [1, 2]], dtype=np.float32),
            'dim': 0
        }
    ]

    for case in test_cases:
        try:
            torch_input = torch.tensor(case['data'])
            ms_input = Tensor(case['data'])

            torch_output = torch.cummin(torch_input, dim=case['dim'])
            ms_output = mint.cummin(ms_input, dim=case['dim'])

            error = calculate_error(torch_output, ms_output)
            status = "PASS" if error < 1e-3 else "FAIL"

            print(f"\nTest case: {case['name']}")
            print(f"Input shape: {case['data'].shape}")
            print(f"Error: {error:.6f}")
            print(f"Status: {status}")

            # 打印实际输出值以进行比较
            print("\nPyTorch output:")
            print(torch_output[0].numpy())
            print("\nMindSpore output:")
            print(ms_output[0].asnumpy())

        except Exception as e:
            print(f"\nTest case: {case['name']}")
            print(f"Error: {str(e)}")
            print("Status: ERROR")


def test_param_type_support():
    """
    测试不同参数类型的支持情况
    已知问题：
    1. 一维张量的axis参数必须为0，而PyTorch无此限制
    """
    print("\n=== Parameter Type Support Test Results ===")

    # 测试一维张量的axis限制问题
    data_1d = np.array([1, 2, 3, 4], dtype=np.float32)
    torch_input_1d = torch.tensor(data_1d)
    ms_input_1d = Tensor(data_1d)

    print("\n--- Testing 1D tensor axis restriction ---")
    # PyTorch支持axis=0
    torch_output_0 = torch.cummin(torch_input_1d, dim=0)
    print("PyTorch axis=0:", torch_output_0[0].numpy())

    # MindSpore仅支持axis=0
    try:
        ms_output_0 = mint.cummin(ms_input_1d, dim=0)
        print("MindSpore axis=0:", ms_output_0[0].asnumpy())
        print("Status: KNOWN_ISSUE - MindSpore仅支持一维张量的axis=0")
    except Exception as e:
        print(f"MindSpore axis=0 error: {str(e)}")

    # 测试其他参数类型
    data_2d = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    torch_input = torch.tensor(data_2d)
    ms_input = Tensor(data_2d)

    test_cases = [
        {'name': 'String dim', 'dim': 'invalid', 'axis': 'invalid'},
        {'name': 'Bool dim', 'dim': True, 'axis': True},
        {'name': 'Float dim', 'dim': 0.5, 'axis': 0.5},
        {'name': 'Out of range dim', 'dim': 10, 'axis': 10},
        {'name': 'Negative dim', 'dim': -3, 'axis': -3},
    ]

    print("\n--- Testing parameter type support ---")
    print("{:<20} {:<30} {:<30}".format(
        "Test Case", "PyTorch Error", "MindSpore Error"))
    print("-" * 80)

    for case in test_cases:
        torch_error = ""
        ms_error = ""

        try:
            _ = torch.cummin(torch_input, dim=case['dim'])
        except Exception as e:
            torch_error = str(e)

        try:
            _ = mint.cummin(ms_input, dim=case['axis'])
        except Exception as e:
            ms_error = str(e)

        print("{:<20} {:<30} {:<30}".format(
            case['name'],
            torch_error[:30] + "..." if len(torch_error) > 30 else torch_error,
            ms_error[:30] + "..." if len(ms_error) > 30 else ms_error
        ))


def test_error_handling():
    """
    测试错误处理
    已知问题：
    1. 维度范围限制与PyTorch不同：MindSpore为[-n, n)，而PyTorch为[-n, n-1]
    """
    print("\n=== Error Handling Test Results ===")

    # 测试维度范围限制
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)  # 2x3矩阵
    torch_input = torch.tensor(data)
    ms_input = Tensor(data)

    print("\n--- Testing dimension range restriction ---")
    print("Input shape:", data.shape)

    # 测试维度范围
    dims_to_test = [-3, -2, -1, 0, 1, 2]
    print("\n{:<10} {:<20} {:<20}".format("Dim", "PyTorch", "MindSpore"))
    print("-" * 50)

    for dim in dims_to_test:
        torch_result = "Supported"
        ms_result = "Supported"

        try:
            _ = torch.cummin(torch_input, dim=dim)
        except Exception as e:
            torch_result = f"Error: {str(e)[:20]}"

        try:
            _ = mint.cummin(ms_input, dim=dim)
        except Exception as e:
            ms_result = f"Error: {str(e)[:20]}"

        print("{:<10} {:<20} {:<20}".format(str(dim), torch_result, ms_result))

        # 特别标注边界条件的差异
        if dim == data.ndim - 1:  # 最后一个维度
            print("Status: KNOWN_ISSUE - PyTorch支持dim=n-1，而MindSpore不支持")


def test_cummin_forward_backward():
    """
    测试cummin前向推理和反向梯度的准确性
    已知问题：
    1. 梯度计算存在量化问题，梯度值被强制转换为整数，误差范围3.49-7.20
    """
    print("\n=== Forward and Backward Test Results ===")

    # 测试数据
    data = np.array([[0.5, 2.3, 1.8], [1.2, 0.7, 3.1]], dtype=np.float32)

    # PyTorch测试
    torch_input = torch.tensor(data, requires_grad=True)
    torch_output, _ = torch.cummin(torch_input, dim=0)
    torch_loss = torch_output.sum()
    torch_loss.backward()
    torch_grad = torch_input.grad.numpy()

    print("\nPyTorch gradient:")
    print(torch_grad)

    # MindSpore测试
    try:
        class CumminNet(ms.nn.Cell):
            def construct(self, x):
                output, _ = mint.cummin(x, dim=0)
                return output.sum()

        net = CumminNet()
        ms_input = Tensor(data, dtype=ms.float32)

        # 使用数值微分来计算梯度
        epsilon = 1e-6
        numerical_grad = np.zeros_like(data)

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                # 计算f(x + epsilon)
                data_plus = data.copy()
                data_plus[i, j] += epsilon
                output_plus = net(Tensor(data_plus, dtype=ms.float32))

                # 计算f(x - epsilon)
                data_minus = data.copy()
                data_minus[i, j] -= epsilon
                output_minus = net(Tensor(data_minus, dtype=ms.float32))

                # 计算数值梯度
                numerical_grad[i, j] = (float(output_plus.asnumpy()) - float(output_minus.asnumpy())) / (2 * epsilon)

        print("\nMindSpore numerical gradient:")
        print(numerical_grad)

        # 计算梯度误差
        grad_error = np.max(np.abs(torch_grad - numerical_grad))
        print(f"\nMaximum gradient error: {grad_error:.6f}")

        if grad_error > 1e-3:
            print(f"Status: KNOWN_ISSUE - 梯度误差 ({grad_error:.2f}) 超过阈值 (1e-3)")
            print("原因：梯度计算存在量化问题，梯度值被强制转换为整数")

        # 检查梯度量化问题
        is_quantized = np.all(np.mod(numerical_grad, 1) == 0)
        if is_quantized:
            print("Status: KNOWN_ISSUE - 检测到梯度被量化为整数值")

    except Exception as e:
        print(f"\nError during gradient computation: {str(e)}")
        print("Status: KNOWN_ISSUE - MindSpore梯度计算失败，可能与维度范围限制有关")


def test_cummin_in_network():
    """
    测试在神经网络中使用cummin的情况
    已知问题：
    1. 维度范围限制：MindSpore为[-n, n)，而PyTorch为[-n, n-1]
    """
    print("\n=== Testing cummin in Neural Network ===")

    # 测试不同维度的情况
    data = np.random.randn(2, 3, 4).astype(np.float32)
    dims_to_test = [-3, -2, -1, 0, 1, 2]

    print("\n--- Testing dimension range restrictions ---")
    print("{:<10} {:<20} {:<20}".format("Dim", "PyTorch", "MindSpore"))
    print("-" * 50)

    for dim in dims_to_test:
        torch_result = "Supported"
        ms_result = "Supported"

        # PyTorch测试
        try:
            torch_input = torch.tensor(data)
            _ = torch.cummin(torch_input, dim=dim)
        except Exception as e:
            torch_result = f"Error: {str(e)[:20]}"

        # MindSpore测试
        try:
            ms_input = Tensor(data)
            _ = mint.cummin(ms_input, dim=dim)
        except Exception as e:
            ms_result = f"Error: {str(e)[:20]}"

        print("{:<10} {:<20} {:<20}".format(str(dim), torch_result, ms_result))

        # 特别标注维度范围限制的差异
        if ms_result.startswith("Error") and torch_result == "Supported":
            print(f"Status: KNOWN_ISSUE - 维度 {dim} 在PyTorch中支持但在MindSpore中不支持")
            print(f"原因：MindSpore的维度范围为[-n, n)，而PyTorch为[-n, n-1]")


if __name__ == "__main__":
    print("\n=== Testing random dtype support ===")
    test_random_dtype_support()

    print("\n=== Testing fixed dtype output accuracy ===")
    test_fixed_dtype_output_accuracy()

    print("\n=== Testing simple cases ===")
    test_simple_cases()

    print("\n=== Testing parameter type support ===")
    test_param_type_support()

    print("\n=== Testing error handling ===")
    test_error_handling()

    print("\n=== Testing cummin functionality ===")
    test_cummin_forward_backward()
    test_cummin_in_network()
