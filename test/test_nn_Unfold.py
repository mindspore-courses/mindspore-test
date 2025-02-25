import numpy as np
import torch
import torch.nn as tnn
import mindspore as ms
from mindspore import Tensor, context, ops
import mindspore.nn as nn
import mindspore.mint.nn as mint_nn

def compare_unfold_dtypes():
    """
    测试 mindspore.nn.Unfold、mindspore.mint.nn.Unfold 和 torch.nn.Unfold 在不同数据类型下的支持度。
    使用固定输入张量，适配 nn.Fold 的参数。
    """
    # 定义测试的数据类型列表
    dtype_list = [
        (ms.float16, torch.float16),(ms.float32, torch.float32),(ms.float64, torch.float64),
        (ms.int8, torch.int8),(ms.int16, torch.int16),(ms.int32, torch.int32),(ms.int64, torch.int64),
        (ms.uint8, torch.uint8),(ms.uint16, torch.uint16),(ms.uint32, torch.uint32),(ms.uint64, torch.uint64),
        (ms.bool_, torch.bool),
    ]

    # 设置 MindSpore 上下文，使用 PYNATIVE_MODE
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")

    # 固定输入张量，扩展为 (1, 2, 6, 6)
    input_np = np.array([
        [
            [1, 2, 3, 4, 5, 6], 
            [10, 11, 12, 13, 14, 15], 
            [19, 20, 21, 22, 23, 24], 
            [28, 29, 30, 31, 32, 33], 
            [37, 38, 39, 40, 41, 42], 
            [46, 47, 48, 49, 50, 51]
        ],
        [
            [7, 8, 9, 10, 11, 12], 
            [16, 17, 18, 19, 20, 21], 
            [25, 26, 27, 28, 29, 30], 
            [34, 35, 36, 37, 38, 39], 
            [43, 44, 45, 46, 47, 48], 
            [52, 53, 54, 55, 56, 57]
        ]
    ]).astype(np.float32)  # (2, 6, 6)
    input_np = input_np[np.newaxis, ...]  # (1, 2, 6, 6)

    print("开始测试 Unfold 在不同数据类型下的支持情况...")
    print(f"输入形状: {input_np.shape}")
    print(f"输入数据:\n{input_np}")

    # 定义 Unfold 参数，与 nn.Fold 匹配
    # nn.Fold: output_size=(6, 6), kernel_size=(2, 2), stride=(2, 2)
    # Unfold 对应参数
    ksizes = [1, 2, 2, 1]  # nn.Unfold
    strides = [1, 2, 2, 1]
    rates = [1, 1, 1, 1]
    padding_nn = "valid"

    kernel_size = (2, 2)  # mint.nn.Unfold 和 torch.nn.Unfold
    stride = (2, 2)
    dilation = (1, 1)
    padding_mint = (0, 0)

    # 计算期望输出形状
    batch_size, channels, height, width = input_np.shape
    out_height = (height - kernel_size[0]) // stride[0] + 1  # valid 模式
    out_width = (width - kernel_size[1]) // stride[1] + 1
    expected_shape = (batch_size, channels * kernel_size[0] * kernel_size[1], out_height * out_width)
    # (1, 2*2*2, 3*3) = (1, 8, 9)

    # 输出适配参数
    print(f"\n适配的 Unfold 参数:")
    print(f"  MindSpore nn.Unfold:")
    print(f"    ksizes: {ksizes}")
    print(f"    strides: {strides}")
    print(f"    rates: {rates}")
    print(f"    padding: {padding_nn}")
    print(f"  MindSpore mint.nn.Unfold & PyTorch nn.Unfold:")
    print(f"    kernel_size: {kernel_size}")
    print(f"    stride: {stride}")
    print(f"    dilation: {dilation}")
    print(f"    padding: {padding_mint}")
    print(f"  预期输出形状: {expected_shape}")

    # 初始化三种 Unfold 操作
    unfold_ms = nn.Unfold(ksizes=ksizes, strides=strides, rates=rates, padding=padding_nn)
    unfold_mint = mint_nn.Unfold(kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding_mint)
    unfold_torch = tnn.Unfold(kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding_mint)

    # 测试每种数据类型
    for ms_dtype, torch_dtype in dtype_list:
        print(f"\n--- 测试数据类型: MindSpore {ms_dtype}, PyTorch {torch_dtype} ---")

        # MindSpore nn.Unfold 测试
        try:
            input_ms = Tensor(input_np, dtype=ms_dtype)
            output_ms = unfold_ms(input_ms)
            print(f"MindSpore nn.Unfold 输出形状: {output_ms.shape}, dtype: {output_ms.dtype}")
        except Exception as e:
            print(f"MindSpore nn.Unfold 不支持 {ms_dtype}")
            print(f"错误信息: {type(e).__name__}: {str(e)[:50]}")

        # MindSpore mint.nn.Unfold 测试
        try:
            input_ms = Tensor(input_np, dtype=ms_dtype)
            output_mint = unfold_mint(input_ms)
            print(f"MindSpore mint.nn.Unfold 输出形状: {output_mint.shape}, dtype: {output_mint.dtype}")
            try:
                print(f'output_mint:{output_mint.asnumpy()}')
            except Exception as e:
                print(f"MindSpore mint.nn.Unfold：没有主动抛出错误，直到调用了output！错误信息：{type(e).__name__}: {str(e)[:40]}")
        except Exception as e:
            print(f"MindSpore mint.nn.Unfold 不支持 {ms_dtype}")
            print(f"错误信息: {type(e).__name__}: {str(e)[:50]}")

        # PyTorch nn.Unfold 测试
        try:
            input_torch = torch.tensor(input_np, dtype=torch_dtype)
            output_torch = unfold_torch(input_torch)
            print(f"PyTorch nn.Unfold 输出形状: {output_torch.shape}, dtype: {output_torch.dtype}")
            try:
                print(f'output_torch:{output_torch.numpy()}')
            except Exception as e:
                print(f"PyTorch nn.Unfold：没有主动抛出错误，直到调用了output！错误信息：{type(e).__name__}: {str(e)[:40]}")
        except Exception as e:
            print(f"PyTorch nn.Unfold 不支持 {torch_dtype}")
            print(f"错误信息: {type(e).__name__}: {str(e)[:50]}")

            
def compare_unfold_accuracy(shape=(1, 3, 100, 100), dtype=ms.float32, torch_dtype=torch.float32, test_cycles=3):
    """
    测试固定 dtype，随机输入值，对比 mindspore.mint.nn.Unfold 和 torch.nn.Unfold 输出是否相等。
    误差范围小于 1e-3。
    参数:
        shape: 输入张量形状，调整为 (1, 3, 100, 100)
        dtype: MindSpore 数据类型，固定为 float32
        torch_dtype: PyTorch 数据类型，固定为 torch.float32
        test_cycles: 测试轮次，默认为 3
    """
    # 设置 MindSpore 上下文，使用 PYNATIVE_MODE
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")

    # 设置随机种子
    np.random.seed(2025)
    torch.manual_seed(2025)
    ms.set_seed(2025)

    # 定义 float32 最大值
    MAX_FLOAT32 = np.finfo(np.float32).max  # 约 3.40282e38
    MIN_VAL = MAX_FLOAT32 / 100
    MAX_VAL = MAX_FLOAT32 * 0.95

    # 定义 Unfold 参数
    kernel_size = (3, 3)
    dilation = (1, 1)
    padding = (0, 0)
    stride = (1, 1)

    # 初始化 Unfold 操作
    unfold_ms = mint_nn.Unfold(kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride)
    unfold_torch = tnn.Unfold(kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride)

    print(f"\n开始测试固定 dtype ({dtype}, {torch_dtype}) 的输出一致性...")
    print(f"输入形状: {shape}")
    print(f"随机数据范围: [{MIN_VAL:.2e}, {MAX_VAL:.2e}]")
    print(f"Unfold 参数: kernel_size={kernel_size}, dilation={dilation}, padding={padding}, stride={stride}")

    # 计算预期输出形状
    batch_size, channels, height, width = shape
    out_height = (height + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
    out_width = (width + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1
    expected_shape = (batch_size, channels * kernel_size[0] * kernel_size[1], out_height * out_width)
    print(f"预期输出形状: {expected_shape}")

    # 多轮测试
    for cycle in range(test_cycles):
        print(f"\n--- 测试轮次 {cycle + 1}/{test_cycles} ---")

        # 生成靠近 float32 最大值的随机输入
        input_np = np.random.uniform(low=MIN_VAL, high=MAX_VAL, size=shape).astype(np.float32)
        input_ms = Tensor(input_np, dtype=dtype)
        input_torch = torch.tensor(input_np, dtype=torch_dtype)

        # MindSpore mint.nn.Unfold 计算
        ms_output = None
        ms_error = None
        try:
            ms_output = unfold_ms(input_ms)
            print(f"MindSpore mint.nn.Unfold 输出形状: {ms_output.shape}, dtype: {ms_output.dtype}")
            try:
                print(f"MindSpore mint.nn.Unfold 输出结果: \n{ms_output.asnumpy()}")
            except Exception as e: 
                print('MindSpore mint.nn.Unfold 输出结果触发异常')
        except Exception as e:
            ms_error = f"MindSpore 错误: {type(e).__name__}: {str(e)[:100]}"
            print(f"MindSpore mint.nn.Unfold 不支持此计算")
            print(ms_error)

        # PyTorch nn.Unfold 计算
        torch_output = None
        torch_error = None
        try:
            torch_output = unfold_torch(input_torch)
            print(f"PyTorch nn.Unfold 输出形状: {torch_output.shape}, dtype: {torch_output.dtype}")
            try:
                print(f"PyTorch nn.Unfold 输出结果: \n{torch_output.numpy()}")
            except Exception as e: 
                print('PyTorch nn.Unfold 输出结果触发异常')
        except Exception as e:
            torch_error = f"PyTorch 错误: {type(e).__name__}: {str(e)[:100]}"
            print(f"PyTorch nn.Unfold 不支持此计算")
            print(torch_error)

        # 对比输出
        if ms_output is not None and torch_output is not None:
            # 转换为 numpy 进行比较
            ms_out_np = ms_output.asnumpy()
            torch_out_np = torch_output.numpy()

            # 检查形状是否一致
            if ms_out_np.shape != torch_out_np.shape:
                print(f"形状不一致: MindSpore {ms_out_np.shape}, PyTorch {torch_out_np.shape}")
                continue

            # 计算所有元素的绝对误差之和
            abs_diff = np.abs(ms_out_np - torch_out_np)
            total_error = np.sum(abs_diff)
            print(f"总绝对误差: {total_error:.6f}")

            # 判断是否在误差范围内
            if total_error < 1e-3:
                print("✅ 输出一致 (总误差 < 1e-3)")
            else:
                print(f"❌ 输出不一致 (总误差 = {total_error:.6f})")
                print(f"MindSpore 输出样例: {ms_out_np.flatten()[:5]}")
                print(f"PyTorch 输出样例: {torch_out_np.flatten()[:5]}")
        else:
            print("对比失败：一方或双方报错")
            if ms_error:
                print(ms_error)
            if torch_error:
                print(torch_error)
                
                
def test_unfold_parameter_support():
    """
    测试固定 shape 和输入值，不同输入参数（string、bool 等类型），对比 mindspore.mint.nn.Unfold 和 torch.nn.Unfold 的支持度。
    输出结果并附带误差值。
    """
    # 设置 MindSpore 上下文，使用 PYNATIVE_MODE
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")

    # 固定输入张量，shape 为 (1, 2, 6, 6)
    input_np = np.array([
        [
            [1, 2, 3, 4, 5, 6], 
            [7, 8, 9, 10, 11, 12], 
            [13, 14, 15, 16, 17, 18], 
            [19, 20, 21, 22, 23, 24], 
            [25, 26, 27, 28, 29, 30], 
            [31, 32, 33, 34, 35, 36]
        ],
        [
            [37, 38, 39, 40, 41, 42], 
            [43, 44, 45, 46, 47, 48], 
            [49, 50, 51, 52, 53, 54], 
            [55, 56, 57, 58, 59, 60], 
            [61, 62, 63, 64, 65, 66], 
            [67, 68, 69, 70, 71, 72]
        ]
    ]).astype(np.float32)  # (2, 6, 6)
    input_np = input_np[np.newaxis, ...]  # (1, 2, 6, 6)

    print("开始测试不同输入参数的支持度...")
    print(f"固定输入形状: {input_np.shape}")
    print(f"固定输入数据:\n{input_np}")

    # 定义测试用例（包括合法和非法参数）
    test_cases = [
        {"kernel_size": (3, 3), "dilation": (1, 1), "padding": (0, 0), "stride": (1, 1), "desc": "合法参数（默认）"},
        {"kernel_size": 3, "dilation": 1, "padding": 0, "stride": 1, "desc": "单值合法参数"},
        {"kernel_size": "invalid", "dilation": (1, 1), "padding": (0, 0), "stride": (1, 1), "desc": "kernel_size 为字符串"},
        {"kernel_size": (3, 3), "dilation": "invalid", "padding": (0, 0), "stride": (1, 1), "desc": "dilation 为字符串"},
        {"kernel_size": (3, 3), "dilation": (1, 1), "padding": "invalid", "stride": (1, 1), "desc": "padding 为字符串"},
        {"kernel_size": (3, 3), "dilation": (1, 1), "padding": (0, 0), "stride": True, "desc": "stride 为布尔值"},
        {"kernel_size": True, "dilation": (1, 1), "padding": (0, 0), "stride": (1, 1), "desc": "kernel_size 为布尔值"},
        {"kernel_size": (3, 3), "dilation": False, "padding": (0, 0), "stride": (1, 1), "desc": "dilation 为布尔值"},
        {"kernel_size": (3, 3), "dilation": (1, 1), "padding": None, "stride": (1, 1), "desc": "padding 为 None"},
        {"kernel_size": (3, 3), "dilation": (1, 1), "padding": (0, 0), "stride": None, "desc": "stride 为 None"},
        {"invalid_param": 1, "desc": "非法参数名"}
    ]

    # 测试结果列表
    results = []

    for case in test_cases:
        print(f"\n测试场景: {case['desc']}")
        case_result = {
            "desc": case["desc"],
            "params": case,
            "ms_support": True,
            "torch_support": True,
            "ms_error": "",
            "torch_error": "",
            "ms_output": None,
            "torch_output": None,
            "error_value": None
        }

        # MindSpore mint.nn.Unfold 测试
        try:
            input_ms = Tensor(input_np, dtype=ms.float32)
            if "invalid_param" in case:
                unfold_ms = mint_nn.Unfold(invalid_param=case["invalid_param"])
            else:
                unfold_ms = mint_nn.Unfold(
                    kernel_size=case.get("kernel_size", (3, 3)),
                    dilation=case.get("dilation", (1, 1)),
                    padding=case.get("padding", (0, 0)),
                    stride=case.get("stride", (1, 1))
                )
            output_ms = unfold_ms(input_ms)
            case_result["ms_output"] = output_ms.asnumpy()
            print(f"MindSpore mint.nn.Unfold 输出结果:\n{case_result['ms_output']}")
        except Exception as e:
            case_result["ms_support"] = False
            case_result["ms_error"] = f"{type(e).__name__}"
            print(f"MindSpore mint.nn.Unfold 不支持此参数配置")
            str_e=str(e).split('\n', 1)[0]
            print(f"错误信息: {case_result['ms_error']}:{str_e}")

        # PyTorch nn.Unfold 测试
        try:
            input_torch = torch.tensor(input_np, dtype=torch.float32)
            if "invalid_param" in case:
                unfold_torch = tnn.Unfold(invalid_param=case["invalid_param"])
            else:
                unfold_torch = tnn.Unfold(
                    kernel_size=case.get("kernel_size", (3, 3)),
                    dilation=case.get("dilation", (1, 1)),
                    padding=case.get("padding", (0, 0)),
                    stride=case.get("stride", (1, 1))
                )
            output_torch = unfold_torch(input_torch)
            case_result["torch_output"] = output_torch.numpy()
            print(f"PyTorch nn.Unfold 输出结果:\n{case_result['torch_output']}")
        except Exception as e:
            case_result["torch_support"] = False
            case_result["torch_error"] = f"{type(e).__name__}"
            print(f"PyTorch nn.Unfold 不支持此参数配置")
            str_e=str(e).split('\n', 1)[0]
            print(f"错误信息: {case_result['torch_error']}:{str_e}")

        # 计算误差（如果两者均支持）
        if case_result["ms_support"] and case_result["torch_support"]:
            abs_diff = np.abs(case_result["ms_output"] - case_result["torch_output"])
            max_error = np.max(abs_diff)
            case_result["error_value"] = max_error
            print(f"最大绝对误差: {max_error:.6f}")

        results.append(case_result)

    # 打印测试报告
    print("\n=== 测试报告 ===")
    print("{:<35} {:<15} {:<15} {:<40} {:<40}".format(
        "测试场景", "MindSpore 支持", "PyTorch 支持", "MindSpore 错误", "PyTorch 错误"
    ))
    print("-" * 160)
    for res in results:
        print("{:<35} {:<15} {:<15} {:<40} {:<40}".format(
            res["desc"][:30],
            "✓" if res["ms_support"] else "✗",
            "✓" if res["torch_support"] else "✗",
            res["ms_error"][:35],
            res["torch_error"][:35],
        ))
        
        
        
def test_unfold_error_cases():
    """测试 mindspore.mint.nn.Unfold 和 torch.nn.Unfold 对随机混乱输入的异常处理能力，对比报错信息的准确性。"""
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    np.random.seed(42); torch.manual_seed(42); ms.set_seed(42)
    default_params = {"kernel_size": (3, 3), "dilation": (1, 1), "padding": (0, 0), "stride": (1, 1)}
    print("\n==================== Unfold 异常测试开始 ====================")

    test_cases = [
        ("输入为非张量（Python 列表）", [[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]]),
        ("输入为 0 维张量", Tensor(42.0, dtype=ms.float32), torch.tensor(42.0, dtype=torch.float32)),
        ("输入为 1 维张量", Tensor(np.random.randn(6), dtype=ms.float32), torch.randn(6)),
        ("输入为 2 维张量", Tensor(np.random.randn(2, 6), dtype=ms.float32), torch.randn(2, 6)),
        ("输入为 5 维张量", Tensor(np.random.randn(1, 2, 3, 4, 5), dtype=ms.float32), torch.randn(1, 2, 3, 4, 5)),
        ("输入包含 NaN", Tensor(np.array([[[[1.0, np.nan, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]], dtype=np.float32)), torch.tensor([[[[1.0, float('nan'), 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]])),
        ("输入包含 Inf", Tensor(np.array([[[[1.0, np.inf, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]], dtype=np.float32)), torch.tensor([[[[1.0, float('inf'), 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]])),
        ("输入维度过小", Tensor(np.ones((1, 1, 2, 2)), dtype=ms.float32), torch.ones(1, 1, 2, 2)),
        ("输入类型不匹配", Tensor(np.ones((1, 1, 6, 6), dtype=np.int32), dtype=ms.int32), torch.ones(1, 1, 6, 6, dtype=torch.int32))
    ]

    def run_test(desc, ms_input, torch_input):
        print(f"\n▌ 测试场景：{desc}\n{'=' * 90}")
        ms_result_b, torch_result_b = False, False
        try:
            ms_result = mint_nn.Unfold(**default_params)(ms_input)
            try:
                print(f"[MindSpore 通过] 输出结果：\n{ms_result.asnumpy()}"); ms_result_b = True
            except Exception as e:
                print(f"MindSpore没有报错直到将ms_result输出！[MindSpore 异常] 错误类型：{type(e).__name__}\n异常详情：{str(e).splitlines()[0]}")
        except Exception as e:
            print(f"[MindSpore 异常] 错误类型：{type(e).__name__}\n异常详情：{str(e).splitlines()[0]}")
        try:
            torch_result = tnn.Unfold(**default_params)(torch_input)
            print(f"[PyTorch 通过] 输出结果：\n{torch_result.numpy()}"); torch_result_b = True
        except Exception as e:
            print(f"[PyTorch 异常] 错误类型：{type(e).__name__}\n异常详情：{str(e).splitlines()[0]}")
        print(f"\n※ 状态对比 ※ | {'双方均计算成功' if ms_result_b and torch_result_b else '双方均触发异常' if not ms_result_b and not torch_result_b else '一方计算成功一方失败'}")
        if ms_result_b and torch_result_b:
            max_error = np.max(np.abs(ms_result.asnumpy() - torch_result.numpy()))
            print(f"最大绝对误差：{max_error:.6f}\n{'✓ 结果一致' if max_error < 1e-6 else '⚠ 结果存在差异'}")
        print("=" * 90)

    for desc, ms_input, torch_input in test_cases:
        run_test(desc, ms_input, torch_input)

    print(f"\n▌ 测试场景：输入形状不规则（非矩形）\n{'=' * 90}")
    irregular_input = [[[1.0, 2.0], [3.0]]]
    try:
        run_test("输入形状不规则", Tensor(irregular_input, dtype=ms.float32), torch.tensor(irregular_input))
    except Exception as e:
        print(f"[MindSpore 异常] 错误类型：{type(e).__name__}\n异常详情：{str(e).splitlines()[0]}")
        print(f"[PyTorch 异常] 错误类型：{type(e).__name__}\n异常详情：{str(e).splitlines()[0]}")
        print("\n※ 状态对比 ※ | 双方均触发异常\n" + "=" * 90)
        
        
def test_unfold_forward_and_backward():
    """
    测试 mindspore.mint.nn.Unfold 和 torch.nn.Unfold 的正向推理和反向梯度准确性。
    b) 固定输入和权重，测试正向推理结果（误差 < 1e-3）。
    c) 测试函数反向，计算输入的梯度。
    """
    # 设置 MindSpore 上下文，使用 PYNATIVE_MODE
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")

    # 固定输入张量，shape 为 (1, 2, 6, 6)
    input_np = np.array([
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 
            [7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 
            [13.0, 14.0, 15.0, 16.0, 17.0, 18.0], 
            [19.0, 20.0, 21.0, 22.0, 23.0, 24.0], 
            [25.0, 26.0, 27.0, 28.0, 29.0, 30.0], 
            [31.0, 32.0, 33.0, 34.0, 35.0, 36.0]
        ],
        [
            [37.0, 38.0, 39.0, 40.0, 41.0, 42.0], 
            [43.0, 44.0, 45.0, 46.0, 47.0, 48.0], 
            [49.0, 50.0, 51.0, 52.0, 53.0, 54.0], 
            [55.0, 56.0, 57.0, 58.0, 59.0, 60.0], 
            [61.0, 62.0, 63.0, 64.0, 65.0, 66.0], 
            [67.0, 68.0, 69.0, 70.0, 71.0, 72.0]
        ]
    ]).astype(np.float32)  # (2, 6, 6)
    input_np = input_np[np.newaxis, ...]  # (1, 2, 6, 6)

    # 设置固定的 Unfold 参数（权重）
    kernel_size = (3, 3)
    dilation = (1, 1)
    padding = (0, 0)
    stride = (1, 1)

    # 计算预期输出形状
    batch_size, channels, height, width = input_np.shape
    out_height = (height + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
    out_width = (width + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1
    expected_shape = (batch_size, channels * kernel_size[0] * kernel_size[1], out_height * out_width)

    print("\n=== 测试 mindspore.mint.nn.Unfold 和 torch.nn.Unfold 正向推理和反向梯度准确性 ===")
    print(f"固定输入形状: {input_np.shape}")
    print(f"固定输入数据:\n{input_np}")
    print(f"Unfold 参数: kernel_size={kernel_size}, dilation={dilation}, padding={padding}, stride={stride}")
    print(f"预期输出形状: {expected_shape}")

    # MindSpore 输入
    input_ms = Tensor(input_np, dtype=ms.float32)
    # PyTorch 输入（需要梯度）
    input_torch = torch.tensor(input_np, dtype=torch.float32, requires_grad=True)

    # 初始化 Unfold 操作
    unfold_ms = mint_nn.Unfold(kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride)
    unfold_torch = tnn.Unfold(kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride)

    # --- 子任务 b: 正向推理测试 ---
    print("\n--- 正向推理测试 ---")

    # MindSpore 正向
    ms_output = None
    ms_error = None
    try:
        ms_output = unfold_ms(input_ms)
        print(f"MindSpore mint.nn.Unfold 输出结果:\n{ms_output.asnumpy()}")
    except Exception as e:
        ms_error = f"MindSpore 错误: {type(e).__name__}: {str(e)[:100]}"
        print(f"MindSpore mint.nn.Unfold 推理失败")
        print(ms_error)

    # PyTorch 正向
    torch_output = None
    torch_error = None
    try:
        torch_output = unfold_torch(input_torch)
        print(f"PyTorch nn.Unfold 输出结果:\n{torch_output.detach().numpy()}")
    except Exception as e:
        torch_error = f"PyTorch 错误: {type(e).__name__}: {str(e)[:100]}"
        print(f"PyTorch nn.Unfold 推理失败")
        print(torch_error)

    # 正向对比
    if ms_output is not None and torch_output is not None:
        ms_out_np = ms_output.asnumpy()
        torch_out_np = torch_output.detach().numpy()  # 使用 detach() 移除梯度
        if ms_out_np.shape != torch_out_np.shape:
            print(f"形状不一致: MindSpore {ms_out_np.shape}, PyTorch {torch_out_np.shape}")
        else:
            abs_diff = np.abs(ms_out_np - torch_out_np)
            max_error = np.max(abs_diff)
            print(f"\n正向最大绝对误差: {max_error:.6f}")
            if max_error < 1e-3:
                print("✅ 正向推理结果一致 (误差 < 1e-3)")
            else:
                print(f"❌ 正向推理结果不一致 (误差 = {max_error:.6f})")
    else:
        print("\n正向对比失败：一方或双方报错")
        if ms_error:
            print(ms_error)
        if torch_error:
            print(torch_error)

    # --- 子任务 c: 反向传播测试 ---
    print("\n--- 反向传播测试 ---")

    # MindSpore 反向
    ms_grad = None
    ms_backward_error = None
    if ms_output is not None:
        try:
            # 定义损失并计算输入梯度
            loss_ms = ops.reduce_sum(ms_output)
            grad_fn = ops.GradOperation(get_all=True)(lambda x: ops.reduce_sum(unfold_ms(x)))
            ms_grad = grad_fn(input_ms)[0].asnumpy()
            print(f"MindSpore 输入梯度:\n{ms_grad}")
        except Exception as e:
            ms_backward_error = f"MindSpore 反向错误: {type(e).__name__}: {str(e)[:100]}"
            print(f"MindSpore mint.nn.Unfold 反向失败")
            print(ms_backward_error)

    # PyTorch 反向
    torch_grad = None
    torch_backward_error = None
    if torch_output is not None:
        try:
            # 定义损失并计算输入梯度
            loss_torch = torch.sum(torch_output)
            loss_torch.backward()
            torch_grad = input_torch.grad.numpy()
            print(f"PyTorch 输入梯度:\n{torch_grad}")
        except Exception as e:
            torch_backward_error = f"PyTorch 反向错误: {type(e).__name__}: {str(e)[:100]}"
            print(f"PyTorch nn.Unfold 反向失败")
            print(torch_backward_error)

    # 反向对比
    if ms_grad is not None and torch_grad is not None:
        if ms_grad.shape != torch_grad.shape:
            print(f"梯度形状不一致: MindSpore {ms_grad.shape}, PyTorch {torch_grad.shape}")
        else:
            grad_diff = np.abs(ms_grad - torch_grad)
            max_grad_error = np.max(grad_diff)
            print(f"\n反向最大绝对误差: {max_grad_error:.6f}")
            if max_grad_error < 1e-3:
                print("✅ 反向梯度结果一致 (误差 < 1e-3)")
            else:
                print(f"❌ 反向梯度结果不一致 (误差 = {max_grad_error:.6f})")
    else:
        print("\n反向对比失败：一方或双方报错")
        if ms_backward_error:
            print(ms_backward_error)
        if torch_backward_error:
            print(torch_backward_error)

    print("\n=== 测试完成 ===")
    
if __name__ == "__main__":
    
    print('\n====================测试random输入不同dtype，对比两个框架的支持度。目前issues：3====================')
    compare_unfold_dtypes()
    print('\n==================== 上一个测试结束 。经测试 不支持图模式。====================')
    
    
    
    print('\n====================测试固定dtype，random输入值，对比两个框架输出是否相等（误差范围为小于1e-3）。目前issues：0====================')
    compare_unfold_accuracy()
    print('\n==================== 上一个测试结束 ====================')
    
    
    print('\n====================测试固定shape，固定输入值，不同输入参数（string\bool等类型），两个框架的支持度。目前issues：0====================')
    test_unfold_parameter_support()
    print('\n==================== 上一个测试结束 ====================')
    
    
    print('\n====================测试随机混乱输入，报错信息的准确性。目前issues：1====================')
    test_unfold_error_cases()
    print('\n==================== 上一个测试结束 ====================')
    
    
    print('\n====================使用Pytorch和MindSpore，固定输入和权重，测试正向推理结果（误差范围小于1e-3，若报错则记录报错信息）测试该神经网络/函数反向，如果是神经网络，则测试Parameter的梯度，如果是函数，则测试函数输入的梯度。目前issues：0====================')
    test_unfold_forward_and_backward()
    
    
    print('===============================All TestingTasks done! 总issues：4==============================')