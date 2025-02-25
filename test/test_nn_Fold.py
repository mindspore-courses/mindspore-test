import numpy as np
import torch
import torch.nn as tnn
import mindspore as ms
from mindspore import Tensor, context, ops
import mindspore.mint.nn as mint_nn

def compare_fold_dtypes():
    """
    测试 mindspore.mint.nn.Fold 和 torch.nn.Fold 在不同数据类型下的支持度。
    使用随机输入张量，适配 nn.Fold 的参数。
    """
    # 定义测试的数据类型列表
    dtype_list = [
        (ms.float16, torch.float16), (ms.float32, torch.float32), (ms.float64, torch.float64),
        (ms.int8, torch.int8), (ms.int16, torch.int16), (ms.int32, torch.int32), (ms.int64, torch.int64),
        (ms.uint8, torch.uint8), (ms.uint16, torch.uint16), (ms.uint32, torch.uint32), (ms.uint64, torch.uint64),
        (ms.bool_, torch.bool),
    ]

    # 设置 MindSpore 上下文，使用 PYNATIVE_MODE（Ascend 支持）
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")

    # 生成随机输入数据，假设从 Unfold 输出而来
    # 例如：batch_size=1, channels=2, kernel_size=(2, 2), output_size=(4, 4), stride=(2, 2)
    # Unfold 输出形状：(batch_size, channels * kernel_size[0] * kernel_size[1], L)
    # 这里 L = (H_out * W_out)，H_out = W_out = 2
    input_np = np.random.randn(1, 8, 4).astype(np.float32)  # (1, 2*2*2, 2*2)

    print("开始测试 Fold 在不同数据类型下的支持情况...")
    print(f"输入形状: {input_np.shape}")
    print(f"输入数据样例:\n{input_np}")

    # 定义 Fold 参数
    output_size = (4, 4)  # 输出特征图大小
    kernel_size = (2, 2)  # 卷积核大小
    stride = (2, 2)      # 步幅
    dilation = (1, 1)    # 膨胀
    padding = (0, 0)     # 填充

    # 计算期望输出形状
    batch_size, _, L = input_np.shape
    expected_shape = (batch_size, 2, output_size[0], output_size[1])  # (1, 2, 4, 4)
    print(f"\nFold 参数:")
    print(f"  output_size: {output_size}")
    print(f"  kernel_size: {kernel_size}")
    print(f"  stride: {stride}")
    print(f"  dilation: {dilation}")
    print(f"  padding: {padding}")
    print(f"  预期输出形状: {expected_shape}")

    # 初始化 Fold 操作
    fold_mint = mint_nn.Fold(output_size=output_size, kernel_size=kernel_size, stride=stride, 
                            dilation=dilation, padding=padding)
    fold_torch = tnn.Fold(output_size=output_size, kernel_size=kernel_size, stride=stride, 
                          dilation=dilation, padding=padding)

    # 测试每种数据类型
    for ms_dtype, torch_dtype in dtype_list:
        print(f"\n--- 测试数据类型: MindSpore {ms_dtype}, PyTorch {torch_dtype} ---")

        # MindSpore mint.nn.Fold 测试
        try:
            input_ms = Tensor(input_np, dtype=ms_dtype)
            output_mint = fold_mint(input_ms)
            print(f"MindSpore mint.nn.Fold 输出形状: {output_mint.shape}, dtype: {output_mint.dtype}")
            try:
                print(f"输出样例:\n{output_mint.asnumpy()[:2]}")  # 打印部分输出
            except Exception as e:
                print(f"MindSpore没有报错直到调用了output_mint")
                print(f"MindSpore mint.nn.Fold 不支持 {ms_dtype}")
                print(f"错误信息: {type(e).__name__}: {str(e)[:50]}")
        except Exception as e:
            print(f"MindSpore mint.nn.Fold 不支持 {ms_dtype}")
            print(f"错误信息: {type(e).__name__}: {str(e)[:50]}")

        # PyTorch nn.Fold 测试
        try:
            input_torch = torch.tensor(input_np, dtype=torch_dtype)
            output_torch = fold_torch(input_torch)
            print(f"PyTorch nn.Fold 输出形状: {output_torch.shape}, dtype: {output_torch.dtype}")
            print(f"输出样例:\n{output_torch.numpy()[:2]}")  # 打印部分输出
        except Exception as e:
            print(f"PyTorch nn.Fold 不支持 {torch_dtype}")
            print(f"错误信息: {type(e).__name__}: {str(e)[:50]}")

            
            
def compare_fold_accuracy(L=1024, channels=500, dtype=ms.float32, torch_dtype=torch.float32, test_cycles=2):
    """
    测试固定 dtype，随机输入值，对比 mindspore.mint.nn.Fold 和 torch.nn.Fold 输出是否相等。
    误差范围小于 1e-3。L 可由用户指定，其他参数根据数学关系自动计算。
    参数:
        L: 输入张量的块数大小，可自由设置
        channels: 通道数，默认为 500
        dtype: MindSpore 数据类型，固定为 float32
        torch_dtype: PyTorch 数据类型，固定为 torch.float32
        test_cycles: 测试轮次，默认为 3
    """
    # 设置 MindSpore 上下文，使用 PYNATIVE_MODE
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")

    # 设置随机种子以确保可重复性
    np.random.seed(2025)
    torch.manual_seed(2025)
    ms.set_seed(2025)

    # 定义固定的 Fold 参数
    kernel_size = (2, 2)    # 卷积核大小
    stride = (1, 1)        # 步幅
    dilation = (1, 1)      # 膨胀
    padding = (0, 0)       # 填充

    # 根据 L 和参数计算 output_size
    # L = [(H - kernel_height + 2 * padding_height - dilation_height * (kernel_height - 1)) / stride_height + 1] 
    #     * [(W - kernel_width + 2 * padding_width - dilation_width * (kernel_width - 1)) / stride_width + 1]
    # 假设 H = W，且 stride=1, padding=0, dilation=1，则：
    # L = [(H - 2 + 1) * (H - 2 + 1)] = (H - 1)^2
    # H - 1 = sqrt(L)
    # H = sqrt(L) + 1
    H = int(np.sqrt(L)) + 1
    output_size = (H, H)

    # 输入形状
    batch_size = 1
    C = channels * kernel_size[0] * kernel_size[1]  # C = channels * kernel_height * kernel_width
    shape = (batch_size, C, L)  # (1, channels * 4, L)

    # 初始化 Fold 操作
    fold_ms = mint_nn.Fold(output_size=output_size, kernel_size=kernel_size, stride=stride, 
                           dilation=dilation, padding=padding)
    fold_torch = tnn.Fold(output_size=output_size, kernel_size=kernel_size, stride=stride, 
                          dilation=dilation, padding=padding)

    # 计算预期输出形状
    expected_shape = (batch_size, channels, output_size[0], output_size[1])
    print(f"\n开始测试固定 dtype ({dtype}, {torch_dtype}) 的输出一致性...")
    print(f"输入形状: {shape}")
    print(f"Fold 参数: output_size={output_size}, kernel_size={kernel_size}, stride={stride}, "
          f"dilation={dilation}, padding={padding}")
    print(f"预期输出形状: {expected_shape}")

    # 验证 L 是否与参数匹配
    computed_L = ((output_size[0] - kernel_size[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1)) // stride[0] + 1) * \
                 ((output_size[1] - kernel_size[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1)) // stride[1] + 1)
    print(f"根据参数计算得到的 L: {computed_L}，输入指定的 L: {L}")
    if computed_L != L:
        print(f"警告：输入的 L={L} 与计算得到的 L={computed_L} 不匹配，可能导致错误")

    # 多轮测试
    for cycle in range(test_cycles):
        print(f"\n--- 测试轮次 {cycle + 1}/{test_cycles} ---")

        # 生成随机输入
        input_np = np.random.randn(*shape).astype(np.float32)
        input_ms = Tensor(input_np, dtype=dtype)
        input_torch = torch.tensor(input_np, dtype=torch_dtype)

        # MindSpore mint.nn.Fold 计算
        ms_output = None
        ms_error = None
        try:
            ms_output = fold_ms(input_ms)
            print(f"MindSpore mint.nn.Fold 输出形状: {ms_output.shape}, dtype: {ms_output.dtype}")
            print(f"MindSpore 输出样例:\n{ms_output.asnumpy()[0, :2, :5, :5]}")
        except Exception as e:
            ms_error = f"MindSpore 错误: {type(e).__name__}: {str(e)[:100]}"
            print(f"MindSpore mint.nn.Fold 不支持此计算")
            print(ms_error)

        # PyTorch nn.Fold 计算
        torch_output = None
        torch_error = None
        try:
            torch_output = fold_torch(input_torch)
            print(f"PyTorch nn.Fold 输出形状: {torch_output.shape}, dtype: {torch_output.dtype}")
            print(f"PyTorch 输出样例:\n{torch_output.numpy()[0, :2, :5, :5]}")
        except Exception as e:
            torch_error = f"PyTorch 错误: {type(e).__name__}: {str(e)[:100]}"
            print(f"PyTorch nn.Fold 不支持此计算")
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

            # 计算最大绝对误差
            abs_diff = np.abs(ms_out_np - torch_out_np)
            max_error = np.max(abs_diff)
            print(f"最大绝对误差: {max_error:.6f}")

            # 判断是否在误差范围内
            if max_error < 1e-3:
                print("✅ 输出一致 (最大误差 < 1e-3)")
            else:
                print(f"❌ 输出不一致 (最大误差 = {max_error:.6f})")
                print(f"MindSpore 输出样例: {ms_out_np[0, 0, :5, :5].flatten()[:5]}")
                print(f"PyTorch 输出样例: {torch_out_np[0, 0, :5, :5].flatten()[:5]}")
        else:
            print("对比失败：一方或双方报错")
            if ms_error:
                print(ms_error)
            if torch_error:
                print(torch_error)
                
                
def test_fold_parameter_support():
    """
    测试固定 shape 和输入值，不同输入参数（string、bool 等类型），对比 mindspore.mint.nn.Fold 和 torch.nn.Fold 的支持度。
    """
    # 设置 MindSpore 上下文，使用 PYNATIVE_MODE，适配 Ascend 910B
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")

    # 固定输入张量，shape 为 (1, 8, 4)，适配 Fold 输入
    # 例如：batch_size=1, channels=2, kernel_size=(2, 2), output_size=(4, 4), stride=(2, 2)
    input_np = np.array([[[1., 2., 3., 4.],
                          [5., 6., 7., 8.],
                          [9., 10., 11., 12.],
                          [13., 14., 15., 16.],
                          [17., 18., 19., 20.],
                          [21., 22., 23., 24.],
                          [25., 26., 27., 28.],
                          [29., 30., 31., 32.]]]).astype(np.float32)  # (1, 8, 4)

    print("开始测试不同输入参数的支持度...")
    print(f"固定输入形状: {input_np.shape}")
    print(f"固定输入数据:\n{input_np}")

    # 定义测试用例（包括合法和非法参数）
    test_cases = [
        {"output_size": (4, 4), "kernel_size": (2, 2), "stride": (2, 2), "dilation": (1, 1), "padding": (0, 0), "desc": "合法参数（默认）"},
        {"output_size": 4, "kernel_size": 2, "stride": 2, "dilation": 1, "padding": 0, "desc": "单值合法参数"},
        {"output_size": "invalid", "kernel_size": (2, 2), "stride": (2, 2), "dilation": (1, 1), "padding": (0, 0), "desc": "output_size 为字符串"},
        {"output_size": (4, 4), "kernel_size": "invalid", "stride": (2, 2), "dilation": (1, 1), "padding": (0, 0), "desc": "kernel_size 为字符串"},
        {"output_size": (4, 4), "kernel_size": (2, 2), "stride": "invalid", "dilation": (1, 1), "padding": (0, 0), "desc": "stride 为字符串"},
        {"output_size": (4, 4), "kernel_size": (2, 2), "stride": (2, 2), "dilation": "invalid", "padding": (0, 0), "desc": "dilation 为字符串"},
        {"output_size": (4, 4), "kernel_size": (2, 2), "stride": (2, 2), "dilation": (1, 1), "padding": "invalid", "desc": "padding 为字符串"},
        {"output_size": True, "kernel_size": (2, 2), "stride": (2, 2), "dilation": (1, 1), "padding": (0, 0), "desc": "output_size 为布尔值"},
        {"output_size": (4, 4), "kernel_size": True, "stride": (2, 2), "dilation": (1, 1), "padding": (0, 0), "desc": "kernel_size 为布尔值"},
        {"output_size": (4, 4), "kernel_size": (2, 2), "stride": True, "dilation": (1, 1), "padding": (0, 0), "desc": "stride 为布尔值"},
        {"output_size": (4, 4), "kernel_size": (2, 2), "stride": (2, 2), "dilation": True, "padding": (0, 0), "desc": "dilation 为布尔值"},
        {"output_size": (4, 4), "kernel_size": (2, 2), "stride": (2, 2), "dilation": (1, 1), "padding": True, "desc": "padding 为布尔值"},
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
            "torch_output": None
        }

        # MindSpore mint.nn.Fold 测试
        try:
            input_ms = Tensor(input_np, dtype=ms.float32)
            if "invalid_param" in case:
                fold_ms = mint_nn.Fold(invalid_param=case["invalid_param"])
            else:
                fold_ms = mint_nn.Fold(
                    output_size=case.get("output_size", (4, 4)),
                    kernel_size=case.get("kernel_size", (2, 2)),
                    stride=case.get("stride", (2, 2)),
                    dilation=case.get("dilation", (1, 1)),
                    padding=case.get("padding", (0, 0))
                )
            output_ms = fold_ms(input_ms)
            try:
                case_result["ms_output"] = output_ms.asnumpy()
                print(f"MindSpore mint.nn.Fold 输出结果:\n{case_result['ms_output']}")
            except Exception as e:
                print('MindSpore没有报错直到调用了output_ms！')
                case_result["ms_support"] = False
                case_result["ms_error"] = f"{type(e).__name__}: {str(e)[:100]}"
                print(f"MindSpore mint.nn.Fold 不支持此参数配置")
                print(f"错误信息: {case_result['ms_error']}")
        except Exception as e:
            case_result["ms_support"] = False
            case_result["ms_error"] = f"{type(e).__name__}: {str(e)[:100]}"
            print(f"MindSpore mint.nn.Fold 不支持此参数配置")
            print(f"错误信息: {case_result['ms_error']}")

        # PyTorch nn.Fold 测试
        try:
            input_torch = torch.tensor(input_np, dtype=torch.float32)
            if "invalid_param" in case:
                fold_torch = tnn.Fold(invalid_param=case["invalid_param"])
            else:
                fold_torch = tnn.Fold(
                    output_size=case.get("output_size", (4, 4)),
                    kernel_size=case.get("kernel_size", (2, 2)),
                    stride=case.get("stride", (2, 2)),
                    dilation=case.get("dilation", (1, 1)),
                    padding=case.get("padding", (0, 0))
                )
            output_torch = fold_torch(input_torch)
            case_result["torch_output"] = output_torch.numpy()
            print(f"PyTorch nn.Fold 输出结果:\n{case_result['torch_output']}")
        except Exception as e:
            case_result["torch_support"] = False
            case_result["torch_error"] = f"{type(e).__name__}: {str(e)[:100]}"
            print(f"PyTorch nn.Fold 不支持此参数配置")
            print(f"错误信息: {case_result['torch_error']}")

        results.append(case_result)

    # 打印测试报告
    print("\n=== 测试报告 ===")
    print("{:<35} {:<15} {:<15} {:<40} {:<40}".format(
        "测试场景", "MindSpore 支持", "PyTorch 支持", "MindSpore 错误", "PyTorch 错误"
    ))
    print("-" * 145)
    for res in results:
        print("{:<35} {:<15} {:<15} {:<40} {:<40}".format(
            res["desc"][:30],
            "✓" if res["ms_support"] else "✗",
            "✓" if res["torch_support"] else "✗",
            res["ms_error"][:35],
            res["torch_error"][:35],
        ))
        
        
def test_fold_error_handling():
    """
    测试 mindspore.mint.nn.Fold 和 torch.nn.Fold 对随机混乱输入的异常处理能力，对比报错信息的准确性。
    """
    # 设置 MindSpore 上下文，使用 PYNATIVE_MODE，适配 Ascend 910B
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")

    # 定义默认的 Fold 参数
    output_size = (4, 4)
    kernel_size = (2, 2)
    stride = (2, 2)
    dilation = (1, 1)
    padding = (0, 0)

    # 初始化 Fold 操作
    fold_ms = mint_nn.Fold(output_size=output_size, kernel_size=kernel_size, stride=stride, 
                           dilation=dilation, padding=padding)
    fold_torch = tnn.Fold(output_size=output_size, kernel_size=kernel_size, stride=stride, 
                          dilation=dilation, padding=padding)

    # 定义测试用例
    test_cases = [
        # 用例 1: 非张量输入（Python 列表）
        {
            "desc": "非张量输入（Python 列表）",
            "input_ms": [[1.0, 2.0], [3.0, 4.0]],
            "input_torch": [[1.0, 2.0], [3.0, 4.0]]
        },
        # 用例 2: 0 维张量
        {
            "desc": "0 维张量",
            "input_ms": Tensor(42.0, dtype=ms.float32),
            "input_torch": torch.tensor(42.0, dtype=torch.float32)
        },
        # 用例 3: 1 维张量
        {
            "desc": "1 维张量",
            "input_ms": Tensor(np.random.randn(6), dtype=ms.float32),
            "input_torch": torch.randn(6)
        },
        # 用例 4: 2 维张量
        {
            "desc": "2 维张量",
            "input_ms": Tensor(np.random.randn(2, 6), dtype=ms.float32),
            "input_torch": torch.randn(2, 6)
        },
        # 用例 5: 5 维张量
        {
            "desc": "5 维张量",
            "input_ms": Tensor(np.random.randn(1, 2, 3, 4, 5), dtype=ms.float32),
            "input_torch": torch.randn(1, 2, 3, 4, 5)
        },
        # 用例 6: 包含 NaN 的张量
        {
            "desc": "包含 NaN 的张量",
            "input_ms": Tensor(np.array([[[1.0, np.nan, 3.0, 4.0]]]), dtype=ms.float32),
            "input_torch": torch.tensor([[[1.0, float('nan'), 3.0, 4.0]]], dtype=torch.float32)
        },
        # 用例 7: 包含 Inf 的张量
        {
            "desc": "包含 Inf 的张量",
            "input_ms": Tensor(np.array([[[1.0, np.inf, 3.0, 4.0]]]), dtype=ms.float32),
            "input_torch": torch.tensor([[[1.0, float('inf'), 3.0, 4.0]]], dtype=torch.float32)
        },
        # 用例 8: 维度过小
        {
            "desc": "维度过小",
            "input_ms": Tensor(np.ones((1, 1, 2, 2)), dtype=ms.float32),
            "input_torch": torch.ones(1, 1, 2, 2)
        },
        # 用例 9: 输入类型不匹配
        {
            "desc": "输入类型不匹配",
            "input_ms": Tensor(np.ones((1, 8, 4), dtype=np.int32), dtype=ms.int32),
            "input_torch": torch.ones(1, 8, 4, dtype=torch.int32)
        },
        # 用例 10: 布尔类型
        {
            "desc": "隐布尔tensor输出",
            "input_ms": Tensor(np.array([[[True,False]]])),
            "input_torch": torch.tensor([[[True,False]]])
        },
        # 用例 11: 布尔类型
        {
            "desc": "显布尔tensor输出",
            "input_ms": Tensor(np.array([[[True,False]]]),dtype=ms.bool_),
            "input_torch": torch.tensor([[[True,False]]],dtype=torch.bool)
        }
    ]

    print("\n==================== 测试随机混乱输入，报错信息的准确性 ====================")

    for case in test_cases:
        print(f"\n▌ 测试场景：{case['desc']}")
        print("=" * 90)

        # MindSpore 测试
        ms_result = None
        ms_error = None
        try:
            ms_result = fold_ms(case["input_ms"])
            print(f"[MindSpore 通过] 输出形状: {ms_result.shape}")
            try:
                print(f"输出样例:\n{ms_result.asnumpy()[:2]}")
            except Exception as e:
                print('MindSpore直到将ms_result输出才报错！')
                ms_error = f"{type(e).__name__}: {str(e)[:100]}"
                print(f"[MindSpore 异常] 错误类型：{type(e).__name__}")
                print(f"异常详情：{str(e).splitlines()[0]}")
        except Exception as e:
            ms_error = f"{type(e).__name__}: {str(e)[:100]}"
            print(f"[MindSpore 异常] 错误类型：{type(e).__name__}")
            print(f"异常详情：{str(e).splitlines()[0]}")

        # PyTorch 测试
        torch_result = None
        torch_error = None
        try:
            torch_result = fold_torch(case["input_torch"])
            print(f"[PyTorch 通过] 输出形状: {tuple(torch_result.shape)}")
            print(f"输出样例:\n{torch_result.numpy()[:2]}")
        except Exception as e:
            torch_error = f"{type(e).__name__}: {str(e)[:100]}"
            print(f"[PyTorch 异常] 错误类型：{type(e).__name__}")
            print(f"异常详情：{str(e).splitlines()[0]}")

        # 对比结果
        if ms_error and torch_error:
            print("\n※ 状态对比 ※ | 双方均触发异常")
            print(f"  MindSpore 错误: {ms_error}")
            print(f"  PyTorch 错误: {torch_error}")
        elif ms_error:
            print("\n※ 状态对比 ※ | MindSpore 触发异常，PyTorch 未触发")
            print(f"  MindSpore 错误: {ms_error}")
        elif torch_error:
            print("\n※ 状态对比 ※ | PyTorch 触发异常，MindSpore 未触发")
            print(f"  PyTorch 错误: {torch_error}")
        else:
            print("\n※ 状态对比 ※ | 双方均计算成功")
            # 可选：对比输出结果（视需求而定）
            ms_out_np = ms_result.asnumpy()
            torch_out_np = torch_result.numpy()
            abs_diff = np.abs(ms_out_np - torch_out_np)
            max_error = np.max(abs_diff)
            print(f"最大绝对误差: {max_error:.6f}")
            if max_error < 1e-3:
                print("✅ 输出一致 (最大误差 < 1e-3)")
            else:
                print(f"❌ 输出不一致 (最大误差 = {max_error:.6f})")

        print("=" * 90)

def test_fold_forward_and_backward():
    """
    测试 mindspore.mint.nn.Fold 和 torch.nn.Fold 的正向推理和反向梯度准确性。
    输入为随机生成，块数千级别（1024），边长接近百级别（64），数值范围为 (MAX_FLOAT32/8, MAX_FLOAT32/2)。
    """
    # 设置 MindSpore 上下文，使用 PYNATIVE_MODE，适配 Ascend 910B
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")

    # 设置随机种子以确保可重复性
    np.random.seed(2025)
    torch.manual_seed(2025)
    ms.set_seed(2025)

    # 定义参数
    batch_size = 1
    channels = 500           # 通道数
    kernel_size = (2, 2)     # 卷积核大小
    stride = (2, 2)          # 步幅
    dilation = (1, 1)        # 膨胀
    padding = (0, 0)         # 填充
    L = 1024                 # 块数，千级别
    H = 64                   # 输出边长，接近百级别
    output_size = (H, H)     # 输出特征图大小 (64, 64)

    # 计算输入形状
    C = channels * kernel_size[0] * kernel_size[1]  # C = 500 * 2 * 2 = 2000
    input_shape = (batch_size, C, L)  # (1, 2000, 1024)

    # 定义随机数范围
    MAX_FLOAT32 = np.finfo(np.float32).max  # 约 3.40282e38
    min_val = MAX_FLOAT32 / 8
    max_val = MAX_FLOAT32 / 2

    # 生成随机输入
    input_np = np.random.uniform(low=min_val, high=max_val, size=input_shape).astype(np.float32)
    input_np[0][0][0]=0;input_np[0][0][1]=0;
    print(f"\n输入形状: {input_shape}")
    print(f"输入数据范围: [{min_val:.2e}, {max_val:.2e}]")
    print(f"输入样例:\n{input_np[0, :2, :5]}")  # 打印部分输入

    # MindSpore 输入（启用梯度）
    input_ms = Tensor(input_np, dtype=ms.float32)
    input_ms.requires_grad = True

    # PyTorch 输入（启用梯度）
    input_torch = torch.tensor(input_np, dtype=torch.float32, requires_grad=True)

    # 初始化 Fold 操作
    fold_ms = mint_nn.Fold(output_size=output_size, kernel_size=kernel_size, stride=stride, 
                           dilation=dilation, padding=padding)
    fold_torch = tnn.Fold(output_size=output_size, kernel_size=kernel_size, stride=stride, 
                          dilation=dilation, padding=padding)

    # --- 2.b: 正向推理测试 ---
    print("\n--- 正向推理测试 ---")

    # MindSpore 正向
    ms_output = None
    ms_error = None
    try:
        ms_output = fold_ms(input_ms)
        print(f"MindSpore mint.nn.Fold 输出形状: {ms_output.shape}")
        print(f"MindSpore 输出样例:\n{ms_output.asnumpy()[0, :2, :5, :5]}")
    except Exception as e:
        ms_error = f"MindSpore 错误: {type(e).__name__}: {str(e)[:100]}"
        print("MindSpore mint.nn.Fold 推理失败")
        print(ms_error)

    # PyTorch 正向
    torch_output = None
    torch_error = None
    try:
        torch_output = fold_torch(input_torch)
        print(f"PyTorch nn.Fold 输出形状: {torch_output.shape}")
        print(f"PyTorch 输出样例:\n{torch_output.detach().numpy()[0, :2, :5, :5]}")
    except Exception as e:
        torch_error = f"PyTorch 错误: {type(e).__name__}: {str(e)[:100]}"
        print("PyTorch nn.Fold 推理失败")
        print(torch_error)

    # 正向对比
    if ms_output is not None and torch_output is not None:
        ms_out_np = ms_output.asnumpy()
        torch_out_np = torch_output.detach().numpy()
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

    # --- 2.c: 反向传播测试 ---
    print("\n--- 反向传播测试 ---")

    # MindSpore 反向
    ms_grad = None
    ms_backward_error = None
    if ms_output is not None:
        try:
            # 定义损失并计算输入梯度
            grad_fn = ops.GradOperation(get_all=True)(lambda x: ops.reduce_sum(fold_ms(x)))
            ms_grad = grad_fn(input_ms)[0].asnumpy()
            print(f"MindSpore 输入梯度形状: {ms_grad.shape}")
            print(f"MindSpore 输入梯度样例:\n{ms_grad[0, :2, :5]}")
        except Exception as e:
            ms_backward_error = f"MindSpore 反向错误: {type(e).__name__}: {str(e)[:100]}"
            print("MindSpore mint.nn.Fold 反向失败")
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
            print(f"PyTorch 输入梯度形状: {torch_grad.shape}")
            print(f"PyTorch 输入梯度样例:\n{torch_grad[0, :2, :5]}")
        except Exception as e:
            torch_backward_error = f"PyTorch 反向错误: {type(e).__name__}: {str(e)[:100]}"
            print("PyTorch nn.Fold 反向失败")
            print(torch_backward_error)

    # 反向对比
    if ms_grad is not None and torch_grad is not None:
        if ms_grad.shape != torch_grad.shape:
            print(f"梯度形状不一致: MindSpore {ms_grad.shape}, PyTorch {torch_grad.shape}")
        else:
            grad_diff = np.abs(ms_grad - torch_grad)
            sum_grad_error = np.sum(grad_diff)
            print(f"\n反向绝对误差和: {sum_grad_error:.6f}")
            if sum_grad_error < 1e-3:
                print("✅ 反向梯度结果一致 (误差 < 1e-3)")
            else:
                print(f"❌ 反向梯度结果不一致 (误差 = {sum_grad_error:.6f})")
    else:
        print("\n反向对比失败：一方或双方报错")
        if ms_backward_error:
            print(ms_backward_error)
        if torch_backward_error:
            print(torch_backward_error)

    print("\n=== 测试完成 ===")
if __name__ == "__main__":
    
    print('\n====================测试random输入不同dtype，对比两个框架的支持度。目前issues：3====================')
    compare_fold_dtypes()
    
    print('\n====================测试固定dtype，random输入值，对比两个框架输出是否相等（误差范围为小于1e-3）。目前issues：0====================')
    compare_fold_accuracy()
    
    print('\n====================测试固定shape，固定输入值，不同输入参数（string\bool等类型），两个框架的支持度。目前issues：0====================')
    test_fold_parameter_support()
    
    print('\n====================测试随机混乱输入，报错信息的准确性。目前issues：1====================')
    test_fold_error_handling()
    
    print('\n====================使用Pytorch和MindSpore，固定输入和权重，测试正向推理结果（误差范围小于1e-3，若报错则记录报错信息）测试该神经网络/函数反向，如果是神经网络，则测试Parameter的梯度，如果是函数，则测试函数输入的梯度。目前issues：0====================')
    test_fold_forward_and_backward()
    
    print('===============================All TestingTasks done! 总issues：4==============================')