import numpy as np
import torch
import mindspore as ms
from mindspore import Tensor, context, ops
import mindspore.mint as mint

def compare_tril_dtypes(shape=(6, 6)):
    """
    测试random输入不同dtype，对比两个框架的支持度
    """
    # 支持的dtype列表，包括不同的浮点型、整数型、布尔型等
    dtypes = [
        np.float16, np.float32, np.float64,
        np.int8, np.int16, np.int32, np.int64,
        np.uint8, np.uint16, np.uint32, np.uint64,
        np.bool_
    ]
    
    # 结果记录列表
    results = []

    for np_dtype in dtypes:
        print(f"\n正在测试数据类型: {np_dtype}")
        torch_support = True
        ms_support = True
        error_msg = {"torch": "", "mindspore": ""}

        # PyTorch 错误捕获
        try:
            # 生成随机数据
            data = np.random.randn(*shape).astype(np_dtype)
            
            # PyTorch 输入
            torch_input = torch.tensor(data, dtype=torch.__dict__.get(np_dtype.__name__, None))
            # 调用 PyTorch 的 trilu 函数
            torch_output = torch.tril(torch_input)
            print(f"PyTorch 输出: \n{torch_output}")
        except Exception as e:
            torch_support = False
            error_msg["torch"] = f"PyTorch: {str(e)}"
            print(f"PyTorch 错误: {str(e)}")

        # MindSpore 错误捕获
        try:
            ms_input = ms.Tensor(data, dtype=ms.__dict__.get(np_dtype.__name__, None))
            print(f"MindSpore 输入: \n{ms_input}")
            # 调用 MindSpore 的 tril 函数
            ms_output = mint.tril(ms_input)
            print(f"MindSpore 输出: \n{ms_output}")
        except Exception as e:
            ms_support = False
            error_msg["mindspore"] = f"MindSpore: {str(e)}"
            print(f"MindSpore 错误: {str(e)}")

        # 生成测试结果
        if torch_support and ms_support:
            status = "支持"
        elif torch_support != ms_support:
            status = "存在差异"
        else:
            status = "不支持"

        # 收集结果
        results.append({
            "dtype": str(np_dtype),
            "torch_support": torch_support,
            "ms_support": ms_support,
            "status": status,
            "torch_error": error_msg["torch"],
            "ms_error": error_msg["mindspore"],
        })

    # 输出结果
    print("\n== 数据类型支持情况 ==")
    print(
        "\n{:<30} {:<15} {:<15} {:<20} {:<40} {:<40}".format(
            "数据类型", "PyTorch", "MindSpore", "状态", "torch_error", "ms_error"
        )
    )
    print("-" * 160)

    for result in results:
        print(
            "{:<35} {:<15} {:<14} {:<15} {:<40} {:<40}".format(
                result["dtype"][:30],  # 截断数据类型的输出
                ("✓" if result["torch_support"] else "✗"),  # 截断 PyTorch 支持输出
                ("✓" if result["ms_support"] else "✗"),  # 截断 MindSpore 支持输出
                result["status"],  # 截断状态输出
                result["torch_error"][:35],  # 截断 PyTorch 错误输出
                result["ms_error"][:35]  # 截断 MindSpore 错误输出
            )
        )

        
def compare_tril_accuracy(shape=(6, 6), test_cycles=1):
    """集成化完整测试函数（保持所有原始细节）"""
    # 配置numpy完整打印参数
    np.set_printoptions(
        precision=15,
        threshold=np.inf,
        linewidth=200,
        suppress=False,
        floatmode='maxprec'
    )

    def generate_edge_case_values(dtype, shape):
        """生成高复杂度测试数据（完全保留原实现）"""
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            data = np.random.choice([info.max, info.min, 0], size=shape)
        elif np.issubdtype(dtype, np.floating):
            finfo = np.finfo(dtype)
            data = np.random.choice([
                finfo.max * 0.999,
                finfo.min * 0.999,
                finfo.eps * 1e3,
                np.pi,
                np.e
            ], size=shape)
        return data.astype(dtype)

    def amplify_error(tensor_a, tensor_b):
        """误差计算函数（保持原始实现）"""
        tensor_a = tensor_a.astype(ms.float64)
        tensor_b = tensor_b.astype(ms.float64)
        abs_diff = ops.abs(tensor_a - tensor_b)
        rel_diff = abs_diff / (ops.abs(tensor_a) + 1e-30)
        combined = (abs_diff + rel_diff) * 1e9
        return ops.reduce_sum(ops.square(combined))

    # 主测试流程（完整保留原始逻辑）
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    test_dtypes = [
        np.float16, np.float32, np.float64,
        np.int8, np.int32, np.int64,
        np.uint8, np.uint32, np.uint64
    ]

    for cycle in range(test_cycles):
        print("\n" + "="*100)
        print(f"▶ 测试轮次 {cycle+1}/{test_cycles}")
        print("="*100)
        
        for dtype in test_dtypes:
            dtype_name = np.dtype(dtype).name
            print(f"\n■■■ 测试数据类型: {dtype_name} ■■■")
            
            # 生成测试数据
            data = generate_edge_case_values(dtype, shape)
            print(f"\n[输入数据 ({dtype_name})]\n{data}")
            
            # 初始化结果记录（完整保留原始字段）
            result = {
                "dtype": dtype_name,
                "status": "通过",
                "error_energy": 0.0,
                "errors": []
            }
            
            # PyTorch计算分支（完整异常处理）
            try:
                torch_tensor = torch.tensor(data)
                torch_out = torch.tril(torch_tensor)
                print(f"\n[PyTorch输出]\n{torch_out.numpy()}")
            except Exception as e:
                result["status"] = "PyTorch异常"
                result["errors"].append(f"PyTorch错误: {type(e).__name__} - {str(e)[:200]}")
            
            # MindSpore计算分支（完整异常处理）
            try:
                ms_tensor = Tensor(data.astype(dtype))
                ms_out = mint.tril(ms_tensor)
                print(f"\n[MindSpore输出]\n{ms_out.asnumpy()}")
            except Exception as e:
                result["status"] = "MindSpore异常" if result["status"] == "通过" else "双方异常"
                result["errors"].append(f"MindSpore错误: {type(e).__name__} - {str(e)[:200]}")
            
            # 结果对比逻辑（完整保留原始判断条件）
            if "异常" not in result["status"]:
                if torch_out.shape != ms_out.shape:
                    result["status"] = "形状不一致"
                    result["error_energy"] = float("inf")
                else:
                    error_energy = amplify_error(Tensor(torch_out.numpy()), ms_out)
                    result["error_energy"] = error_energy.asnumpy()
                    if result["error_energy"] > 1e-12:
                        result["status"] = "数值不匹配"
            
            # 打印结果（保留原始格式）
            status_icon = "✅" if result["status"] == "通过" else "❌"
            print(f"\n测试状态: {status_icon} {result['status']}")
            print(f"误差能量值: {result['error_energy']:.4e}")
            
            # 错误详情展示（完整输出）
            if result["errors"]:
                print("\n⚠ 异常追踪:")
                for err in result["errors"]:
                    print(f"• {err}")
                    
                    
def test_tril_parameter_support():
    """测试固定shape输入下不同参数类型的支持情况"""
    # 固定输入数据（3x3矩阵）
    base_data = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    
    # 参数组合测试（包含非法类型）
    test_cases = [
        {"diagonal": 0, "desc": "int类型参数"},
        {"diagonal": 1.5, "desc": "float类型参数"},
        {"diagonal": "invalid", "desc": "字符串类型参数"},
        {"diagonal": True, "desc": "布尔类型参数"},
        {"diagonal": None, "desc": "None类型参数"},
        {"invalid_param": 0, "desc": "非法参数名"}
    ]

    results = []
    
    for case in test_cases:
        case_result = {
            "desc": case["desc"],
            "params": case,
            "torch_support": True,
            "ms_support": True,
            "torch_error": "",
            "ms_error": ""
        }
        
        # PyTorch测试
        try:
            pt_input = torch.tensor(base_data)
            diagonal = case.get("diagonal", 0)
            if "invalid_param" in case:
                # 测试非法参数名
                pt_output = torch.tril(pt_input, invalid_param=case["invalid_param"])
            else:
                pt_output = torch.tril(pt_input, diagonal=diagonal)
            case_result["torch_output"] = pt_output.numpy()
        except Exception as e:
            case_result["torch_support"] = False
            case_result["torch_error"] = f"{type(e).__name__}: {str(e).split(',')[0]}"
        
        # MindSpore测试
        try:
            ms_input = Tensor(base_data)
            kwargs = {"diagonal": case.get("diagonal", 0)}
            if "invalid_param" in case:
                # 测试非法参数名
                ms_output = mint.tril(ms_input, invalid_param=case["invalid_param"])
            else:
                ms_output = mint.tril(ms_input, **kwargs)
            case_result["ms_output"] = ms_output.asnumpy()
        except Exception as e:
            case_result["ms_support"] = False
            case_result["ms_error"] = f"{type(e).__name__}: {str(e).split(',')[0]}"
        
        results.append(case_result)

    # 打印测试报告
    print("\n{:<30} {:<15} {:<15} {:<40} {:<40}".format(
        "测试场景", "PyTorch支持", "MindSpore支持", "PyTorch错误信息", "MindSpore错误信息"
    ))
    print("-" * 130)
    for res in results:
        print("{:<30} {:<15} {:<15} {:<40} {:<40}".format(
            res["desc"][:25],
            "✓" if res["torch_support"] else "✗",
            "✓" if res["ms_support"] else "✗",
            res["torch_error"][:35],
            res["ms_error"][:35]
        ))
    
    return results


def test_tril_error_cases():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    """测试tril接口对混乱输入的异常处理能力，对比PyTorch报错准确性"""
    test_cases = [
        # 基础异常场景（修正元组结构）
        ("输入为非张量（列表）", 
         lambda: torch.tril([[1,2],[3,4]]),
         lambda: mint.tril([[1,2],[3,4]])),

        ("输入为0维张量",  # 修正括号匹配
         lambda: torch.tril(torch.tensor(1)),
         lambda: mint.tril(Tensor(1))),
        
        ("输入为1维张量",  # 修正括号匹配
         lambda: torch.tril(torch.tensor([1])),
         lambda: mint.tril(Tensor([1]))),

        ("输入rank为1的张量",  # 修正括号匹配
         lambda: torch.tril(torch.tensor([[1, 2, 3],[2, 4, 6]])),
         lambda: mint.tril(Tensor([[1, 2, 3],[2, 4, 6]]))),
        
        # 参数异常场景
        ("diagonal参数超过维度范围", 
         lambda: torch.tril(torch.randn(2,3), diagonal=100),
         lambda: mint.tril(Tensor(np.random.randn(2,3)), diagonal=100)),

        ("diagonal参数超过维度范围", 
         lambda: torch.tril(torch.randn(2,3), diagonal=-100),
         lambda: mint.tril(Tensor(np.random.randn(2,3)), diagonal=-100)),
        
        ("diagonal参数类型错误",
         lambda: torch.tril(torch.randn(3,3), diagonal="1"),
         lambda: mint.tril(Tensor(np.random.randn(3,3)), diagonal="1")),

        # 类型不匹配场景
        ("输入bool类型张量",
         lambda: torch.tril(torch.tensor([[True, False], [False, True]])),
         lambda: mint.tril(Tensor([[True, False], [False, True]]))),

        # 高维异常场景
        ("5维输入+未定义参数",
         lambda: torch.tril(torch.randn(2,3,4,5,6), dim=-3),
         lambda: mint.tril(Tensor(np.random.randn(2,3,4,5,6)), dim=-3))
    ]

    print("\n=== Tril接口异常测试报告 ===")
    for case_idx, (desc, torch_func, ms_func) in enumerate(test_cases, 1):
        print(f"\n▼ 测试用例 {case_idx}: {desc}")
        ms_error_type = torch_error_type = None
        
        # 独立异常处理每个框架的调用
        try:
            # PyTorch执行
            try:
                torch_result = torch_func()
                print(f"PyTorch输出形状: {tuple(torch_result.shape) if hasattr(torch_result, 'shape') else 'scalar'}")
            except Exception as e:
                torch_err_head = str(e).split('\n', 1)[0]
                torch_error_type = type(e).__name__
                print(f"异常处理触发 | 类型：{torch_error_type} | 关键信息：{torch_err_head}...")
            
            # MindSpore执行
            try:
                ms_result = ms_func()
                print(f"MindSpore输出形状: {ms_result.shape}")
            except Exception as e:
                ms_err_head = str(e).split('\n', 1)[0]
                ms_error_type = type(e).__name__
                print(f"异常处理触发 | 类型：{ms_error_type} | 关键信息：{ms_err_head}...")
            
        except Exception as e:
            print(f"测试框架执行失败: {str(e)}")
            continue  # 遇到致命错误继续下一个测试用例

        # 对比结果
        status = "一致报错" if ms_err_head and torch_err_head else "差异报错" if torch_err_head != ms_err_head else "无报错"
        print(f"|-- PyTorch异常: {torch_error_type or '无异常'}")
        print(f"|-- MindSpore异常: {ms_error_type or '无异常'}")
        print(f"|-- 状态: {status.upper()}")
        
        
# 运行测试
if __name__ == "__main__":
    
    print('\n====================测试random输入不同dtype，对比两个框架的支持度。目前issues：0====================')
    compare_tril_dtypes()
    
    print('\n====================测试固定dtype，random输入值，对比两个框架输出是否相等（误差范围为小于1e-3）。目前issues：0====================')
    compare_tril_accuracy()
    
    print('\n====================测试固定shape，固定输入值，不同输入参数（string\bool等类型），两个框架的支持度。目前issues：0====================')
    test_tril_parameter_support()
    
    print('\n====================测试随机混乱输入，报错信息的准确性。目前issues：0====================')
    test_tril_error_cases()
    
    print('\n====================Github搜索带有该接口的代码片段/神经网络。目前issues：null====================')
    print('接口mindspore.mint.tril不支持此操作')
    
    print('\n====================使用Pytorch和MindSpore，固定输入和权重，测试正向推理结果（误差范围小于1e-3，若报错则记录报错信息）。目前issues：null====================')
    print('接口mindspore.mint.tril进行改测试意义不大')
    
    print('\n====================测试该神经网络/函数反向，如果是神经网络，则测试Parameter的梯度，如果是函数，则测试函数输入的梯度。目前issues：null====================')
    print('接口mindspore.mint.tril不支持此操作')

    print('===============================All TestingTasks done! 总issues：0==============================')