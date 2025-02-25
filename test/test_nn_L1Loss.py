import numpy as np
import torch
import torch.nn as tnn
import mindspore as ms
from mindspore import Tensor, context, ops
import mindspore.nn as nn
import mindspore.mint.nn as mint_nn

def compare_l1_loss_dtypes():
    """
    测试 mindspore.mint.nn.L1Loss 在不同数据类型下的支持度，对比 GRAPH_MODE 和 PYNATIVE_MODE，以及 PyTorch。
    """

    # 定义数据类型组合
    dtype_list = [
        (ms.float16, torch.float16),
        (ms.float32, torch.float32),
        (ms.float64, torch.float64),
        (ms.int8, torch.int8),
        (ms.int16, torch.int16),
        (ms.int32, torch.int32),
        (ms.int64, torch.int64),
        (ms.uint8, torch.uint8),
        (ms.uint64, torch.uint64),
        (ms.bool_, torch.bool),
    ]

    # 定义测试模式
    modes = [
        ("GRAPH_MODE", context.GRAPH_MODE),
        ("PYNATIVE_MODE", context.PYNATIVE_MODE),
    ]

    # 输入数据
    logits_np = np.array([1, 2, 3])
    labels_np = np.array([1, 2, 2])

    print("开始测试 L1Loss 在不同数据类型和模式下的支持情况...")

    for mode_name, mode in modes:
        print(f"\n=== 测试模式: {mode_name} ===")
        if mode_name=='GRAPH_MODE':
            print("提示：该模式不被测试接口支持。mint.nn.L1Loss 的底层实现依赖于 L1LossExt 算子，而此算子在 Ascend 平台的 Graph Engine (GE) 中未被完全支持（报错显示 Unsupported op type:L1LossExt）。 ")
        # 设置上下文
        context.set_context(mode=mode, device_target="Ascend")

        # 定义损失函数
        loss_ms = nn.L1Loss()
        loss_mint = mint_nn.L1Loss()
        loss_torch = tnn.L1Loss()

        for ms_dtype, torch_dtype in dtype_list:
            print(f"\n--- 测试数据类型: MindSpore {ms_dtype}, PyTorch {torch_dtype} ---")

            # MindSpore Tensor
            logits_ms = Tensor(logits_np, ms_dtype)
            labels_ms = Tensor(labels_np, ms_dtype)

            # PyTorch Tensor
            logits_torch = torch.tensor(logits_np, dtype=torch_dtype)
            labels_torch = torch.tensor(labels_np, dtype=torch_dtype)

            # 测试 PyTorch L1Loss
            try:
                output_torch = loss_torch(logits_torch, labels_torch)
                print(f"PyTorch L1Loss 输出: {output_torch} (dtype: {output_torch.dtype})")
            except Exception as e:
                print(f"PyTorch L1Loss 不支持 {torch_dtype}")

            # 测试 MindSpore nn.L1Loss
            try:
                output_ms = loss_ms(logits_ms, labels_ms)
                print(f"MindSpore nn.L1Loss 输出: {output_ms} (dtype: {output_ms.dtype})")
            except Exception as e:
                print(f"MindSpore nn.L1Loss 不支持 {ms_dtype}")

            # 测试 MindSpore mint.nn.L1Loss
            try:
                m_output_ms = loss_mint(logits_ms, labels_ms)
                print(f"MindSpore mint.nn.L1Loss 输出: {m_output_ms} (dtype: {m_output_ms.dtype})")
            except Exception as e:
                print(f"MindSpore mint.nn.L1Loss 不支持 {ms_dtype}")

def comprehensive_l1loss_test():
    """集成化动态范围L1Loss精度验证测试"""
    # 配置测试参数
    test_shapes = [(1,), (20,), (100,), (10000,), (16, 8), (4, 16, 8), (2, 32, 16, 8)]
    max_float32 = np.finfo(np.float32).max
    base_safety_factor = 0.8  # 基础安全系数
    noise_ratio = 0.05        # 相对噪声比例
    seed = 2024               # 全局随机种子
    
    # 初始化随机种子
    np.random.seed(seed)
    torch.manual_seed(seed)
    ms.set_seed(seed)
    
    print("="*60)
    print("mindspore.mint.nn.L1Loss 动态范围精度验证测试")
    print(f"Float32理论最大值: {max_float32:.3e}")
    print("="*60)

    for shape in test_shapes:
        print(f"\n\n{'='*60}\n▶ 开始测试形状: {shape}")
        
        # ===================== 动态范围计算 =====================
        # 计算形状乘积因子
        dim_product = np.prod(shape).astype(float)
        dynamic_max = max_float32 / (dim_product+1.0)
        safe_max = dynamic_max * base_safety_factor
        safe_min = safe_max * 0.2  # 保持20%动态范围
        
        print(f"动态范围参数:")
        print(f"维度乘积: {dim_product:.1e}")
        print(f"理论最大值: {dynamic_max:.3e}")
        print(f"安全范围: [{safe_min:.3e} ~ {safe_max:.3e}]")

        # ===================== 数据生成阶段 =====================
        try:
            # 生成基准数据（对数均匀分布）
            log_min = np.log(safe_min)
            log_max = np.log(safe_max)
            
            # 生成预测值
            pred_base = np.exp(np.random.uniform(log_min, log_max, size=shape))
            # 生成目标值（扩展10%范围）
            target_base = np.exp(np.random.uniform(log_min*0.9, log_max*1.1, size=shape))
            
            # 添加相对噪声
            pred_noise = pred_base * (1 + np.random.normal(0, noise_ratio, shape))
            target_noise = target_base * (1 + np.random.normal(0, noise_ratio, shape))
            
            # 数据裁剪和转换
            pred_np = np.clip(pred_noise, safe_min, safe_max).astype(np.float32)
            target_np = np.clip(target_noise, safe_min*0.8, safe_max*1.2).astype(np.float32)
            
            # 验证数据有效性
            if np.isnan(pred_np).any() or np.isinf(pred_np).any():
                raise ValueError("预测值包含非法值(nan/inf)")
            if np.isnan(target_np).any() or np.isinf(target_np).any():
                raise ValueError("目标值包含非法值(nan/inf)")
                
        except Exception as e:
            print(f"\n❌ 数据生成失败: {str(e)}")
            continue
        
        # ===================== 数据统计阶段 =====================
        def safe_statistics(arr):
            """安全计算统计指标"""
            arr_f64 = arr.astype(np.float64)
            return {
                'min': arr_f64.min(),
                'max': arr_f64.max(),
                'mean': arr_f64.mean(),
                'std': arr_f64.std()
            }
        
        pred_stats = safe_statistics(pred_np)
        target_stats = safe_statistics(target_np)
        
        print("\n输入数据统计:")
        print(f"[预测值] 最小值:{pred_stats['min']:.3e} 最大值:{pred_stats['max']:.3e}")
        print(f"        均值:{pred_stats['mean']:.3e} 标准差:{pred_stats['std']:.3e}")
        print(f"[目标值] 最小值:{target_stats['min']:.3e} 最大值:{target_stats['max']:.3e}")
        print(f"        均值:{target_stats['mean']:.3e} 标准差:{target_stats['std']:.3e}")

        # ===================== 计算阶段 =====================
        result = {'status': 'Pending', 'errors': []}
        
        try:
            # PyTorch计算
            torch_pred = torch.tensor(pred_np, dtype=torch.float32)
            torch_target = torch.tensor(target_np, dtype=torch.float32)
            torch_loss = torch.nn.L1Loss(reduction='sum')(torch_pred, torch_target).item()
            
            # MindSpore计算
            ms_pred = Tensor(pred_np, dtype=ms.float32)
            ms_target = Tensor(target_np, dtype=ms.float32)
            ms_loss = mint_nn.L1Loss(reduction='sum')(ms_pred, ms_target).asnumpy().item()
            
            # 精度验证
            abs_error = abs(torch_loss - ms_loss)
            rel_error = abs_error / (abs(torch_loss) + 1e-30)
            
            print(f"\n计算结果:")
            print(f"PyTorch Loss:   {torch_loss:.30e}")
            print(f"MindSpore Loss: {ms_loss:.30e}")
            print(f"绝对误差:       {abs_error:.2e} (阈值: <1.00e-03)")
            print(f"相对误差:       {rel_error:.2e} (阈值: <1.00e-03)")
            
            # 误差判断
            error_flags = []
            if abs_error > 1e-3:
                error_flags.append(f"绝对误差超标({abs_error:.2e})")
            if rel_error > 1e-3:
                error_flags.append(f"相对误差超标({rel_error:.2e})")
            if np.isnan(abs_error) or np.isinf(abs_error):
                error_flags.append("无效误差值")
                
            if error_flags:
                result['status'] = 'Failed'
                result['errors'] = error_flags
                print("⚠️ 精度验证失败:", ", ".join(error_flags))
            else:
                result['status'] = 'Passed'
                print("✅ 精度验证通过")
                
        except RuntimeError as e:
            error_msg = str(e).split('\n')[0]
            result.update({'status': 'Error', 'errors': [f"计算错误: {error_msg}"]})
            print(f"\n❌ 运行时错误: {error_msg}")
        except Exception as e:
            error_type = type(e).__name__
            result.update({'status': 'Error', 'errors': [f"{error_type}: {str(e)[:100]}"]})
            print(f"\n❌ 未处理异常: {error_type} - {str(e)[:100]}")

        # ===================== 结果报告阶段 =====================
        print(f"\n测试状态: {result['status']}")
        if result['errors']:
            print("详细错误信息:")
            for err in result['errors']:
                print(f"- {err}")
        
def test_l1loss_parameter_support():
    """测试固定输入下不同参数类型的支持情况"""
    # 固定输入数据
    pred_data = np.array([[1.2, 3.4], [5.6, 7.8]], dtype=np.float32)
    target_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    # 参数组合测试（包含非法类型）
    test_cases = [
        {"reduction": 'mean', "desc": "合法字符串参数"},
        {"reduction": 'invalid', "desc": "非法字符串参数"},
        {"reduction": True, "desc": "布尔类型参数"},
        {"reduction": 123, "desc": "整数类型参数"},
        {"reduction": None, "desc": "None类型参数"},
        {"reduction": [1,2,3], "desc": "数组类型参数"},
        {"reduction": Tensor([1,2,3]), "desc": "tensor类型参数"},
        {"invalid_param": 'mean', "desc": "非法参数名"}
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
            pred = torch.tensor(pred_data)
            target = torch.tensor(target_data)
            if "invalid_param" in case:
                # 测试非法参数名
                loss_fn = torch.nn.L1Loss(invalid_param=case["invalid_param"])
            else:
                loss_fn = torch.nn.L1Loss(reduction=case.get("reduction", 'mean'))
            output = loss_fn(pred, target)
        except Exception as e:
            case_result["torch_support"] = False
            case_result["torch_error"] = f"{type(e).__name__}: {str(e).split(',')[0]}"
        
        # MindSpore测试
        try:
            pred_ms = Tensor(pred_data)
            target_ms = Tensor(target_data)
            if "invalid_param" in case:
                # 测试非法参数名
                loss_fn = ms.mint.nn.L1Loss(invalid_param=case["invalid_param"])
            else:
                loss_fn = ms.mint.nn.L1Loss(reduction=case.get("reduction", 'mean'))
            output_ms = loss_fn(pred_ms, target_ms)
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


def test_l1loss_error_cases():
    """测试L1Loss接口对混乱输入的异常处理能力，对比PyTorch报错准确性"""
    #ms.context.set_context(mode=ms.context.GRAPH_MODE, device_target="Ascend")
    
    test_cases = [
        # 类型不匹配场景（修正数据类型转换）
        {
            "desc": "输入类型不匹配（float32 vs int32）",
            "torch_data": (torch.tensor([[ 0.4822, -1.3873,  0.2989],[ 1.2683, -0.4563, -0.6342]]).float(), 
                           torch.tensor([[1, 0, 1],[0, 1, 0]]).int()),
            "ms_data": (Tensor([[ 0.4822, -1.3873,  0.2989],[ 1.2683, -0.4563, -0.6342]],dtype=ms.float32), 
                      Tensor([[1, 0, 1],[0, 1, 0]], dtype=ms.int32))
        },
        
        # 维度不匹配场景 
        {
            "desc": "输入维度不匹配（2D vs 3D）",
            "torch_data": (torch.randn(2,3), torch.randn(2,3,1)),
            "ms_data": (Tensor(np.random.randn(2,3)), Tensor(np.random.randn(2,3,1)))
        },
        
        # 数据类型非法场景（修正布尔类型转换）
        {
            "desc": "输入布尔类型张量",
            "torch_data": (torch.tensor([[True, False], [False, True]]),
                          torch.tensor([[True, True], [False, False]])),
            "ms_data": (Tensor([[True, False], [False, True]]),
                        Tensor([[True, True], [False, False]]))
        },
        
        # 非张量输入场景
        {
            "desc": "Python列表输入",
            "torch_data": ([1.0, 2.0], [1.0, 2.0]),
            "ms_data": ([1.0, 2.0], [1.0, 2.0])
        },
        
        # 数值溢出场景
        {
            "desc": "极大值输入（可能溢出）",
            "torch_data": (torch.tensor([1e40]), 
                          torch.tensor([-1e40])),
            "ms_data": (Tensor([1e40]), Tensor([-1e40]))
        },
        
        # 特殊参数场景
        {
            "desc": "非法reduction参数",
            "torch_data": (torch.randn(3), torch.randn(3)),
            "ms_data": (Tensor(np.random.randn(3)), Tensor(np.random.randn(3))),
            "params": {"reduction": "invalid"}
        }
    ]

    print("\n==================== L1Loss异常测试开始 ====================\n")
    
    for case in test_cases:
        print(f"\n▌ 测试场景：{case['desc']}")
        print("=" * 90)
        
        # PyTorch测试分支
        torch_loss = None
        try:
            pred, target = case["torch_data"]
            params = case.get("params", {"reduction": 'mean'})
            loss_fn = torch.nn.L1Loss(**params)
            torch_loss = loss_fn(pred, target).item()
            print(f"[PyTorch 通过] 计算结果：{torch_loss:.6f}")
        except Exception as e:
            print(f"[PyTorch 异常] 错误类型：{type(e).__name__}")
            es=str(e).split('\n')[0]
            print(f"异常详情：{es}")
        
        # MindSpore测试分支
        ms_loss = None
        try:
            pred_ms, target_ms = case["ms_data"]
            params = case.get("params", {"reduction": 'mean'})
            loss_fn = ms.mint.nn.L1Loss(**params)
            ms_loss = loss_fn(pred_ms, target_ms).asnumpy()
            print(f"[MindSpore 通过] 计算结果：{ms_loss:.6f}")
        except Exception as e:
            print(f"[MindSpore 异常] 错误类型：{type(e).__name__}")
            es=str(e).split('\n')[0]
            print(f"异常详情：{es}")
        
        # 结果对比分析
        if torch_loss is not None and ms_loss is not None:
            diff = abs(torch_loss - float(ms_loss))
            print(f"\n※ 数值对比 ※ | PyTorch: {torch_loss:.6f} | MindSpore: {float(ms_loss):.6f}")
            print(f"绝对差值：{diff:.6e}", end=' ')
            print("✓ 数值一致" if diff < 1e-6 else "⚠ 存在差异")
        elif torch_loss or ms_loss:
            print("\n※ 状态对比 ※ | 一方计算成功一方失败")
        else:
            print("\n※ 状态对比 ※ | 双方均触发异常但无计算结果")
        
        print("=" * 90 + "\n")
        
def test_l1_loss_forward_and_backward():


    # 设置 MindSpore 上下文，针对 Ascend 910B 平台
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")  # PYNATIVE_MODE 支持动态梯度计算

    # 固定输入数据（模拟神经网络预测值和目标值）
    logits_np = np.array([[215.520412, 3.544234537, 3.0], [78.621276, 685.454568,0.0]], dtype=np.float32)  # 预测值
    labels_np = np.array([[124.34505545, 12.111541, 3.3255], [56.786785142, 85.989,0.0]], dtype=np.float32)  # 目标值

    # MindSpore 输入（移除 requires_grad）
    logits_ms = Tensor(logits_np, ms.float32)
    labels_ms = Tensor(labels_np, ms.float32)

    # PyTorch 输入
    logits_torch = torch.tensor(logits_np, dtype=torch.float32, requires_grad=True)
    labels_torch = torch.tensor(labels_np, dtype=torch.float32)

    # 定义损失函数
    loss_mint = mint_nn.L1Loss(reduction='mean')
    loss_torch = tnn.L1Loss(reduction='mean')

    print("\n=== 测试 mindspore.mint.nn.L1Loss 正向推理和反向梯度准确性 ===")
    print(f"固定输入数据:\n  logits:\n{logits_np}\n  labels:\n{labels_np}")


    # --- 子任务 b: 正向推理测试 ---
    print("\n--- 正向推理测试 ---")

    # PyTorch 正向
    torch_result = None
    torch_error = None
    try:
        torch_output = loss_torch(logits_torch, labels_torch)
        torch_result = torch_output.item()
        print(f"PyTorch nn.L1Loss 输出: {torch_result}")
    except Exception as e:
        torch_error = f"PyTorch 错误: {type(e).__name__} - {str(e)[:100]}"
        print(torch_error)

    # MindSpore 正向
    ms_result = None
    ms_error = None
    try:
        ms_output = loss_mint(logits_ms, labels_ms)
        ms_result = ms_output.asnumpy().item()
        print(f"MindSpore mint.nn.L1Loss 输出: {ms_result}")
    except Exception as e:
        ms_error = f"MindSpore 错误: {type(e).__name__} - {str(e)[:100]}"
        print(ms_error)

    # 正向对比
    if torch_result is not None and ms_result is not None:
        abs_diff = abs(torch_result - ms_result)
        if abs_diff < 1e-3:
            status = "一致 (误差 < 1e-3)"
        else:
            status = f"不一致 (误差 = {abs_diff:.6f})"
        print(f"正向对比结果: {status}")
    else:
        print("正向对比结果: 无法比较，因存在报错")

    # --- 子任务 c: 反向传播测试 ---
    print("\n--- 反向传播测试 ---")

    # PyTorch 反向
    torch_grad = None
    if torch_result is not None:
        try:
            torch_output.backward()
            torch_grad = logits_torch.grad.numpy()
            print(f"PyTorch 输入梯度:\n{torch_grad}")
        except Exception as e:
            torch_error = f"PyTorch 反向错误: {type(e).__name__} - {str(e)[:100]}"
            print(torch_error)

    # MindSpore 反向
    ms_grad = None
    if ms_result is not None:
        try:
            # 定义梯度计算函数
            grad_fn = ops.GradOperation(get_all=True)(loss_mint)
            ms_grads = grad_fn(logits_ms, labels_ms)
            ms_grad = ms_grads[0].asnumpy()  # 取第一个输入的梯度
            print(f"MindSpore 输入梯度:\n{ms_grad}")
        except Exception as e:
            ms_error = f"MindSpore 反向错误: {type(e).__name__} - {str(e)[:100]}"
            print(ms_error)

    # 反向对比
    if torch_grad is not None and ms_grad is not None:
        grad_diff = np.abs(torch_grad - ms_grad).max()
        if grad_diff < 1e-3:
            status = "一致 (最大误差 < 1e-3)"
        else:
            status = f"不一致 (最大误差 = {grad_diff:.6f})"
        print(f"反向对比结果: {status}")
    else:
        print("反向对比结果: 无法比较，因存在报错")
        if torch_error:
            print(f"  {torch_error}")
        if ms_error:
            print(f"  {ms_error}")

    print("\n=== 测试完成 ===")    
if __name__ == "__main__":
    
    print('\n====================测试random输入不同dtype，对比两个框架的支持度。目前issues：4====================')
    compare_l1_loss_dtypes()
    print('\n====================上一个测试结束。提示：我们在上面已经讨论出mint.nn.l1loss不支持GRAPH_MODE，故我们下面不再用GRAPH_MODE进行优化！====================')
    

    print('\n====================测试固定dtype，random输入值，对比两个框架输出是否相等（误差范围为小于1e-3）。目前issues：1====================')
    comprehensive_l1loss_test()
    print('\n====================上一个测试结束。提示：两个框架在某些时候有误差，但是量级在1e-8到1e-7浮动，是可容许的！====================')
    

    print('\n====================测试固定shape，固定输入值，不同输入参数（string\bool等类型），两个框架的支持度。目前issues：0====================')
    test_l1loss_parameter_support()
    print('\n====================上一个测试结束。====================')
    

    print('\n====================测试随机混乱输入，报错信息的准确性。目前issues：1====================')
    test_l1loss_error_cases()
    print('\n====================上一个测试结束。====================')
    

    print('\n====================使用Pytorch和MindSpore，固定输入和权重，测试正向推理结果（误差范围小于1e-3，若报错则记录报错信息）测试该神经网络/函数反向，如果是神经网络，则测试Parameter的梯度，如果是函数，则测试函数输入的梯度。目前issues：0====================')
    test_l1_loss_forward_and_backward()
    
    
    print('\n===============================All TestingTasks done! 总issues：6==============================')