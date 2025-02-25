"""
【任务背景】
mindspore.mint.searchsorted
【需求描述】
1.对应Pytorch 的相应接口进行测试：
a) 测试random输入不同dtype，对比两个框架的支持度                                                                 issue：1
b) 测试固定dtype，random输入值，对比两个框架输出是否相等（误差范围为小于1e-3）                                   issue：0
c) 测试固定shape，固定输入值，不同输入参数（string\bool等类型），两个框架的支持度                                issue：2
d) 测试随机混乱输入，报错信息的准确性                                                                            issue：3
2. 测试使用接口构造函数/神经网络的准确性                                                                         
a) Github搜索带有该接口的代码片段/神经网络                                                                       issue：null
b) 使用Pytorch和MindSpore，固定输入和权重，测试正向推理结果（误差范围小于1e-3，若报错则记录报错信息）            issue：null
c) 测试函数反向,测试函数输入的梯度                                                                               issue：null
"""
import numpy as np
import torch
import mindspore as ms
from mindspore import Tensor, nn, Parameter, context
import mindspore.mint as mint
import itertools
from collections import defaultdict
import pytest


def compare_searchsorted_dtypes(shape = (4, 5)):# 测试输入的形状
    """
    测试随机输入不同dtype，对比两个框架的支持度
    对于
    MindSpore未主动捕获了异常，直到调用了ms_output！
    """
    
    dtypes = [
        np.float16, np.float32, np.float64,
        np.int8, np.int16, np.int32, np.int64,
        np.uint8, np.uint16, np.uint32, np.uint64,
        np.bool_
    ]
    
    results = []

    for np_dtype in dtypes:
        print(f"\n正在测试数据类型: {np_dtype}")
        torch_support = True
        ms_support = True
        error_msg = {"torch": "", "mindspore": ""}
        status = ""

        # PyTorch 错误捕获
        try:
            # 生成随机数据
            data = np.random.randn(*shape).astype(np_dtype)
            
            # PyTorch 输入
            torch_input = torch.tensor(data, dtype=torch.__dict__.get(np_dtype.__name__, None))
            try:
                print(f'torch_input:\n{torch_input}')
            except Exception as e:
                print(f'Pytorch未主动捕获了异常，直到调用了torch_input！')
                torch_support = False
                error_msg["torch"] = f"PyTorch: {str(e)}"
            torch_output = torch.searchsorted(torch_input, torch_input, side="left")
            try:
                print(f'torch_output:\n{torch_output}')
            except Exception as e:
                print(f'Pytorch未主动捕获了异常，直到调用了torch_output！')
                torch_support = False
                error_msg["torch"] = f"PyTorch: {str(e)}"
                
        except Exception as e:
            print(f'Pytorch主动捕获了异常！')
            torch_support = False
            error_msg["torch"] = f"PyTorch: {str(e)}"

        # MindSpore 错误捕获
        try:
            ms_input = ms.Tensor(data, dtype=ms.__dict__.get(np_dtype.__name__, None))
            try:
                print(f'ms_input:\n{ms_input}')
            except Exception as e:
                print(f'MindSpore未主动捕获了异常，直到调用了ms_input！')
                ms_support = False
                error_msg["mindspore"] = f"MindSpore: {str(e)}"
            ms_output = mint.searchsorted(ms_input, ms_input, side="left")
            try:
                print(f'ms_output:\n{ms_output}')
            except Exception as e:
                print(f'MindSpore未主动捕获了异常，直到调用了ms_output！')
                ms_support = False
                error_msg["mindspore"] = f"MindSpore: {str(e)}"

        except Exception as e:
            print(f'MindSpore主动捕获了异常！')
            ms_support = False
            error_msg["mindspore"] = f"MindSpore: {str(e)}"

        # 生成测试结果
        if torch_support and ms_support:
            status = "支持"
        elif torch_support != ms_support:
            status = "存在差异"
        else:
            status = "不支持"

        results.append(
            {
                "dtype": str(np_dtype),
                "torch_support": torch_support,
                "ms_support": ms_support,
                "status": status,
                "torch_error": error_msg["torch"],
                "ms_error": error_msg["mindspore"],
            }
        )

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
        

def compare_searchsorted_accuracy(test_cycles=1, max_dims=2):
    
    """
    测试固定dtype，random输入值，对比两个框架输出是否相等（误差范围为小于1e-3）
    1.由于searchsorted返回的是下标索引的Tensor，故我们容许的误差是0
    2.由于前面已经提过不支持uint，故去除了uint的测试
    """
    # 环境配置
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    
    # 测试配置
    dtype_settings = [
        (np.bool_, 0),
        (np.float16, 0),(np.float32, 0),(np.float64, 0),
        (np.int8, 0),(np.int16, 0),(np.int32, 0),(np.int64, 0),
        (np.complex64, 0),(np.complex128, 0),
    ]
    
    # 维度模板（优化布尔测试规模）
    DIM_CONFIG = {
        1: {'seq': (10,), 'val': (5,)},
        2: {'seq': (5, 8), 'val': (5, 3)},
        3: {'seq': (2, 3, 5), 'val': (2, 3, 2)},
    }

    # 主测试流程
    for dtype, tol in dtype_settings:
        print(f"\n{'='*40}")
        print(f"▶ 开始测试数据类型：{dtype.__name__}")
        
        # 类型兼容性检查
        if np.issubdtype(dtype, np.complexfloating):
            framework_support = all([
                hasattr(torch.Tensor, 'searchsorted'),
                'searchsorted' in dir(ms.ops),
            ])
            if not framework_support:
                print(f"⚠ 框架不支持复数排序，跳过 {dtype} 测试")
                print(f"✓ 符合预期")
                continue

        for cycle in range(1, test_cycles+1):
            print(f"\n—— 第 {cycle}/{test_cycles} 轮 ——")
            
            for dim in range(1, max_dims+1):
                conf = DIM_CONFIG[dim]
                seq_shape, val_shape = conf['seq'], conf['val']
                print(f"\n▌ 维度 {dim}D | 序列：{seq_shape} | 值：{val_shape}")

                # === 数据生成 ===
                if dtype == np.bool_:
                    # 布尔类型特殊处理
                    sorted_seq = np.sort(np.random.choice([False, True], size=seq_shape), axis=-1)
                    values = np.random.choice([False, True], size=val_shape)
                elif np.issubdtype(dtype, np.complexfloating):
                    # 复数生成逻辑
                    sorted_seq = (np.random.randn(*seq_shape) + 1j*np.random.randn(*seq_shape)).astype(dtype)
                    sorted_seq = np.sort(sorted_seq, axis=-1)
                    values = (np.random.randn(*val_shape) + 1j*np.random.randn(*val_shape)).astype(dtype)
                elif np.issubdtype(dtype, np.floating):
                    # 浮点生成逻辑
                    sorted_seq = np.sort(np.random.randn(*seq_shape).astype(dtype)*5, axis=-1)
                    values = np.random.randn(*val_shape).astype(dtype)*8
                else:
                    # 整数处理
                    info = np.iinfo(dtype)
                    sorted_seq = np.sort(
                        np.random.randint(info.min//2, info.max//2, seq_shape, dtype=dtype),
                        axis=-1
                    )
                    values = np.random.randint(info.min, info.max, val_shape, dtype=dtype)

                # === 数据打印 ===
                print(f"序列范围：[{sorted_seq.min()}, {sorted_seq.max()}]")
                if dtype == np.bool_:
                    print(f"True占比：{np.count_nonzero(sorted_seq)/sorted_seq.size:.1%}")

                # === 双框架计算 ===
                try:
                    # 统一转换逻辑
                    if dtype == np.bool_:
                        # 布尔类型转为float32处理
                        torch_in = torch.from_numpy(sorted_seq.astype(np.float32))
                        torch_val = torch.from_numpy(values.astype(np.float32))
                        ms_in = ms.Tensor(sorted_seq.astype(np.float32))
                        ms_val = ms.Tensor(values.astype(np.float32))
                    else:
                        torch_in = torch.from_numpy(sorted_seq)
                        torch_val = torch.from_numpy(values)
                        ms_in = ms.Tensor(sorted_seq)
                        ms_val = ms.Tensor(values)
                    
                    # 执行计算
                    torch_out = torch.searchsorted(torch_in, torch_val, side='left').numpy()
                    ms_out = ms.mint.searchsorted(ms_in, ms_val, side='left').asnumpy()
                except Exception as e:
                    print(f"✗ 计算失败：{str(e)}")
                    continue

                # === 精度验证 ===
                # 布尔类型特殊处理
                if dtype == np.bool_:
                    # 转换回布尔类型进行比较
                    torch_out = torch_out.astype(bool)
                    ms_out = ms_out.astype(bool)
                    diff = torch_out != ms_out  # 直接比较布尔结果差异
                else:
                    diff = np.abs(torch_out - ms_out)
                
                max_err = diff.max() if not np.issubdtype(diff.dtype, np.bool_) else diff.any()
                error_count = diff.sum() if not np.issubdtype(diff.dtype, np.bool_) else diff.sum()

                # === 结果处理 ===
                print(f"精度结果：差异点 {error_count}/{diff.size}")
                if (max_err > tol) if not isinstance(max_err, bool) else max_err:
                    error_msg = [
                        f"\n【warming】 测试失败：{dtype.__name__} {dim}D",
                        f"输入值样例：{values[0] if values.size > 0 else values}",
                        f"Torch结果：{torch_out[0] if torch_out.size > 0 else torch_out}",
                        f"MS结果   ：{ms_out[0] if ms_out.size > 0 else ms_out}",
                        f"序列片段：{sorted_seq[-10:] if sorted_seq.size > 10 else sorted_seq}"
                    ]
                    if dtype == np.bool_:
                        error_msg.insert(1, f"差异类型：布尔值不匹配")
                    raise AssertionError("\n".join(map(str, error_msg)))
                else:
                    print(f"✓ 测试通过")

    print("\n" + "="*40)
    print("✓ 所有测试完成！")
    




def validate_fixed_input_support(verbose=False, show_details=False):
    """固定输入值参数兼容性验证（详细报告版）"""
    # 局部环境配置
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    
    # 配置参数空间（新增字符串类型测试）
    param_config = {
        'input_dtype': [np.float32, np.int32, np.bool_, np.uint8, np.uint16],
        'value_dtype': [np.float32, np.int32, np.bool_, np.uint8, np.uint16],
        'search_side': ['left', 'right', ' left', 'middle', 666, True],
        'output_int32': [False, True,-1 ,0, 1, "str_123"]
    }
    
    # 固定基础数据
    base_data = {
        'sorted_seq': np.array([-1, 0, 1, 2, 3, 3, 4, 5]),
        'values': np.array([0.5, 2.5, 3, 4.5])
    }
    
    # 结果收集器（增强数据结构）
    results = []
    param_combinations = list(itertools.product(*param_config.values()))
    
    # ------------------------- 详细测试执行 -------------------------
    for idx, params in enumerate(param_combinations):
        param_dict = dict(zip(param_config.keys(), params))
        case_id = idx + 1
        case_info = {
            'id': case_id,
            'params': param_dict.copy(),
            'status': 'pending',
            'details': {}
        }
        
        try:
            # ------------------------- 数据准备 -------------------------
            # 增强类型转换处理
            def enhanced_cast(arr, dtype):
                if dtype == str:
                    return np.array(['a', 'b', 'c', 'd', 'e'])[:len(arr)]  # 生成字符串序列
                if dtype == np.bool_:
                    return np.sort(np.random.choice([False, True], size=len(arr)))
                return arr.astype(dtype)
            
            test_data = {
                'sorted_seq': enhanced_cast(base_data['sorted_seq'], param_dict['input_dtype']),
                'values': enhanced_cast(base_data['values'], param_dict['value_dtype'])
            }
            
            # ------------------------- 预期判断 -------------------------
            expect_error = (
                param_dict['search_side'] not in ['left', 'right'] or
                param_dict['input_dtype'] == str or  # 字符串类型预期报错
                param_dict['value_dtype'] == str or
                (param_dict['input_dtype'] == np.bool_ and 
                 param_dict['value_dtype'] in [np.float32, np.int32])
            )
            
            # ------------------------- 执行测试 -------------------------
            # PyTorch测试
            torch_err, torch_out = None, None
            try:
                t_seq = torch.from_numpy(test_data['sorted_seq'])
                t_val = torch.from_numpy(test_data['values'])
                torch_res = torch.searchsorted(
                    t_seq, t_val,
                    side=param_dict['search_side'],
                    out_int32=param_dict['output_int32']
                )
                torch_out = torch_res.numpy().tolist()
            except Exception as e:
                torch_err = str(e)
            
            # MindSpore测试
            ms_err, ms_out = None, None
            try:
                m_seq = ms.Tensor(test_data['sorted_seq'])
                m_val = ms.Tensor(test_data['values'])
                ms_res = ms.Tensor(np.array([0, 0, 0], dtype=np.int32))
                ms_res = ms.mint.searchsorted(
                    m_seq, m_val,
                    side=param_dict['search_side'],
                    out_int32=param_dict['output_int32']
                )
                ms_out = ms_res.asnumpy().tolist()
            except Exception as e:
                ms_err = str(e)
            
            # ------------------------- 结果分析 -------------------------
            error_sources = []
            if torch_err: error_sources.append('torch')
            if ms_err: error_sources.append('ms')
            
            # 状态分类逻辑
            if error_sources:
                if len(error_sources) == 2 and expect_error:
                    status = 'expected_error'
                else:
                    status = 'unexpected_error'
            else:
                if np.array_equal(torch_out, ms_out):
                    status = 'passed'
                else:
                    status = 'value_mismatch'
            
            # 构建详细结果
            case_info.update({
                'status': status,
                'input_sample': {
                    'sorted_seq': test_data['sorted_seq'].tolist(),
                    'values': test_data['values'].tolist()
                },
                'torch': {'output': torch_out, 'error': torch_err},
                'ms': {'output': ms_out, 'error': ms_err}
            })
            
            # ------------------------- 实时详细输出 -------------------------
            if verbose:
                print(f"\n▼ 测试案例 {case_id} ▼")
                params_str = ", ".join([f"{k}={v}" for k, v in param_dict.items()])
                print(f"▌ 参数配置: {params_str}")
                
                print(f"▌ 输入样本:")
                print(f"Sorted Seq: {case_info['input_sample']['sorted_seq']}")
                print(f"Values:     {case_info['input_sample']['values']}")
                
                if status == 'passed':
                    print(f"✓ 通过 | 输出结果: {case_info['torch']['output']}")
                elif status == 'value_mismatch':
                    print(f"✗ 数值差异")
                    print(f"Torch输出: {case_info['torch']['output']}")
                    print(f"MS输出:   {case_info['ms']['output']}")
                else:
                    print(f"⚠ 异常类型: {status}")
                    if case_info['torch']['error']:
                        print(f"Torch错误: {case_info['torch']['error'][:100]}")
                    if case_info['ms']['error']:
                        print(f"MS错误:   {case_info['ms']['error'][:100]}")
                print("-"*80)
            
        except Exception as e:
            case_info.update({
                'status': 'system_error',
                'error': str(e)
            })
        
        results.append(case_info)
    
    # ------------------------- 详细报告生成 -------------------------
    def generate_report():
        # 统计指标
        stats = defaultdict(int)
        for case in results:
            stats[case['status']] += 1
        
        # 打印报告头
        print("\n" + "="*80)
        print("固定输入兼容性验证报告".center(80))
        print("="*80)
        print(f"总测试用例: {len(results)}")
        print(f"通过用例: {stats['passed']} ({stats['passed']/len(results):.1%})")
        print(f"预期异常: {stats['expected_error']} ({stats['expected_error']/len(results):.1%})")
        print(f"未预期异常: {stats['unexpected_error']} ({stats['unexpected_error']/len(results):.1%})")
        print(f"数值差异: {stats['value_mismatch']}")
        
        # 异常根因分析
        param_stats = defaultdict(lambda: defaultdict(int))
        for case in filter(lambda c: c['status'] == 'unexpected_error', results):
            for param, val in case['params'].items():
                param_stats[param][f"{val}"] += 1
        
        # 修改后的参数分布打印部分
        print("\n▌ 未预期异常参数分布:")
        for param, vals in param_stats.items():
            # 将值-次数对格式化为字符串
            value_counts = [f"{val}:{count}次" for val, count in sorted(vals.items(), key=lambda x: -x[1])]
            # 组合成单行输出
            print(f"{param}: {', '.join(value_counts)}")
        # ==================== 绝对异常根因分析 ====================
        print("\n" + "="*80)
        print("绝对异常根因分析".center(80))
        print("判断条件：该参数值出现的所有用例均引发异常，且至少包含一次未预期异常")
    
        # 参数统计数据结构
        param_stats = defaultdict(lambda: {
            'total_cases': 0,
            'total_errors': 0,
            'unexpected_errors': 0
        })
    
        # 遍历所有用例收集统计信息
        for case in results:
            for param, value in case['params'].items():
                param_key = f"{param}={value}"
                param_stats[param_key]['total_cases'] += 1
                if case['status'] in ['expected_error', 'unexpected_error']:
                    param_stats[param_key]['total_errors'] += 1
                if case['status'] == 'unexpected_error':
                    param_stats[param_key]['unexpected_errors'] += 1
    
        # 识别绝对根因参数
        root_causes = []
        for param_key, stats in param_stats.items():
            if stats['total_errors'] == stats['total_cases'] and stats['unexpected_errors'] > 0:
                root_causes.append((
                    param_key,
                    stats['unexpected_errors'],
                    stats['total_cases'],
                    stats['unexpected_errors'] / stats['total_cases']
                ))
    
        # 展示分析结果
        if root_causes:
            print("\n发现以下绝对异常根因参数：")
            for param, unexp, total, ratio in sorted(root_causes, key=lambda x: -x[3]):
                print(f"▌ 参数：{param}")
                print(f"  总出现次数: {total}")
                print(f"  引发未预期异常次数: {unexp} ({ratio:.1%})")
                print("-"*80)
        else:
            print("\n未发现符合判定条件的绝对异常根因参数")
        # 详细案例展示
        if show_details:
            print("\n" + "="*80)
            print("详细异常案例".center(80))
            for case in filter(lambda c: c['status'] in ['unexpected_error', 'value_mismatch'], results):
                print(f"\n▼ 案例 {case['id']} ▼")
                print(f"状态: {case['status'].upper()}")
                params_str = ", ".join([f"{k}={v}" for k, v in case['params'].items()])
                print(f"参数配置: {params_str}")
                print("\n输入样本:")
                print(f"Sorted Seq: {case['input_sample']['sorted_seq']}")
                print(f"Values:     {case['input_sample']['values']}")
                
                if case['torch']['error']:
                    print(f"\nTorch错误: {case['torch']['error'][:100]}")
                if case['ms']['error']:
                    print(f"MS错误:   {case['ms']['error'][:100]}")
                
                if case['status'] == 'value_mismatch':
                    print("\n输出对比:")
                    print(f"Torch: {case['torch']['output']}")
                    print(f"MS:    {case['ms']['output']}")
                print("="*80)
    
    generate_report()
    return results

def test_mindspore_searchsorted_error_cases():
    """综合测试mindspore.mint.searchsorted接口的异常处理能力"""
    ##############################################
    """测试点1：PyTorch和MindSpore的精度溢出测试"""
    def merged_test_overflow_cases():
        """精度溢出测试函数"""
        MAX_INT32 = 2**31 - 1

        # PyTorch测试部分
        with torch.no_grad():
            pt_sorted = torch.arange(MAX_INT32 + 500, dtype=torch.int64)

        pt_test_cases = [
            {
                'values': [1, MAX_INT32-100, MAX_INT32+100],
                'out_int32': True,
                'desc': "超过int32上限(out_int32=True)"
            },
            {
                'values': [1, MAX_INT32-100, MAX_INT32+100],
                'out_int32': False,
                'desc': "超过int32上限(out_int32=False)"
            },
            {
                'values': [1, MAX_INT32],
                'out_int32': True,
                'desc': "精确边界值测试"
            }
        ]

        print("\n=== 开始PyTorch测试 ===")
        for case in pt_test_cases:
            print(f"\n=== 执行测试用例：{case['desc']} ===")
            try:
                pt_values = torch.tensor(case['values'], dtype=torch.int64)
                pt_result = torch.searchsorted(pt_sorted, pt_values, out_int32=case['out_int32'])
                print(f'pt_result:{pt_result}')
                pt_num_result = pt_result.numpy()
                for i in range(len(case['values'])):
                    if case['values'][i] > MAX_INT32:
                        if pt_num_result[i] < 0:
                            print("√ 溢出检测成功：检测到负索引值")
                        elif case['out_int32']:
                            print("x 溢出检测失败：可能发生截断")
            except MemoryError:
                print("! 内存不足：需要至少70GB内存执行本测试")
            except Exception as e:
                err_head = str(e).split('\n', 1)[0]
                error_type = type(e).__name__
                print(f"异常处理触发 | 类型：{error_type} | 关键信息：{err_head}...")

        del pt_sorted

        # MindSpore测试部分
        ms_sorted = Tensor(np.arange(MAX_INT32 + 500, dtype=np.int64), ms.int64)

        ms_test_cases = [
            {
                'values': [1, MAX_INT32-100, MAX_INT32+100],
                'out_int32': True,
                'desc': "超过int32上限(out_int32=True)"
            },
            {
                'values': [1, MAX_INT32-100, MAX_INT32+100],
                'out_int32': False,
                'desc': "超过int32上限(out_int32=False)"
            },
            {
                'values': [1, MAX_INT32-100, MAX_INT32],
                'out_int32': True,
                'desc': "精确边界值测试"
            }
        ]

        print("\n=== 开始MindSpore测试 ===")
        for case in ms_test_cases:
            print(f"\n=== 执行测试用例：{case['desc']} ===")
            try:
                ms_values = Tensor(case['values'], ms.int64)
                ms_result = mint.searchsorted(ms_sorted, ms_values, out_int32=case['out_int32'])
                print(ms_result)
                ms_num_result = ms_result.numpy()
                for i in range(len(case['values'])):
                    if case['values'][i] > MAX_INT32:
                        if ms_num_result[i] < 0:
                            print("√ 溢出检测成功：检测到负索引值")
                        elif case['out_int32']:
                            print("x 溢出检测失败：可能发生截断")
            except RuntimeError:
                print("! 内存不足：需要至少70GB内存执行本测试")
            except Exception as e:
                err_head = str(e).split('\n', 1)[0]
                error_type = type(e).__name__
                print(f"异常处理触发 | 类型：{error_type} | 关键信息：{err_head}...")
    print('============测试点1：PyTorch和MindSpore的精度溢出测试开始=============')
    merged_test_overflow_cases()
    print('============测试点1：PyTorch和MindSpore的精度溢出测试结束=============')
    ##############################################
    """测试点2：PyTorch和MindSpore的各种报错准确度"""
    def torch_error_test_cases():
        """PyTorch版独立异常测试用例"""
        cases = [
            ("用Python列表作为sorted_sequence", 
             lambda: torch.searchsorted([1.0, 2.0, 3.0], torch.tensor([2.5])), 
             lambda: mint.searchsorted([1.0, 2.0, 3.0], Tensor([2.5]))),

            ("用Python列表作为values", 
             lambda: torch.searchsorted(torch.tensor([1.0, 2.0, 3.0]), [2.5]), 
             lambda: mint.searchsorted(Tensor([1.0, 2.0, 3.0]), [2.5])),
            
            #pytorch和ms有差异
            ("sorted_sequence空序列测试", 
             lambda: torch.searchsorted(torch.tensor([], dtype=torch.float32), torch.tensor([1.0])), 
             lambda: mint.searchsorted(Tensor(np.array([], dtype=np.float32)), Tensor([1.0]))),
            
            #pytorch和ms有差异
            ("values空序列测试", 
             lambda: torch.searchsorted(torch.tensor([1.0]), torch.tensor([], dtype=torch.float32)), 
             lambda: mint.searchsorted(Tensor([1.0]), Tensor(np.array([], dtype=np.float32)))),
            
            #pytorch和ms有差异
            ("全空序列测试", 
             lambda: torch.searchsorted(torch.tensor([], dtype=torch.float32), torch.tensor([], dtype=torch.float32)), 
             lambda: mint.searchsorted(Tensor(np.array([], dtype=np.float32)), Tensor(np.array([], dtype=np.float32)))),

            ("无效参数测试", 
             lambda: torch.searchsorted(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0]), invalid_param=True), 
             lambda: mint.searchsorted(Tensor([1.0, 2.0, 3.0]), Tensor([1.0]), invalid_param=True)),

            ("类型冲突测试", 
             lambda: torch.searchsorted(torch.tensor([1, 2, 3], dtype=torch.int32), torch.tensor([2.5], dtype=torch.float32)), 
             lambda: mint.searchsorted(Tensor([1, 2, 3], ms.int32), Tensor([2.5], ms.float32))),
            
            #pytorch和ms有差异
            ("非单调3维+1维", 
             lambda: torch.searchsorted(torch.tensor([[[1,3,2,4], [5,6,7,8],[9,10,11,12]], [[13,14,15,16],[17,18,19,20],[21,23,22,24]]]), torch.tensor([2.5])), 
             lambda: mint.searchsorted(Tensor([[[5,6,7,8],[1,2,3,4],[9,10,11,12]], [[13,14,15,16],[17,18,19,20],[21,23,22,24]]]), Tensor([[2.5]], ms.float32))),
            
            #pytorch和ms有差异
            ("非单调3维+2维", 
             lambda: torch.searchsorted(torch.tensor([[[5,6,7,8],[1,2,3,4],[9,10,11,12]], [[13,14,15,16],[17,18,19,20],[21,23,22,24]]]), torch.tensor([[2.5]])), 
             lambda: mint.searchsorted(Tensor([[[5,6,7,8],[1,2,3,4],[9,10,11,12]], [[13,14,15,16],[17,18,19,20],[21,23,22,24]]]), Tensor([[2.5]], ms.float32))),

            ("1维+3维", 
             lambda: torch.searchsorted(torch.tensor([2.5]), torch.tensor([[[1,2,3,4],[5,6,7,8],[9,10,11,12]], [[[13,14,15,16],[17,18,19,20],[21,23,22,24]]]])), 
             lambda: mint.searchsorted(Tensor([2.5]), Tensor([[[1,2,3,4],[5,6,7,8],[9,10,11,12]], [[[13,14,15,16],[17,18,19,20],[21,23,22,24]]]]))),

            ("3维+2维", 
             lambda: torch.searchsorted(torch.tensor([[[5,6,7,8],[1,2,3,4],[9,10,11,12]], [[13,14,15,16],[17,18,19,20],[21,23,22,24]]]), torch.tensor([[1,2,3],[4,5,6]])), 
             lambda: mint.searchsorted(Tensor([[[5,6,7,8],[1,2,3,4],[9,10,11,12]], [[13,14,15,16],[17,18,19,20],[21,23,22,24]]]), Tensor([[1,2,3],[4,5,6]])))
        ]

        for case_desc, torch_func, ms_func in cases:
            print(f"Case: {case_desc}")
            for func, framework in [(torch_func, 'PyTorch'), (ms_func, 'MindSpore')]:
                try:
                    func()
                    print(f"{framework} 未触发错误")
                except Exception as e:
                    err_msg = str(e).split('\n', 1)[0]
                    print(f"{framework} 错误类型：{type(e).__name__} | 关键信息：{err_msg}...")
    print('============测试点2：PyTorch和MindSpore的各种报错准确度开始=============')
    torch_error_test_cases()
    print('============测试点2：PyTorch和MindSpore的各种报错准确度结束=============')


if __name__ == "__main__":
    
    print('\n====================测试random输入不同dtype，对比两个框架的支持度。目前issues：1====================')
    compare_searchsorted_dtypes()
    
    print('\n====================测试固定dtype，random输入值，对比两个框架输出是否相等（误差范围为小于1e-3）。目前issues：0====================')
    compare_searchsorted_accuracy()
    
    print('\n====================测试固定shape，固定输入值，不同输入参数（string\bool等类型），两个框架的支持度。目前issues：2====================')
    validate_fixed_input_support()
    
    print('\n====================测试随机混乱输入，报错信息的准确性。目前issues：3====================')
    test_mindspore_searchsorted_error_cases()
    
    print('\n====================Github搜索带有该接口的代码片段/神经网络。目前issues：null====================')
    print('接口mindspore.mint.searchsorted不支持此操作')
    
    print('\n====================使用Pytorch和MindSpore，固定输入和权重，测试正向推理结果（误差范围小于1e-3，若报错则记录报错信息）。目前issues：null====================')
    print('接口mindspore.mint.searchsorted进行改测试意义不大')
    
    print('\n====================测试该神经网络/函数反向，如果是神经网络，则测试Parameter的梯度，如果是函数，则测试函数输入的梯度。目前issues：null====================')
    print('接口mindspore.mint.searchsorted不支持此操作')

    print('===============================All TestingTasks done! 总issues：6==============================')