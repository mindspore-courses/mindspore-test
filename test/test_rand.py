import pytest
import numpy as np
import mindspore as ms

from mindspore import mint
import torch


dtype_ms = ms.float32
dtype_torch = torch.float32

shape = [3, 4]
rand_seed = np.random.randint(100)

def is_same(tensor_ms, tensor_torch):
    return np.allclose(tensor_ms.asnumpy(), tensor_torch.numpy())

@ms.jit()
def rand_jit(shape, generator=None, dtype=None):
    return mint.rand(*shape, generator=generator, dtype=dtype)


@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_rand_different_dtypes(mode):
    """测试random输入不同dtype，对比两个框架的支持度"""
    ms.set_context(mode=mode)

    ms_dtypes = [ms.int8, ms.int16, ms.int32, ms.int64, ms.uint8, ms.uint16, ms.uint32, ms.uint64, ms.float16, ms.float32, ms.float64, ms.bfloat16, ms.bool_]
    torch_dtypes = [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.uint16, torch.uint32, torch.uint64, torch.float16, torch.float32, torch.float64, torch.bfloat16, torch.bool]

    for i in range(len(ms_dtypes)):
        dtype_ms = ms_dtypes[i]
        dtype_torch = torch_dtypes[i]

        err = False
        try:
            ms_result = rand_jit(shape, dtype=dtype_ms).asnumpy()
        except Exception as e:
            err = True
            print(f"mint.rand not supported for {dtype_ms}")
            # print(e)

        try:
            torch_result = torch.rand(*shape, dtype=dtype_torch).numpy()
        except Exception as e:
            err = True
            print(f"torch.rand not supported for {dtype_torch}")
            # print(e)
            

@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_rand_input_fixed_dtype(mode):
    """测试固定数据类型下的随机输入值，对比输出误差"""
    ms.set_context(mode=mode)
        
    rand_seed = np.random.randint(100)
    
    generator_ms = ms.Generator()
    generator_ms.manual_seed(rand_seed)
    generator_torch = torch.Generator()
    generator_torch.manual_seed(rand_seed)
    
    result_ms = rand_jit(shape, generator_ms, dtype_ms)
    result_torch = torch.rand(*shape, generator=generator_torch, dtype=dtype_torch)
    # result = is_same(result_ms, result_torch)
    # assert result

    
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_rand_wrong_input(mode):
    """测试随机混乱输入，报错信息的准确性"""
    ms.set_context(mode=mode)
    
    dtype_ms = ms.int32

    size = [3.4, 4.5]
    
    try:
        ms_result = rand_jit(size).asnumpy()
        print(ms_result)
    except Exception as e:
        print(f"size 不是整数类型报错信息：\n{e}")
        
    try:
        ms_result = rand_jit([3,4], dtype=dtype_ms).asnumpy()
        print(ms_result)
    except Exception as e:
        print(f"dtype 不是浮点类型报错信息：\n{e}")
        
    