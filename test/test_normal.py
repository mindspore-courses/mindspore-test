import mindspore as ms
import numpy as np
from mindspore import mint, Tensor, ops, Parameter
import torch

#normal:根据正态（高斯）随机数分布生成随机数。

#1.对应Pytorch 的相应接口进行测试：
#a) 测试random输入不同dtype，对比两个框架的支持度
def test_normal_dtype():
    print("--->This function: test mindspore.mint.normal [dtype]<---")

    np_dtypes = [np.int8, np.int16, np.int32, np.int64,
                 np.float16, np.float32, np.float64,
                 np.uint8, np.uint16, np.uint32, np.uint64,
                 np.complex64, np.complex128, np.bool_,
                'bfloat16']

    for dtype in np_dtypes:
        if dtype == np.bool_:
            mean_np = np.random.choice(np.array([True, False]), size=3)
            std_np = np.random.choice(np.array([True, False]), size=3)
            mean_torch = torch.from_numpy(mean_np)
            std_torch = torch.from_numpy(std_np)
            mean_ms = Tensor.from_numpy(mean_np)
            std_ms = Tensor.from_numpy(std_np)
        elif dtype == 'bfloat16':
            mean_np = np.random.rand(3).astype(np.float32)
            std_np = np.random.rand(3).astype(np.float32)
            mean_torch = torch.from_numpy(mean_np).type(torch.bfloat16)
            std_torch = torch.from_numpy(std_np).type(torch.bfloat16)
            mean_ms = Tensor.from_numpy(mean_np).astype(ms.bfloat16)
            std_ms = Tensor.from_numpy(std_np).astype(ms.bfloat16)
        elif (dtype == np.float16 or dtype == np.float32 or dtype == np.float64 or 
        dtype == np.complex64 or dtype == np.complex128):
            mean_np = np.random.rand(3).astype(dtype)
            std_np = np.random.rand(3).astype(dtype)
            mean_torch = torch.from_numpy(mean_np)
            std_torch = torch.from_numpy(std_np)
            mean_ms = Tensor.from_numpy(mean_np)
            std_ms = Tensor.from_numpy(std_np)
        else:
            mean_np = np.random.randint(0, 10, size=3, dtype=dtype)
            std_np = np.random.randint(0, 10, size=3, dtype=dtype)
            mean_torch = torch.from_numpy(mean_np)
            std_torch = torch.from_numpy(std_np)
            mean_ms = Tensor.from_numpy(mean_np)
            std_ms = Tensor.from_numpy(std_np)
        
        torch_support = True
        ms_support = True
        #test
        try:
            normal_torch = torch.normal(mean=mean_torch, std=std_torch)
            print(f"<torch_result>: {normal_torch}")
        except:
            torch_support = False

        try:
            normal_ms = mint.normal(mean=mean_ms, std=std_ms)
            print(f"<torch_result>: {normal_ms}")
        except:
            ms_support = False

        #result
        if torch_support == ms_support == True:
            print(f"[dtype] BOTH SUPPORT (from test_normal_dtype): "
                  f"torch.normal support {normal_torch.dtype} "
                  f"& mindspore.mint.normal support {normal_ms.dtype}.")
        elif torch_support == ms_support == False:
            print(f"[dtype] BOTH NOT SUPPORT (from test_normal_dtype): "
                  f"torch.normal does not support {mean_torch.dtype} "
                  f"& mindspore.mint.normal does not support {mean_ms.dtype}.")
        elif torch_support == True and ms_support == False:
            print(f"[dtype] ONLY TORCH (from test_normal_dtype): "
                  f"torch.normal support {normal_torch.dtype} "
                  f"but mindspore.mint.normal NOT.")
        else:
            print(f"[dtype] ONLY MS (from test_normal_dtype): "
                  f"mindspore.mint.normal support {normal_ms.dtype} "
                  f"but torch.normal NOT.")

        print('='*50)

#b) 测试固定dtype，random输入值，对比两个框架输出是否相等（误差范围为小于1e-3）
def test_normal_output():
    print("--->This function: test mindspore.mint.normal [output]<---")
    #函数本身是随机生成，无法进行误差比较

#c) 测试固定shape，固定输入值，不同输入参数（string\bool等类型），两个框架的支持度
def test_normal_paramType():
    
    print("--->This function: test mindspore.mint.normal [paramType]<---")

    #normal的size参数只在mean和std都为常量float时起作用
    mean = 1.0
    std = 2.0

    params = (("abc", 'string'), 
              (True, 'bool'), 
              (0, 'int'), 
              (1.0, 'float'), 
              ([1, 2], 'list'),
              ((1, 2), 'tuple'),
              )

    for (param, param_type) in params:
        #test
        torch_support = True
        ms_support = True

        try:
            normal_torch = torch.normal(mean=mean, std=std, size=param)
        except:
            torch_support = False

        try:
            normal_ms = mint.normal(mean=mean, std=std, size=param)
        except:
            ms_support = False

        #result
        if torch_support == ms_support == True:
            print(f"[paramType: {param_type}] BOTH SUPPORT(from test_normal_paramType).")
        elif torch_support == ms_support == False:
            print(f"[paramType: {param_type}] BOTH NOT SUPPORT(from test_normal_paramType).")
        elif torch_support == True and ms_support == False:
            print(f"[paramType: {param_type}] ONLY TORCH(from test_normal_paramType): "
                  "torch.normal support but mindspore.mint.normal NOT.")
        else:
            print(f"[paramType: {param_type}] ONLY MS(from test_normal_paramType): "
                  "mindspore.mint.normal support but torch.normal NOT.")

        print('='*50)

#d) 测试随机混乱输入，报错信息的准确性
def test_normal_errorMessage():
    print("--->This function: test mindspore.mint.normal [errorMessage]<---")
    
    #input mean TypeError
    try:
        mean_ms1 = True
        std_np1 = np.random.rand(3).astype(np.float32)
        std_ms1 = Tensor.from_numpy(std_np1)
        normal_ms1 = mint.normal(mean=mean_ms1, std=std_ms1)
        print(normal_ms1)
    except Exception as e:
        print("[errorMessage] ACCURACY TEST (from test_normal_errorMessage): "
              "Test Target: input mean type error-TypeError.\n"
              f"Catch Exception:\n{type(e).__name__}: {e}.")
    print('='*50)

    #input std TypeError
    try:
        mean_np2 = np.random.rand(3).astype(np.float32)
        mean_ms2 = Tensor.from_numpy(mean_np2)
        std_ms2 = True
        normal_ms2 = mint.normal(mean=mean_ms2, std=std_ms2)
        print(normal_ms2)
    except Exception as e:
        print("[errorMessage] ACCURACY TEST (from test_normal_errorMessage): "
              "Test Target: input std type error-TypeError.\n"
              f"Catch Exception:\n{type(e).__name__}: {e}.")
    print('='*50)

#2. 测试使用接口构造函数/神经网络的准确性
#a) Github搜索带有该接口的代码片段/神经网络
#b) 使用Pytorch和MindSpore，固定输入和权重，测试正向推理结果（误差范围小于1e-3，若报错则记录报错信息）
#c) 测试该神经网络/函数反向，如果是神经网络，则测试Parameter的梯度，如果是函数，则测试函数输入的梯度

#函数本身随机，无法测试第二部分内容


if __name__ == '__main__':
    test_normal_dtype()
    test_normal_output()
    test_normal_paramType()
    test_normal_errorMessage()