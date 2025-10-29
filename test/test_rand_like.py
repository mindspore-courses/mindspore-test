import mindspore as ms
import numpy as np
from mindspore import mint, Tensor, ops, Parameter
import torch

#rand_like:返回shape与输入相同，类型为 dtype 的Tensor，dtype由输入决定，其元素取值服从区间内的均匀分布。

#1.对应Pytorch 的相应接口进行测试：
#a) 测试random输入不同dtype，对比两个框架的支持度
def test_rand_like_dtype():
    print("--->This function: test mindspore.mint.rand_like [dtype]<---")

    np_dtypes = [np.int8, np.int16, np.int32, np.int64,
                 np.float16, np.float32, np.float64,
                 np.uint8, np.uint16, np.uint32, np.uint64,
                 np.complex64, np.complex128, np.bool_,
                'bfloat16']

    for dtype in np_dtypes:
        if dtype == np.bool_:
            input_np = np.random.choice(np.array([True, False]), size=(3, 3))
            input_torch = torch.from_numpy(input_np)
            input_ms = Tensor.from_numpy(input_np)
        elif dtype == 'bfloat16':
            input_np = np.random.rand(3, 3).astype(np.float32)
            input_torch = torch.from_numpy(input_np).type(torch.bfloat16)
            input_ms = Tensor.from_numpy(input_np).astype(ms.bfloat16)
        elif (dtype == np.float16 or dtype == np.float32 or dtype == np.float64 or 
        dtype == np.complex64 or dtype == np.complex128):
            input_np = np.random.rand(3, 3).astype(dtype)
            input_torch = torch.from_numpy(input_np)
            input_ms = Tensor.from_numpy(input_np)
        else:
            input_np = np.random.randint(0, 10, size=(3, 3), dtype=dtype)
            input_torch = torch.from_numpy(input_np)
            input_ms = Tensor.from_numpy(input_np)

        torch_support = True
        ms_support = True
        #test
        try:
            rand_like_torch = torch.rand_like(input_torch)
            print(f"<torch_result>: {rand_like_torch}")
        except:
            torch_support = False

        try:
            rand_like_ms = mint.rand_like(input_ms)
            print(f"<torch_result>: {rand_like_ms}")
        except:
            ms_support = False

        #result
        if torch_support == ms_support == True:
            print(f"[dtype] BOTH SUPPORT (from test_rand_like_dtype): "
                  f"torch.rand_like support {rand_like_torch.dtype} "
                  f"& mindspore.mint.rand_like support {rand_like_ms.dtype}.")
        elif torch_support == ms_support == False:
            print(f"[dtype] BOTH NOT SUPPORT (from test_rand_like_dtype): "
                  f"torch.rand_like does not support {input_torch.dtype} "
                  f"& mindspore.mint.rand_like does not support {input_ms.dtype}.")
        elif torch_support == True and ms_support == False:
            print(f"[dtype] ONLY TORCH (from test_rand_like_dtype): "
                  f"torch.rand_like support {rand_like_torch.dtype} "
                  f"but mindspore.mint.rand_like NOT.")
        else:
            print(f"[dtype] ONLY MS (from test_rand_like_dtype): "
                  f"mindspore.mint.rand_like support {rand_like_ms.dtype} "
                  f"but torch.rand_like NOT.")

        print('='*50)

#b) 测试固定dtype，random输入值，对比两个框架输出是否相等（误差范围为小于1e-3）
def test_rand_like_output():
    print("--->This function: test mindspore.mint.rand_like [output]<---")
    #函数本身是随机生成，无法进行误差比较

#c) 测试固定shape，固定输入值，不同输入参数（string\bool等类型），两个框架的支持度
def test_rand_like_paramType():
    print("--->This function: test mindspore.mint.rand_like [paramType]<---")

    input_np = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.float32)

    input_torch = torch.from_numpy(input_np)
    input_ms = Tensor.from_numpy(input_np)
    #只能dtype类型
    params = (("abc", 'string'), 
              (True, 'bool'), 
              (0, 'int'), 
              (1.0, 'float'), 
              ([1, 2], 'list'),
              ((0, 1), 'tuple')
              )

    for (param, param_type) in params:
        #test
        torch_support = True
        ms_support = True
        try:
            rand_like_torch = torch.rand_like(input=input_torch, dtype=param)
        except:
            torch_support = False

        try:
            rand_like_ms = mint.rand_like(input=input_ms, dtype=param)
        except:
            ms_support = False

        #result
        if torch_support == ms_support == True:
            print(f"[paramType: {param_type}] BOTH SUPPORT(from test_rand_like_paramType).")
        elif torch_support == ms_support == False:
            print(f"[paramType: {param_type}] BOTH NOT SUPPORT(from test_rand_like_paramType).")
        elif torch_support == True and ms_support == False:
            print(f"[paramType: {param_type}] ONLY TORCH(from test_rand_like_paramType): "
                  "torch.rand_like support but mindspore.mint.rand_like NOT.")
        else:
            print(f"[paramType: {param_type}] ONLY MS(from test_rand_like_paramType): "
                  "mindspore.mint.rand_like support but torch.rand_like NOT.")

        print('='*50)

#d) 测试随机混乱输入，报错信息的准确性
def test_rand_like_errorMessage():
    print("--->This function: test mindspore.mint.rand_like [errorMessage]<---")

    #dtype ValueError
    try:
        input_np1 = np.random.rand(3, 3).astype(np.float32)
        input_ms1 = Tensor.from_numpy(input_np1)
        dtype = ms.int32
        rand_like_ms1 = mint.rand_like(input=input_ms1, dtype=dtype)
        print(rand_like_ms1)
    except Exception as e:
        print("[errorMessage] ACCURACY TEST (from test_rand_like_errorMessage): "
              "Test Target: dtype error-ValueError.\n"
              f"Catch Exception:\n{type(e).__name__}: {e}.")
    print('='*50)


#2. 测试使用接口构造函数/神经网络的准确性
#a) Github搜索带有该接口的代码片段/神经网络
#b) 使用Pytorch和MindSpore，固定输入和权重，测试正向推理结果（误差范围小于1e-3，若报错则记录报错信息）
#c) 测试该神经网络/函数反向，如果是神经网络，则测试Parameter的梯度，如果是函数，则测试函数输入的梯度

#函数本身随机，无法测试第二部分内容

if __name__ == '__main__':
    test_rand_like_dtype()
    test_rand_like_output()
    test_rand_like_paramType()
    test_rand_like_errorMessage()