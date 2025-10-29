import mindspore as ms
import numpy as np
from mindspore import mint, Tensor, ops, nn
import torch
import torch.nn as nn_torch

#triu:返回输入Tensor input 的上三角形部分(包含对角线和下面的元素)，并将其他元素设置为0。

#1.对应Pytorch 的相应接口进行测试：
#a) 测试random输入不同dtype，对比两个框架的支持度
def test_triu_dtype():
    print("--->This function: test mindspore.mint.triu [dtype]<---")

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
        else :
            input_np = np.random.randint(0, 10, size=(3, 3), dtype=dtype)
            input_torch = torch.from_numpy(input_np)
            input_ms = Tensor.from_numpy(input_np)

        torch_support = True
        ms_support = True
        #test
        try:
            triu_torch = torch.triu(input_torch)
            print(f"<torch_result>: {triu_torch}")
        except:
            torch_support = False
        
        try:
            triu_ms = mint.triu(input_ms)
            print(f"<ms_result>: {triu_ms}")
        except:
            ms_support = False
        
        #result
        if torch_support == ms_support == True:
            print(f"[dtype] BOTH SUPPORT (from test_triu_dtype): "
                  f"torch.triu support {triu_torch.dtype} "
                  f"& mindspore.mint.triu support {triu_ms.dtype}.")
        elif torch_support == ms_support == False:
            print(f"[dtype] BOTH NOT SUPPORT (from test_triu_dtype): "
                  f"torch.triu does not support {input_torch.dtype} "
                  f"& mindspore.mint.triu does not support {input_ms.dtype}.")
        elif torch_support == True and ms_support == False:
            print(f"[dtype] ONLY TORCH (from test_triu_dtype): "
                  f"torch.triu support {triu_torch.dtype} "
                  f"but mindspore.mint.triu NOT.")
        else:
            print(f"[dtype] ONLY MS (from test_triu_dtype): "
                  f"mindspore.mint.triu support {triu_ms.dtype} "
                  f"but torch.triu NOT.")

        print('='*50)

#b) 测试固定dtype，random输入值，对比两个框架输出是否相等（误差范围为小于1e-3）
def test_triu_output():
    print("--->This function: test mindspore.mint.triu [output]<---")
    #共同支持以下类型
    np_dtypes = [np.int8, np.int16, np.int32, np.int64,
                 np.float16, np.float32, np.float64,
                 np.uint8, np.bool_, 'bfloat16']
    
    for dtype in np_dtypes:
        if dtype == np.bool_:
            input_np = np.random.choice(np.array([True, False]), size=(3, 3))
            input_torch = torch.from_numpy(input_np)
            input_ms = Tensor.from_numpy(input_np)
        elif dtype == 'bfloat16':
            input_np = np.random.rand(3, 3).astype(np.float32)
            input_torch = torch.from_numpy(input_np).type(torch.bfloat16)
            input_ms = Tensor.from_numpy(input_np).astype(ms.bfloat16)
        elif (dtype == np.float16 or dtype == np.float32 or dtype == np.float64):
            input_np = np.random.rand(3, 3).astype(dtype)
            input_torch = torch.from_numpy(input_np)
            input_ms = Tensor.from_numpy(input_np)
        else :
            input_np = np.random.randint(0, 10, size=(3, 3), dtype=dtype)
            input_torch = torch.from_numpy(input_np)
            input_ms = Tensor.from_numpy(input_np)

        triu_torch = torch.triu(input_torch)
        triu_ms = mint.triu(input_ms)
        #numpy中没有bfloat16类型
        if dtype == 'bfloat16':
            print("current type: bfloat16\n"
                 f"<result_torch>: {triu_torch}\n"
                 f"<result_torch>: {triu_ms}")
        else:
            if np.allclose(triu_torch.numpy(), triu_ms.asnumpy(), atol=1e-3):
                print("[output] WITHIN TOLERANCE (from test_triu_output): "
                      "output discrepancy between torch.triu "
                      "mindspore.mint.triu is less than 1e-3.")
            else:
                print("[output] BEYOND TOLERANCE (from test_triu_output): "
                      "output discrepancy is beyond tolerance.")
        
        print('='*50)

#c) 测试固定shape，固定输入值，不同输入参数（string\bool等类型），两个框架的支持度
def test_triu_paramType():
    print("--->This function: test mindspore.mint.triu [paramType]<---")

    input_np = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.float32)

    input_torch = torch.from_numpy(input_np)
    input_ms = Tensor.from_numpy(input_np)

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
            triu_torch = torch.triu(input_torch, param)
        except:
            torch_support = False

        try:
            triu_ms = mint.triu(input_ms, param)
        except:
            ms_support = False

        #result
        if torch_support == ms_support == True:
            print(f"[paramType: {param_type}] BOTH SUPPORT(from test_triu_paramType).")
        elif torch_support == ms_support == False:
            print(f"[paramType: {param_type}] BOTH NOT SUPPORT(from test_triu_paramType).")
        elif torch_support == True and ms_support == False:
            print(f"[paramType: {param_type}] ONLY TORCH(from test_triu_paramType): "
                  "torch.triu support but mindspore.mint.triu NOT.")
        else:
            print(f"[paramType: {param_type}] ONLY MS(from test_triu_paramType): "
                  "mindspore.mint.triu support but torch.triu NOT.")

        print('='*50)

#d) 测试随机混乱输入，报错信息的准确性
def test_triu_errorMessage():
    print("--->This function: test mindspore.mint.triu [errorMessage]<---")

    #input shape ValueError
    try:
        input_np1 = np.random.rand(3).astype(np.float32)
        input_ms1 = Tensor.from_numpy(input_np1)
        diagonal = 0
        triu_ms1 = mint.triu(input_ms1, diagonal=diagonal)
        print(triu_ms1)
    except Exception as e:
        print("[errorMessage] ACCURACY TEST (from test_triu_errorMessage): "
              "Test Target: input tensor shape error-ValueError.\n"
              f"Catch Exception:\n{type(e).__name__}: {e}.")
    print('='*50)
    
    #param diagonal TypeError
    try:
        input_np2 = np.random.rand(3, 3).astype(np.float32)
        input_ms2 = Tensor.from_numpy(input_np2)
        diagonal = False
        triu_ms2 = mint.triu(input_ms2, diagonal=diagonal)
        print(triu_ms2)
    except Exception as e:
        print("[errorMessage] ACCURACY TEST (from test_triu_errorMessage): "
              "Test Target: param diagonal type error-TypeError.\n"
              f"Catch Exception:\n{type(e).__name__}: {e}.")
    print('='*50)
    #input tensor TypeError
    
    try:
        input_ms3 = (1, 1)
        diagonal = 0
        triu_ms3 = mint.triu(input_ms3, diagonal=diagonal)
        print(triu_ms3)
    except Exception as e:
        print("[errorMessage] ACCURACY TEST (from test_triu_errorMessage): "
              "Test Target: input tensor type error-TypeError.\n"
              f"Catch Exception:\n{type(e).__name__}: {e}.")
    print('='*50)

#2. 测试使用接口构造函数/神经网络的准确性
#a) Github搜索带有该接口的代码片段/神经网络
#b) 使用Pytorch和MindSpore，固定输入和权重，测试正向推理结果（误差范围小于1e-3，若报错则记录报错信息）
#c) 测试该神经网络/函数反向，如果是神经网络，则测试Parameter的梯度，如果是函数，则测试函数输入的梯度
def test_triu_value_and_grad():  
    def gaussian_tailbound(d, b):
        return ( d + 2*( d * math.log(1/b) )**0.5 + 2*math.log(1/b) )**0.5
    
    def test_torch(Y):
        rho = 5
        n = 5
        d = 0.5
        X = torch.tensor([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5], [7.5, 8.5, 9.5]], dtype=torch.float32)
        A = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
        W = torch.mm(X, A)
        gamma = gaussian_tailbound(d, 0.1)
        noise_var = (gamma**4/(rho*n**2))
        Y = Y * np.sqrt(noise_var)    
        Y = torch.triu(Y)
        Y = Y + Y.t() - torch.diag_embed(Y.diagonal()) #Don't duplicate diagonal entries
        Z = (torch.mm(W.t(), W))/n
        #add noise    
        Z = Z + Y
        return Z
    
    def test_ms(Y):
        rho = 5
        n = 5
        d = 0.5
        X = Tensor([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5], [7.5, 8.5, 9.5]], dtype=ms.float32)
        A = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=ms.float32)
        W = ops.mm(X, A)
        gamma = gaussian_tailbound(d, 0.1)
        noise_var = (gamma**4/(rho*n**2))
        Y = Y * np.sqrt(noise_var)     
        Y = mint.triu(Y)
        Y = Y + Y.t() - ops.diag_embed(Y.diagonal()) #Don't duplicate diagonal entries
        Z = (ops.mm(W.t(), W))/n
        #add noise    
        Z = Z + Y
        return Z
    
    Y_np = np.random.randn(3, 3).astype(np.float32)
    Y_torch = torch.from_numpy(Y_np)
    Y_torch.requires_grad = True
    Y_ms = Parameter(Tensor.from_numpy(Y_np))

    y_torch = test_torch(Y_torch)
    y_ms = test_ms(Y_ms)

    print(f"<value torch> {y_torch}\n"
          f"<value_ms> {y_ms}")
    if np.allclose(y_torch.detach().numpy(), y_ms.asnumpy(), atol=1e-3):
        print("[value] WITHIN TOLERANCE (from test_stack_value_and_grad): "
              "value discrepancy between torch.stack "
              "mindspore.mint.stack is less than 1e-3.")
    else:
        print("[value] BEYOND TOLERANCE (from test_stack_value_and_grad): "
              "value discrepancy is beyond tolerance.")
    print('='*50)
    
    y_sum = y_torch.sum()
    y_sum.backward()
    grad_ms = ms.grad(test_ms, grad_position=0)(Y_ms)
    print(f"<grad torch> {Y_torch.grad}\n"
          f"<grad> {grad_ms}")
    if np.allclose(Y_torch.grad.numpy(), grad_ms.asnumpy(), atol=1e-3):
        print("[grad] WITHIN TOLERANCE (from test_stack_value_and_grad): "
              "grad discrepancy between torch.stack "
              "mindspore.mint.stack is less than 1e-3.")
    else:
        print("[grad] BEYOND TOLERANCE (from test_stack_value_and_grad): "
              "grad discrepancy is beyond tolerance.")
    print('='*50)
    
if __name__ == '__main__':
    test_triu_dtype()
    test_triu_output()
    test_triu_paramType()
    test_triu_errorMessage()
    test_triu_value_and_grad()