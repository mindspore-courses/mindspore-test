import mindspore as ms
import numpy as np
from mindspore import mint, Tensor, ops, Parameter
import torch

#stack:在指定轴上对输入Tensor序列进行堆叠。

#1.对应Pytorch 的相应接口进行测试：
#a) 测试random输入不同dtype，对比两个框架的支持度
def test_stack_dtype():
    print("--->This function: test mindspore.mint.stack [dtype]<---")

    np_dtypes = [np.int8, np.int16, np.int32, np.int64,
                 np.float16, np.float32, np.float64,
                 np.uint8, np.uint16, np.uint32, np.uint64,
                 np.complex64, np.complex128, np.bool_,
                'bfloat16']

    for dtype in np_dtypes:
        if dtype == np.bool_:
            input_np1 = np.random.choice(np.array([True, False]), size=(3, 3))
            input_np2 = np.random.choice(np.array([True, False]), size=(3, 3))
            input_torch1 = torch.from_numpy(input_np1)
            input_ms1 = Tensor.from_numpy(input_np1)
            input_torch2 = torch.from_numpy(input_np2)
            input_ms2 = Tensor.from_numpy(input_np2)
        elif dtype == 'bfloat16':
            input_np1 = np.random.rand(3, 3).astype(np.float32)
            input_np2 = np.random.rand(3, 3).astype(np.float32)
            input_torch1 = torch.from_numpy(input_np1).type(torch.bfloat16)
            input_ms1 = Tensor.from_numpy(input_np1).astype(ms.bfloat16)
            input_torch2 = torch.from_numpy(input_np2).type(torch.bfloat16)
            input_ms2 = Tensor.from_numpy(input_np2).astype(ms.bfloat16)
        elif (dtype == np.float16 or dtype == np.float32 or dtype == np.float64 or 
        dtype == np.complex64 or dtype == np.complex128):
            input_np1 = np.random.rand(3, 3).astype(dtype)
            input_np2 = np.random.rand(3, 3).astype(dtype)
            input_torch1 = torch.from_numpy(input_np1)
            input_ms1 = Tensor.from_numpy(input_np1)
            input_torch2 = torch.from_numpy(input_np2)
            input_ms2 = Tensor.from_numpy(input_np2)
        else:
            input_np1 = np.random.randint(0, 10, size=(3, 3), dtype=dtype)
            input_np2 = np.random.randint(0, 10, size=(3, 3), dtype=dtype)
            input_torch1 = torch.from_numpy(input_np1)
            input_ms1 = Tensor.from_numpy(input_np1)
            input_torch2 = torch.from_numpy(input_np2)
            input_ms2 = Tensor.from_numpy(input_np2)

        torch_support = True
        ms_support = True
        #test
        try:
            stack_torch = torch.stack([input_torch1, input_torch2], dim=0)
            print(f"<torch_result>: {stack_torch}")
        except:
            torch_support = False

        try:
            stack_ms = mint.stack([input_ms1, input_ms2], dim=0)
            print(f"<torch_result>: {stack_ms}")
        except:
            ms_support = False

        #result
        if torch_support == ms_support == True:
            print(f"[dtype] BOTH SUPPORT (from test_stack_dtype): "
                  f"torch.stack support {stack_torch.dtype} "
                  f"& mindspore.mint.stack support {stack_ms.dtype}.")
        elif torch_support == ms_support == False:
            print(f"[dtype] BOTH NOT SUPPORT (from test_stack_dtype): "
                  f"torch.stack does not support {input_torch1.dtype} "
                  f"& mindspore.mint.stack does not support {input_ms1.dtype}.")
        elif torch_support == True and ms_support == False:
            print(f"[dtype] ONLY TORCH (from test_stack_dtype): "
                  f"torch.stack support {stack_torch.dtype} "
                  f"but mindspore.mint.stack NOT.")
        else:
            print(f"[dtype] ONLY MS (from test_stack_dtype): "
                  f"mindspore.mint.stack support {stack_ms.dtype} "
                  f"but torch.stack NOT.")

        print('='*50)

#b) 测试固定dtype，random输入值，对比两个框架输出是否相等（误差范围为小于1e-3）
def test_stack_output():
    print("--->This function: test mindspore.mint.stack [output]<---")
    #共同支持以下类型
    np_dtypes = [np.int8, np.int16, np.int32, np.int64,
                 np.float16, np.float32, np.float64,
                 np.uint8, np.uint16, np.uint32, np.uint64,
                 np.complex64, np.complex128, np.bool_,
                'bfloat16']
    
    for dtype in np_dtypes:
        if dtype == np.bool_:
            input_np1 = np.random.choice(np.array([True, False]), size=(3, 3))
            input_np2 = np.random.choice(np.array([True, False]), size=(3, 3))
            input_torch1 = torch.from_numpy(input_np1)
            input_ms1 = Tensor.from_numpy(input_np1)
            input_torch2 = torch.from_numpy(input_np2)
            input_ms2 = Tensor.from_numpy(input_np2)
        elif dtype == 'bfloat16':
            input_np1 = np.random.rand(3, 3).astype(np.float32)
            input_np2 = np.random.rand(3, 3).astype(np.float32)
            input_torch1 = torch.from_numpy(input_np1).type(torch.bfloat16)
            input_ms1 = Tensor.from_numpy(input_np1).astype(ms.bfloat16)
            input_torch2 = torch.from_numpy(input_np2).type(torch.bfloat16)
            input_ms2 = Tensor.from_numpy(input_np2).astype(ms.bfloat16)
        elif (dtype == np.float16 or dtype == np.float32 or dtype == np.float64 or 
        dtype == np.complex64 or dtype == np.complex128):
            input_np1 = np.random.rand(3, 3).astype(dtype)
            input_np2 = np.random.rand(3, 3).astype(dtype)
            input_torch1 = torch.from_numpy(input_np1)
            input_ms1 = Tensor.from_numpy(input_np1)
            input_torch2 = torch.from_numpy(input_np2)
            input_ms2 = Tensor.from_numpy(input_np2)
        else:
            input_np1 = np.random.randint(0, 10, size=(3, 3), dtype=dtype)
            input_np2 = np.random.randint(0, 10, size=(3, 3), dtype=dtype)
            input_torch1 = torch.from_numpy(input_np1)
            input_ms1 = Tensor.from_numpy(input_np1)
            input_torch2 = torch.from_numpy(input_np2)
            input_ms2 = Tensor.from_numpy(input_np2)

        stack_torch = torch.stack([input_torch1, input_torch2], dim=0)
        stack_ms = mint.stack([input_ms1, input_ms2], dim=0)
        
        if dtype == 'bfloat16':
            print("current type: bfloat16\n"
                 f"<result_torch>: {stack_torch}\n"
                 f"<result_torch>: {stack_ms}")
        else:
            if np.allclose(stack_torch.numpy(), stack_ms.asnumpy(), atol=1e-3):
                print("[output] WITHIN TOLERANCE (from test_stack_output): "
                      "output discrepancy between torch.stack "
                      "mindspore.mint.stack is less than 1e-3.")
            else:
                print("[output] BEYOND TOLERANCE (from test_stack_output): "
                      "output discrepancy is beyond tolerance.")
        
        print('='*50)

#c) 测试固定shape，固定输入值，不同输入参数（string\bool等类型），两个框架的支持度
def test_stack_paramType():
    print("--->This function: test mindspore.mint.stack [paramType]<---")

    input_np1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.float32)
    input_torch1 = torch.from_numpy(input_np1)
    input_ms1 = Tensor.from_numpy(input_np1)

    input_np2= np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5], [7.5, 8.5, 9.5]]).astype(np.float32)
    input_torch2 = torch.from_numpy(input_np2)
    input_ms2 = Tensor.from_numpy(input_np2)

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
            stack_torch = torch.stack([input_torch1, input_torch2], dim=param)
        except:
            torch_support = False

        try:
            stack_ms = mint.stack([input_ms1, input_ms2], dim=param)
        except:
            ms_support = False

        #result
        if torch_support == ms_support == True:
            print(f"[paramType: {param_type}] BOTH SUPPORT(from test_stack_paramType).")
        elif torch_support == ms_support == False:
            print(f"[paramType: {param_type}] BOTH NOT SUPPORT(from test_stack_paramType).")
        elif torch_support == True and ms_support == False:
            print(f"[paramType: {param_type}] ONLY TORCH(from test_stack_paramType): "
                  "torch.stack support but mindspore.mint.stack NOT.")
        else:
            print(f"[paramType: {param_type}] ONLY MS(from test_stack_paramType): "
                  "mindspore.mint.stack support but torch.stack NOT.")

        print('='*50)

#d) 测试随机混乱输入，报错信息的准确性
def test_stack_errorMessage():
    print("--->This function: test mindspore.mint.stack [errorMessage]<---")

    #input tensors type not same TypeError
    try:
        input_np1 = np.random.rand(3, 3).astype(np.float32)
        input_ms1 = Tensor.from_numpy(input_np1)
        input_np2 = np.random.rand(3, 3).astype(np.int32)
        input_ms2 = Tensor.from_numpy(input_np2)
        dim1 = 0
        stack_ms1 = mint.stack([input_ms1, input_ms2], dim=dim1)
        print(stack_ms1)
    except Exception as e:
        print("[errorMessage] ACCURACY TEST (from test_stack_errorMessage): "
              "Test Target: input tensors type not same error-TypeError.\n"
              f"Catch Exception:\n{type(e).__name__}: {e}.")
    print('='*50)
    
    #param dim out of range ValueError
    try:
        input_np3 = np.random.rand(3, 3).astype(np.float32)
        input_ms3 = Tensor.from_numpy(input_np3)
        input_np4 = np.random.rand(3, 3).astype(np.float32)
        input_ms4 = Tensor.from_numpy(input_np4)
        dim2 = 4
        stack_ms2 = mint.stack([input_ms3, input_ms4], dim=dim2)
        print(stack_ms2)
    except Exception as e:
        print("[errorMessage] ACCURACY TEST (from test_stack_errorMessage): "
              "Test Target: param dim value out of range error-ValueError.\n"
              f"Catch Exception:\n{type(e).__name__}: {e}.")
    print('='*50)

    #input tensors shape not same ValueError
    try:
        input_np5 = np.random.rand(3).astype(np.float32)
        input_ms5 = Tensor.from_numpy(input_np5)
        input_np6 = np.random.rand(3, 3).astype(np.float32)
        input_ms6 = Tensor.from_numpy(input_np6)
        dim3 = 0
        stack_ms3 = mint.stack([input_ms5, input_ms6], dim=dim3)
        print(stack_ms3)
    except Exception as e:
        print("[errorMessage] ACCURACY TEST (from test_stack_errorMessage): "
              "Test Target: input tensors shape not same error-ValueError.\n"
              f"Catch Exception:\n{type(e).__name__}: {e}.")
    print('='*50)

#2. 测试使用接口构造函数/神经网络的准确性
#a) Github搜索带有该接口的代码片段/神经网络
#b) 使用Pytorch和MindSpore，固定输入和权重，测试正向推理结果（误差范围小于1e-3，若报错则记录报错信息）
#c) 测试该神经网络/函数反向，如果是神经网络，则测试Parameter的梯度，如果是函数，则测试函数输入的梯度
def test_stack_value_and_grad():
    w_np = np.random.randn(3, 1).astype(np.float32)
    def model_torch(x):
        w = torch.from_numpy(w_np)
        f = torch.stack([x * x, x, torch.ones_like(x)], 1)
        yhat = torch.squeeze(f @ w, 2)
        return yhat
    def model_ms(x):
        w = Tensor.from_numpy(w_np)
        f = mint.stack([x * x, x, ops.ones_like(x)], 1)
        yhat = ops.squeeze(ops.matmul(f, w), 2)
        return yhat
    x_torch = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32, requires_grad=True)
    x_ms = Parameter(Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=ms.float32))

    y_torch = model_torch(x_torch)
    y_ms = model_ms(x_ms)
    
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
    grad_ms = ms.grad(model_ms, grad_position=0)(x_ms)
    print(f"<grad torch> {x_torch.grad}\n"
          f"<grad> {grad_ms}")
    if np.allclose(x_torch.grad.numpy(), grad_ms.asnumpy(), atol=1e-3):
        print("[grad] WITHIN TOLERANCE (from test_stack_value_and_grad): "
              "grad discrepancy between torch.stack "
              "mindspore.mint.stack is less than 1e-3.")
    else:
        print("[grad] BEYOND TOLERANCE (from test_stack_value_and_grad): "
              "grad discrepancy is beyond tolerance.")
    print('='*50)
            
if __name__ == '__main__':
    test_stack_dtype()
    test_stack_output()
    test_stack_paramType()
    test_stack_errorMessage()
    test_stack_value_and_grad()