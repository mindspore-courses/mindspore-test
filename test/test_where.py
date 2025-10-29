import mindspore as ms
import numpy as np
from mindspore import mint, Tensor, ops, Parameter
import torch

#where:返回一个Tensor，Tensor的元素从 input 或 other 中根据 condition 选择。

#1.对应Pytorch 的相应接口进行测试：
#a) 测试random输入不同dtype，对比两个框架的支持度
def test_where_dtype():
    print("--->This function: test mindspore.mint.where [dtype]<---")

    np_dtypes = [np.int8, np.int16, np.int32, np.int64,
                 np.float16, np.float32, np.float64,
                 np.uint8, np.uint16, np.uint32, np.uint64,
                 np.complex64, np.complex128, np.bool_,
                'bfloat16']

    for dtype in np_dtypes:
        condition_np = np.array(True)
        condition_torch = torch.from_numpy(condition_np)
        condition_ms = Tensor.from_numpy(condition_np)
        if dtype == np.bool_:
            input_np = np.random.choice(np.array([True, False]), size=(3, 3))
            other_np = np.random.choice(np.array([True, False]), size=(3, 3))
            input_torch = torch.from_numpy(input_np)
            input_ms = Tensor.from_numpy(input_np)
            other_torch = torch.from_numpy(other_np)
            other_ms = Tensor.from_numpy(other_np)
        elif dtype == 'bfloat16':
            input_np = np.random.rand(3, 3).astype(np.float32)
            other_np = np.random.rand(3, 3).astype(np.float32)
            input_torch = torch.from_numpy(input_np).type(torch.bfloat16)
            input_ms = Tensor.from_numpy(input_np).astype(ms.bfloat16)
            other_torch = torch.from_numpy(other_np).type(torch.bfloat16)
            other_ms = Tensor.from_numpy(other_np).astype(ms.bfloat16)
        elif (dtype == np.float16 or dtype == np.float32 or dtype == np.float64 or 
        dtype == np.complex64 or dtype == np.complex128):
            input_np = np.random.rand(3, 3).astype(dtype)
            other_np = np.random.rand(3, 3).astype(dtype)
            input_torch = torch.from_numpy(input_np)
            input_ms = Tensor.from_numpy(input_np)
            other_torch = torch.from_numpy(other_np)
            other_ms = Tensor.from_numpy(other_np)
        else:
            input_np = np.random.randint(0, 10, size=(3, 3), dtype=dtype)
            other_np = np.random.randint(0, 10, size=(3, 3), dtype=dtype)
            input_torch = torch.from_numpy(input_np)
            input_ms = Tensor.from_numpy(input_np)
            other_torch = torch.from_numpy(other_np)
            other_ms = Tensor.from_numpy(other_np)

        torch_support = True
        ms_support = True
        #test
        try:
            where_torch = torch.where(condition_torch, input_torch, other_torch)
            print(f"<torch_result>: {where_torch}")
        except:
            torch_support = False

        try:
            where_ms = mint.where(condition_ms, input_ms, other_ms)
            print(f"<torch_result>: {where_torch}")
        except:
            ms_support = False

        #result
        if torch_support == ms_support == True:
            print(f"[dtype] BOTH SUPPORT (from test_where_dtype): "
                  f"torch.where support {where_torch.dtype} "
                  f"& mindspore.mint.where support {where_ms.dtype}.")
        elif torch_support == ms_support == False:
            print(f"[dtype] BOTH NOT SUPPORT (from test_where_dtype): "
                  f"torch.where does not support {input_torch.dtype} "
                  f"& mindspore.mint.where does not support {input_ms.dtype}.")
        elif torch_support == True and ms_support == False:
            print(f"[dtype] ONLY TORCH (from test_where_dtype): "
                  f"torch.where support {where_torch.dtype} "
                  f"but mindspore.mint.where NOT.")
        else:
            print(f"[dtype] ONLY MS (from test_where_dtype): "
                  f"mindspore.mint.where support {where_ms.dtype} "
                  f"but torch.where NOT.")

        print('='*50)

#b) 测试固定dtype，random输入值，对比两个框架输出是否相等（误差范围为小于1e-3）
def test_where_output():
    print("--->This function: test mindspore.mint.where [output]<---")
    #共同支持以下类型
    np_dtypes = [np.int8, np.int16, np.int32, np.int64,
                 np.float16, np.float32, np.float64,
                 np.uint8, np.complex64, np.complex128, np.bool_,
                'bfloat16']
    
    for dtype in np_dtypes:
        condition_np = np.array(True)
        condition_torch = torch.from_numpy(condition_np)
        condition_ms = Tensor.from_numpy(condition_np)
        if dtype == np.bool_:
            input_np = np.random.choice(np.array([True, False]), size=(3, 3))
            other_np = np.random.choice(np.array([True, False]), size=(3, 3))
            input_torch = torch.from_numpy(input_np)
            input_ms = Tensor.from_numpy(input_np)
            other_torch = torch.from_numpy(other_np)
            other_ms = Tensor.from_numpy(other_np)
        elif dtype == 'bfloat16':
            input_np = np.random.rand(3, 3).astype(np.float32)
            other_np = np.random.rand(3, 3).astype(np.float32)
            input_torch = torch.from_numpy(input_np).type(torch.bfloat16)
            input_ms = Tensor.from_numpy(input_np).astype(ms.bfloat16)
            other_torch = torch.from_numpy(other_np).type(torch.bfloat16)
            other_ms = Tensor.from_numpy(other_np).astype(ms.bfloat16)
        elif (dtype == np.float16 or dtype == np.float32 or dtype == np.float64 or 
        dtype == np.complex64 or dtype == np.complex128):
            input_np = np.random.rand(3, 3).astype(dtype)
            other_np = np.random.rand(3, 3).astype(dtype)
            input_torch = torch.from_numpy(input_np)
            input_ms = Tensor.from_numpy(input_np)
            other_torch = torch.from_numpy(other_np)
            other_ms = Tensor.from_numpy(other_np)
        else:
            input_np = np.random.randint(0, 10, size=(3, 3), dtype=dtype)
            other_np = np.random.randint(0, 10, size=(3, 3), dtype=dtype)
            input_torch = torch.from_numpy(input_np)
            input_ms = Tensor.from_numpy(input_np)
            other_torch = torch.from_numpy(other_np)
            other_ms = Tensor.from_numpy(other_np)

        where_torch = torch.where(condition_torch, input_torch, other_torch)
        where_ms = mint.where(condition_ms, input_ms, other_ms)
        #numpy中没有bfloat16类型
        if dtype == 'bfloat16':
            print("current type: bfloat16\n"
                 f"<result_torch>: {where_torch}\n"
                 f"<result_torch>: {where_ms}")
        else:
            if np.allclose(where_torch.numpy(), where_ms.asnumpy(), atol=1e-3):
                print("[output] WITHIN TOLERANCE (from test_where_output): "
                      "output discrepancy between torch.where "
                      "mindspore.mint.where is less than 1e-3.")
            else:
                print("[output] BEYOND TOLERANCE (from test_where_output): "
                      "output discrepancy is beyond tolerance.")
        
        print('='*50)

#c) 测试固定shape，固定输入值，不同输入参数（string\bool等类型），两个框架的支持度
def test_where_paramType():
    print("--->This function: test mindspore.mint.where [paramType]<---")

    input_np = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.float32)
    input_torch = torch.from_numpy(input_np)
    input_ms = Tensor.from_numpy(input_np)

    other_np = np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5], [7.5, 8.5, 9.5]]).astype(np.float32)
    other_torch = torch.from_numpy(other_np)
    other_ms = Tensor.from_numpy(other_np)

    params = (("abc", 'string'), 
              (True, 'bool'), 
              (0, 'int'), 
              (1.0, 'float'), 
              ([1, 2], 'list'),
              ((0, 1), 'tuple'),
              (np.random.choice(np.array([True, False]), size=3), 'Tensor')
              )

    for (param, param_type) in params:
        #test
        torch_support = True
        ms_support = True
        if param_type == 'Tensor':
            param_torch = torch.from_numpy(param)
            param_ms = Tensor.from_numpy(param)
            try:
                where_torch = torch.where(param_torch, input_torch, other_torch)
            except:
                torch_support = False

            try:
                where_ms = mint.where(param_ms, input_ms, other_ms)
            except:
                ms_support = False
        else:
            try:
                where_torch = torch.where(param, input_torch, other_torch)
            except:
                torch_support = False

            try:
                where_ms = mint.where(param, input_ms, other_ms)
            except:
                ms_support = False

        #result
        if torch_support == ms_support == True:
            print(f"[paramType: {param_type}] BOTH SUPPORT(from test_where_paramType).")
        elif torch_support == ms_support == False:
            print(f"[paramType: {param_type}] BOTH NOT SUPPORT(from test_where_paramType).")
        elif torch_support == True and ms_support == False:
            print(f"[paramType: {param_type}] ONLY TORCH(from test_where_paramType): "
                  "torch.where support but mindspore.mint.where NOT.")
        else:
            print(f"[paramType: {param_type}] ONLY MS(from test_where_paramType): "
                  "mindspore.mint.where support but torch.where NOT.")

        print('='*50)

#d) 测试随机混乱输入，报错信息的准确性
def test_where_errorMessage():
    print("--->This function: test mindspore.mint.where [errorMessage]<---")

    #param condition TypeError
    try:
        condition1 = [True, True]
        input_ms1 = Tensor([1, 2])
        other_ms1 = Tensor([3, 4])
        where_ms1 = mint.where(condition1, input_ms1, other_ms1)
        print(where_ms1)
    except Exception as e:
        print(f"[errorMessage] ACCURACY TEST (from test_where_errorMessage): "
              "Test Target: param condition type error-TypeError.\n"
              f"Catch Exception:\n{type(e).__name__}: {e}.")
    print('='*50)

    #input other all const TypeError
    try:
        condition2 = Tensor(True)
        input_ms2 = 1
        other_ms2 = 2
        where_ms2 = mint.where(condition2, input_ms2, other_ms2)
        print(where_ms2)
    except Exception as e:
        print(f"[errorMessage] ACCURACY TEST (from test_where_errorMessage): "
              "Test Target: input&other tensor type error-TypeError.\n"
              f"Catch Exception:\n{type(e).__name__}: {e}.")
    print('='*50)

    #can not broadcast ValueError
    try:
        condition3 = Tensor([[True, False],[False, True]])
        input_ms3 = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        other_ms3 = Tensor([[[1, 2, 3, 0], [4, 5, 6, 0], [7, 8, 9, 0]],
                           [[1, 2, 3, 0], [4, 5, 6, 0], [7, 8, 9, 0]]])
        where_ms3 = mint.where(condition3, input_ms3, other_ms3)
        print(where_ms3)
    except Exception as e:
        print(f"[errorMessage] ACCURACY TEST (from test_where_errorMessage): "
              "Test Target: can not broadcast error-ValueError.\n"
              f"Catch Exception:\n{type(e).__name__}: {e}.")
    print('='*50)

#2. 测试使用接口构造函数/神经网络的准确性
#a) Github搜索带有该接口的代码片段/神经网络
#b) 使用Pytorch和MindSpore，固定输入和权重，测试正向推理结果（误差范围小于1e-3，若报错则记录报错信息）
#c) 测试该神经网络/函数反向，如果是神经网络，则测试Parameter的梯度，如果是函数，则测试函数输入的梯度
def test_where_value_and_grad():
    def ternarize_torch(tensor):
        delta = get_delta_torch(tensor)
        alpha = get_alpha_torch(tensor, delta)
        pos = torch.where(tensor > delta, 1, tensor)
        neg = torch.where(pos<-delta, -1, pos)
        ternary = torch.where((neg >= -delta) & (neg <= delta), 0, neg)
        return ternary * alpha

    def get_alpha_torch(tensor, delta):
        ndim = len(tensor.shape)
        view_dims = (-1,) + (ndim - 1)*(1,)
        i_delta = (torch.abs(tensor) > delta)
        i_delta_count = i_delta.view(i_delta.shape[0], -1).sum(1)
        tensor_thresh = torch.where((i_delta), tensor, 0)
        alpha = (1 / i_delta_count)*(torch.abs(tensor_thresh.view(tensor.shape[0], -1)).sum(1))
        alpha = alpha.view(view_dims)
        return alpha

    def get_delta_torch(tensor):
        ndim = len(tensor.shape)
        view_dims = (-1,) + (ndim - 1) * (1,)
        n = tensor[0].nelement()
        norm = tensor.norm(1, ndim - 1).view(tensor.shape[0], -1)
        norm_sum = norm.sum(1)
        delta = (0.75 / n) * norm_sum
        return delta.view(view_dims)
    
    def ternarize_ms(tensor):
        delta = get_delta_ms(tensor)
        alpha = get_alpha_ms(tensor, delta)
        pos = mint.where(tensor > delta, 1, tensor)
        neg = mint.where(pos<-delta, -1, pos)
        condition = ops.logical_and((neg >= -delta), (neg <= delta))
        ternary = mint.where(condition, 0, neg)
        return ternary * alpha

    def get_alpha_ms(tensor, delta):
        ndim = len(tensor.shape)
        view_dims = (-1,) + (ndim - 1)*(1,)
        i_delta = (ops.abs(tensor) > delta)
        i_delta_count = i_delta.view(i_delta.shape[0], -1).sum(1)
        tensor_thresh = mint.where((i_delta), tensor, 0)
        alpha = (1 / i_delta_count)*(ops.abs(tensor_thresh.view(tensor.shape[0], -1)).sum(1))
        alpha = alpha.view(view_dims)
        return alpha

    def get_delta_ms(tensor):
        ndim = len(tensor.shape)
        view_dims = (-1,) + (ndim - 1) * (1,)
        n = tensor[0].nelement()
        norm = tensor.norm(1, ndim - 1).view(tensor.shape[0], -1)
        norm_sum = norm.sum(1)
        delta = (0.75 / n) * norm_sum
        return delta.view(view_dims)
    
    x_np = np.random.randn(3, 3).astype(np.float32)
    x_torch = torch.from_numpy(x_np)
    x_torch.requires_grad = True
    x_ms = Parameter(Tensor.from_numpy(x_np))

    y_torch = ternarize_torch(x_torch)
    y_ms = ternarize_ms(x_ms)

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
    grad_ms = ms.grad(ternarize_ms, grad_position=0)(x_ms)
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
    test_where_dtype()
    test_where_output()
    test_where_paramType()
    test_where_errorMessage()
    test_where_value_and_grad()