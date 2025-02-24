import mindspore as ms
import numpy as np
import random
import torch
import pytest

def generate_shape():#随机生成5维以内的shape
  """随机生成1到5维的shape，每个维度大小的范围也可以随机变化"""
  n = random.randint(1, 5)  # 随机选择维度数
  shape = []
  for _ in range(n):
      # 随机选择一个大小范围
      min_size, max_size = random.choice([(1, 5), (5, 10), (10, 20)])
      size = random.randint(min_size, max_size)
      shape.append(size)
  return tuple(shape)

def generate_data(shape,dtype):#根据shape和类型生成
  if dtype in [np.int8, np.int16, np.int32, np.int64]:
      input_data = np.random.randint(-10, 10, size=shape).astype(dtype)
  elif dtype in [np.uint8,np.uint16,np.uint32,np.uint64]:
      input_data = np.random.randint(0, 10, size=shape).astype(dtype)
  elif dtype == np.bool_:
      input_data = np.random.choice([True, False], size=shape).astype(dtype)
  else:
      input_data = np.random.uniform(low=-0.999, high=10, size=shape).astype(dtype)
  return input_data

def check(arr1, arr2, threshold=1e-3):
    """
    检查两个numpy数组在对应位置上的 inf、-inf 或 nan 是否相同，
    并且在其他位置上的误差是否小于给定的阈值。
    """
    if arr1 is None or arr2 is None:
       if arr1 is None and arr2 is None:
          return True,None
       return False,None
    # 检查形状是否相同
    if arr1.shape != arr2.shape:
        raise ValueError("Arrays must have the same shape")
    
    # 检查 nan 是否对应相同
    is_nan_arr1 = np.isnan(arr1)
    is_nan_arr2 = np.isnan(arr2)
    if not np.array_equal(is_nan_arr1, is_nan_arr2):
        return False,None
    
    # 检查 inf 和 -inf 是否对应相同
    is_pos_inf_arr1 = np.isposinf(arr1)
    is_pos_inf_arr2 = np.isposinf(arr2)
    if not np.array_equal(is_pos_inf_arr1, is_pos_inf_arr2):
        return False,None
    
    is_neg_inf_arr1 = np.isneginf(arr1)
    is_neg_inf_arr2 = np.isneginf(arr2)
    if not np.array_equal(is_neg_inf_arr1, is_neg_inf_arr2):
        return False,None
    
    # 找到非 inf 和 nan 的位置
    valid_mask = ~(is_nan_arr1 | is_pos_inf_arr1 | is_neg_inf_arr1)
    
    # 在有效位置上计算误差
    diff = np.abs(arr1[valid_mask] - arr2[valid_mask])
    
    # 检查误差是否小于阈值
    if np.any(diff >= threshold):
        return False,diff
    
    return True,None

def check2(arr1, arr2, threshold=1e-3):
    """
    检查两个numpy数组在对应位置上的 inf、-inf 或 nan 是否相同，
    并且在其他位置上的误差是否小于给定的阈值。
    """
    res,_=check(arr1,arr2,threshold)
    
    return res

@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_any_different_dtypes(mode):
    """测试random输入不同dtype，对比两个框架的支持度"""
    print(f"---------------------------1.test_any_different_dtypes(mode={mode})-------------------------")
    ms.set_context(mode=mode)
    ms_dtypes = [ms.int8, ms.int16, ms.int32, ms.int64, ms.uint8,ms.uint16, ms.uint32, ms.uint64, ms.float16, ms.float32, ms.float64, ms.bfloat16, ms.bool_]
    torch_dtypes = [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.uint16, torch.uint32, torch.uint64, torch.float16, torch.float32, torch.float64, torch.bfloat16, torch.bool]
    input_data = [[1,0],[0,0],[1,1]]

    for i in range(len(ms_dtypes)):
      ms_type=ms_dtypes[i]
      torch_type=torch_dtypes[i]

      ms_input=ms.Tensor(input_data,ms_type)
      torch_input=torch.tensor(input_data,dtype=torch_type)
      ms_res=None
      torch_res=None
      print(f"当前数据类型为{ms_type}:")
      try:
        ms_res=ms.mint.special.log_softmax(ms_input)
        print(ms_res)
      except Exception as e:
        print(f"Mindspore不支持{ms_type}类型")
        print(e)
            
      try:
        torch_res=torch.nn.functional.log_softmax(torch_input)
        print(torch_res)
      except Exception as e:
        print(f"Torch不支持{torch_type}类型")
        print(e)

@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_any_random_input_fixed_dtype(mode):
    """测试固定数据类型下的随机输入值，对比输出误差"""
    print(f"---------------------------2.test_any_random_input_fixed_dtype(mode={mode})-------------------------")
    ms.set_context(mode=mode)

    #100次随机生成数据，看是否有输出误差
    types=[np.float16, np.float32, np.float64]
    ms_dtypes = [ms.float16, ms.float32, ms.float64]
    torch_dtypes = [torch.float16, torch.float32, torch.float64]
    np.set_printoptions(formatter={'float': '{: 0.5f}'.format})
    flag=True
    for i,type in enumerate(types):
        print(f"当前数据类型为{type}:")
        ms_type=ms_dtypes[i]
        torch_type=torch_dtypes[i]
        for _ in range(100):
#           shape=(5,1)
          shape=generate_shape()
          inputs=generate_data(shape,type)
          dim=np.random.randint(0,len(shape))
          ms_inputs = ms.Tensor(inputs,ms_type)
          torch_inputs = torch.tensor(inputs,dtype=torch_type)
          ms_res = ms.mint.special.log_softmax(ms_inputs,dim=dim)
          torch_res=torch.nn.functional.log_softmax(torch_inputs,dim=dim)
#           print(f"torch_res:\n{torch_res}\nms_res:\n{ms_res}")
          #转为numpy数组进行比较
          ms_res1=ms_res.asnumpy()
          torch_res1=np.asarray(torch_res)
          f,diff= check(ms_res1,torch_res1)
          if not f:
            flag=False
#             print("inputs:",inputs)
#             print("torch_res:",torch_res)
#             print("ms_res:",ms_res)
#             print("diff:",diff)
            if diff is not None:
                print(f"当输入类型为{type}时，输出误差为{min(diff)}~{max(diff)}")
            else:
                print(f"当输入类型为{type}时，输出误差为{diff}")
    assert flag

@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_any_different_para(mode):
    """测试固定shape，固定输入值，不同输入参数（string\bool等类型），两个框架的支持度"""
    inputs=[[1.1,2.2],[3.3,4.4]]
    torch_input=torch.tensor(inputs)
    ms_input=ms.Tensor(inputs)

    dims=['1',True,{'a':1},[1,2],float("nan"),float('inf'),float('-inf'),None,-1,-2,-3,2,3]
    for dim in dims:
       torch_res=None
       ms_res=None
       print(f"当dim的输入为{dim}，即输入类型为{type(dim)}时：")
       try:
          torch_res=torch.nn.functional.log_softmax(torch_input,dim=dim)
          print(torch_res)
       except Exception as e:
          print(f"Pytorch不支持{type(dim)}类型")
          print(e)
       try:
          ms_res=ms.mint.special.log_softmax(ms_input,dim=dim)
          print(ms_res)
       except Exception as e:
          print(f"Mindspore不支持{type(dim)}类型")
          ms_res=None
          print(e)
       if torch_res is None or ms_res is None:
          assert check2(torch_res,ms_res)
       else:
          assert check2(torch_res.numpy(),ms_res.asnumpy())
    assert True

@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_any_wrong_input(mode):
    """测试随机混乱输入，报错信息的准确性"""
    print(f"---------------------------4.test_any_wrong_input(mode={mode})-------------------------")
    ms.set_context(mode=mode)
    #4.1特殊数据nan或inf
    torch_input=torch.tensor([[float("nan"),1],[float("inf"),1],[float("-inf"),1]]) 
    ms_input=ms.Tensor([[float("nan"),1],[float("inf"),1],[float("-inf"),1]]) 
    print(f"\n特殊数据nan或inf:{torch_input}")
    torch_res=None
    ms_res=None
    try:
      torch_res=torch.nn.functional.log_softmax(torch_input,dim=0)
      ms_res=ms.mint.special.log_softmax(ms_input,dim=0)
      print(torch_res)
      print(ms_res)
    except Exception as e:
      print(e)
    if torch_res is None or ms_res is None:
        assert check2(torch_res,ms_res)
    else:
        assert check2(torch_res.numpy(),ms_res.asnumpy())

    #4.2非法数据结构（非torch）
    inputs=['123',[0.1,-0.4],{'a':1,'b':2},0.1,True,np.array([1,-3]),None,{'a':1,'b':2},np.array([1,-3]),None]
    print("\n输入非法数据结构：")
    for inp in inputs:
      torch_res=None
      ms_res=None
      print(f"当输入是{inp}时，")
      print("PyTorch：",end="")
      try:
        torch_res=torch.nn.functional.log_softmax(inp)
        print(torch_res)  
      except Exception as e:
        print(e)
      print("MindSpore：",end="")
      try:
        ms_res=ms.mint.special.log_softmax(inp)
        print(ms_res)  
      except Exception as e:
        print(e)
      if torch_res is None or ms_res is None:
        assert check2(torch_res,ms_res)
      else:
        assert check2(torch_res.numpy(),ms_res.asnumpy())

@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_any_forward_back(mode):
    """使用Pytorch和MindSpore, 固定输入和权重, 测试正向推理结果和反向梯度"""
    print(f"---------------------------5.test_any_forward_back(mode={mode})-------------------------")
    ms.set_context(mode=mode)
    ms_dtypes = [ms.float16, ms.float32]
    torch_dtypes = [torch.float16, torch.float32]
    types=[np.float16, np.float32]
    np.set_printoptions(formatter={'float': '{: 0.6f}'.format})
    flag=True
    # 随机100次，看是否会出现误差
    for i,ms_type in enumerate(ms_dtypes):
      torch_type=torch_dtypes[i]
      print(f"当前数据类型为{ms_type}:")
      for u in range(100):
        shape=generate_shape()
#         shape=(3,1)
        input_data = generate_data(shape,types[i])
        torch_inp = torch.tensor(input_data, dtype=torch_type, requires_grad=True)
        ms_inp=ms.Tensor(input_data,ms_type)
        dim=np.random.randint(0,len(shape))
        
        def forward_pt(x,dim):
            return torch.nn.functional.log_softmax(x,dim=dim)

        def forward_ms(x,dim):
            return ms.mint.special.log_softmax(x,dim=dim)
        
        #5.1测试正向推理结果是否小于1e-3
        torch_res=forward_pt(torch_inp,dim)
        ms_res=forward_ms(ms_inp,dim)
        f,diff=check(np.asarray(torch_res.detach()),ms_res.asnumpy())
        if not f:
            flag=False
#             print("input_data:",input_data)
#             print("torch_res:",torch_res)
#             print("ms_res:",ms_res)
#             print("diff:",diff)
            if diff is not None:
                print(f"当输入类型为{ms_type}时，前向传播输出误差为{min(diff)}~{max(diff)}")
            else:
                print(f"当输入类型为{ms_type}时，前向传播输出误差为{diff}")
#         assert check2(np.asarray(torch_res.detach()),ms_res.asnumpy())

        #5.2测试反向传播梯度
        torch_res.backward(torch.ones_like(torch_res))
        torch_inp_grad=torch_inp.grad
        grad_fn = ms.value_and_grad(forward_ms)
        _, gradient_ms = grad_fn(ms_inp,dim)
        ms_inp_grad=gradient_ms
#         print(torch_inp_grad,ms_inp_grad)
        f,diff=check(np.asarray(torch_inp_grad.detach()),ms_inp_grad.asnumpy())
        if not f:
            flag=False
#             print("input_data:",input_data)
#             print("torch_res:",torch_res)
#             print("ms_res:",ms_res)
#             print("ms_inp_grad:",ms_inp_grad)
#             print("torch_inp_grad:",torch_inp_grad)
#             print("diff:",diff)
            if diff is not None:
                print(f"当输入类型为{ms_type}时，梯度输出误差为{min(diff)}~{max(diff)}")
            else:
                print(f"当输入类型为{ms_type}时，梯度输出误差为{diff}")
    assert flag
      
if __name__=="__main__":
#     test_any_different_dtypes(ms.PYNATIVE_MODE)
#     test_any_random_input_fixed_dtype(ms.PYNATIVE_MODE)
#     test_any_different_para(ms.PYNATIVE_MODE)
#     test_any_wrong_input(ms.PYNATIVE_MODE)
    test_any_forward_back(ms.PYNATIVE_MODE)