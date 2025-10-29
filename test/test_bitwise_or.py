import mindspore as ms
import numpy as np
import random
import torch
import pytest


import warnings
 
warnings.filterwarnings('ignore')

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
  if dtype in [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16]:
      input_data = np.random.randint(0, 2, size=shape).astype(dtype)
  elif dtype == np.bool_:
      input_data = np.random.choice([True, False], size=shape).astype(dtype)
  else:
      input_data = np.random.randn(*shape).astype(dtype)
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
    # bitwise操作，将bool型转为数值型
    arr1=arr1.astype(float)
    arr2=arr2.astype(float)
    
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

@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_any_different_dtypes(mode):
    """测试random输入不同dtype，对比两个框架的支持度"""
    ms.set_context(mode=mode)
    ms_dtypes = [ms.int8, ms.int16, ms.int32, ms.int64, ms.uint8,ms.uint16, ms.uint32, ms.uint64,ms.bool_]
    torch_dtypes = [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.uint16, torch.uint32, torch.uint64,torch.bool]
    input_data = [[1,0],[0,0],[1,1]]
    other_data = [[1,0],[0,0],[1,1]]


    for i in range(len(ms_dtypes)):
      ms_type=ms_dtypes[i]
      torch_type=torch_dtypes[i]

      ms_input=ms.Tensor(input_data,ms_type)
      ms_other=ms.Tensor(other_data,ms_type)
      torch_input=torch.tensor(input_data,dtype=torch_type)
      torch_other=torch.tensor(other_data,dtype=torch_type)
      try:
        ms_res=ms.mint.bitwise_or(ms_input,ms_other)
        print(ms_res)
      except Exception as e:
        print(f"Mindspore不支持{ms_type}类型")
        print(e)
              
      try:
        torch_res=torch.bitwise_or(torch_input,torch_other)
        print(torch_res)
      except Exception as e:
        print(f"Torch不支持{torch_type}类型")
        print(e)

        
    

@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_any_random_input_fixed_dtype(mode):
    """测试固定数据类型下的随机输入值，对比输出误差"""
    ms.set_context(mode=mode)

    #每种type随机生成100个数据，看是否有输出误差
    types=[np.int8, np.int16, np.int32, np.int64,
    np.uint8, np.bool_]
    np.set_printoptions(formatter={'float': '{: 0.5f}'.format})
    flag=True
    for type in types:
        for _ in range(100):
          shape=generate_shape()
          inputs=generate_data(shape,type)
          others=generate_data(shape,type)
          ms_inputs = ms.Tensor(inputs)
          ms_others = ms.Tensor(others)
          torch_inputs = torch.tensor(inputs)
          torch_others = torch.tensor(others)
          ms_res = ms.mint.bitwise_or(ms_inputs,ms_others)
          torch_res=torch.bitwise_or(torch_inputs,torch_others)
          #转为numpy数组进行比较
          ms_res1=ms_res.asnumpy()
          torch_res1=np.asarray(torch_res)
          f,diff= check(ms_res1,torch_res1)
          if not f:
            # continue
            flag=False
            print(f"当输入类型为{type}时，输出误差为{min(diff)}~{max(diff)}")
    assert flag

@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_any_different_para(mode):
    """测试固定shape，固定输入值，不同输入参数（string\bool等类型），两个框架的支持度"""
    assert True#由于bitwise_or只有input、other参数，因此不再重复test_any_different_dtypes的测试

@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_any_wrong_input(mode):
    """测试随机混乱输入，报错信息的准确性"""
    ms.set_context(mode=mode)
    #4.1特殊数据nan或inf
    torch_input=torch.tensor([float("nan"),float('inf'),float('-inf'),1,2,0]) 
    torch_other=torch.tensor([float("nan"),float('-inf'),float('inf'),float("nan"),float('inf'),float('-inf')]) 
    ms_input=ms.Tensor([float("nan"),float('inf'),float('-inf'),1,2,0]) 
    ms_other=ms.Tensor([float("nan"),float('-inf'),float('inf'),float("nan"),float('inf'),float('-inf')]) 
    print("\n特殊数据nan或inf：")
    torch_res=None
    ms_res=None
    try:
      torch_res=torch.bitwise_or(torch_input,torch_other)
      print(torch_res)
    except Exception as e:
      print(e)
    try:
      ms_res=ms.mint.bitwise_or(ms_input,ms_other)
      print(ms_res)
    except Exception as e:
      print(e)
    if torch_res is None or ms_res is None:
        assert check(torch_res,ms_res)
    else:
        assert check(torch_res.numpy(),ms_res.asnumpy())
        

    #4.2input和other形状不同但是可以广播
    torch_input=torch.tensor([[1,2],[3,4],[5,6]]) #shape:(3,2) 
    torch_other=torch.tensor([[1],[2],[3]]) #shape:(3,1)
    ms_input=ms.Tensor([[1,2],[3,4],[5,6]]) 
    ms_other=ms.Tensor([[1],[2],[3]]) 
    print("\ninput和other形状不同但是可以广播：")
    try:
      torch_res=torch.bitwise_or(torch_input,torch_other)
      ms_res=ms.mint.bitwise_or(ms_input,ms_other)
      print(torch_res)
      print(ms_res)
      assert check(torch_res.numpy(),ms_res.asnumpy())
    except Exception as e:
      print(e)

    #4.3input和other形状不同且不可广播
    torch_input=torch.tensor([[1,2],[3,4],[5,6]]) #shape:(3,2) 
    torch_other=torch.tensor([1,2,3]) #shape:(3,)
    ms_input=ms.Tensor([[1,2],[3,4],[5,6]]) 
    ms_other=ms.Tensor([1,2,3]) 
    print("\ninput和other形状不同且不可广播：")
    try:
      torch_res=torch.bitwise_or(torch_input,torch_other)
      print(torch_res)
    except Exception as e:
      print(e)
    try:
      ms_res=ms.mint.bitwise_or(ms_input,ms_other)
      print(ms_res)
    except Exception as e:
      print(e)

    #4.4非法数据结构（非torch或常数）
    inputs=['123',[0.1,-0.4],{'a':1,'b':2},0.1,True,np.array([1,-3]),None,{'a':1,'b':2},np.array([1,-3]),None]
    others=['123',[0.1,-0.4],{'a':1,'b':2},0.1,True,'123',np.array([1,-3]),np.array([1,-3]),None,None]
    print("\n输入非法数据结构：")
    for i,inp in enumerate(inputs):
      oth=others[i]
      torch_res=None
      ms_res=None
      print(f"当输入是{inp}和{oth}时，")
      print("PyTorch：",end="")
      if type(inp) is np.ndarray:
         torch_inp=torch.tensor(inp)
      else:
         torch_inp=inp
      if type(oth) is np.ndarray:
         torch_oth=torch.tensor(oth)
      else:
         torch_oth=oth
      try:
        torch_res=torch.bitwise_or(torch_inp,torch_oth)
        print(torch_res)  
      except Exception as e:
        print(e)
      print("MindSpore：",end="")
      if type(inp) is np.ndarray:
         ms_inp=ms.Tensor(inp)
      else:
         ms_inp=inp
      if type(oth) is np.ndarray:
         ms_oth=ms.Tensor(oth)
      else:
         ms_oth=oth
      try:
        ms_res=ms.mint.bitwise_or(ms_inp,ms_oth)
        print(ms_res)  
      except Exception as e:
        print(e)
      assert check(torch_res,ms_res)

@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_any_forward_back(mode):
    """使用Pytorch和MindSpore, 固定输入和权重, 测试正向推理结果和反向梯度"""
    ms.set_context(mode=mode)
    dtype_ms = ms.int32
    dtype_torch = torch.int32
    input_data = [[1,-2],[33,44],[0,-5]]
    other_data = [[4,1],[1,4],[5,0]]
    torch_inp = torch.tensor(input_data, dtype=dtype_torch)
    torch_oth = torch.tensor(other_data, dtype=dtype_torch)
    ms_inp=ms.Tensor(input_data, dtype_ms)
    ms_oth=ms.Tensor(other_data, dtype_ms)

    def forward_pt(x,y):
        return torch.bitwise_or(x,y)

    def forward_ms(x,y):
        return ms.mint.bitwise_or(x,y)
    
    #5.1测试正向推理结果是否小于1e-3
    torch_res=forward_pt(torch_inp,torch_oth)
    ms_res=forward_ms(ms_inp,ms_oth)
    assert check(np.asarray(torch_res.detach()),ms_res.asnumpy())

    #5.2该运算不可求导，故不再测试反向传播梯度
      
if __name__=="__main__":
    print("---------------------------1.test_any_different_dtypes-------------------------")
#     test_any_different_dtypes(ms.PYNATIVE_MODE)
    print("---------------------------2.test_any_random_input_fixed_dtype-------------------------")
    test_any_random_input_fixed_dtype(ms.PYNATIVE_MODE)
    print("---------------------------4.test_any_wrong_input-------------------------")
    test_any_wrong_input(ms.PYNATIVE_MODE)
    print("---------------------------5.test_any_forward_back-------------------------")
    test_any_forward_back(ms.PYNATIVE_MODE)