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
    # 检查形状是否相同
    if arr1.shape != arr2.shape:
        raise ValueError("Arrays must have the same shape")
    
    # 检查 nan 是否对应相同
    is_nan_arr1 = np.isnan(arr1)
    is_nan_arr2 = np.isnan(arr2)
    if not np.array_equal(is_nan_arr1, is_nan_arr2):
        return False
    
    # 检查 inf 和 -inf 是否对应相同
    is_pos_inf_arr1 = np.isposinf(arr1)
    is_pos_inf_arr2 = np.isposinf(arr2)
    if not np.array_equal(is_pos_inf_arr1, is_pos_inf_arr2):
        return False
    
    is_neg_inf_arr1 = np.isneginf(arr1)
    is_neg_inf_arr2 = np.isneginf(arr2)
    if not np.array_equal(is_neg_inf_arr1, is_neg_inf_arr2):
        return False
    
    # 找到非 inf 和 nan 的位置
    valid_mask = ~(is_nan_arr1 | is_pos_inf_arr1 | is_neg_inf_arr1)
    
    # 在有效位置上计算误差
    diff = np.abs(arr1[valid_mask] - arr2[valid_mask])
    
    # 检查误差是否小于阈值
    if np.any(diff >= threshold):
        return False
    
    return True

@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_any_different_dtypes(mode):
    """测试random输入不同dtype，对比两个框架的支持度"""
    ms.set_context(mode=mode)
    ms_dtypes = [ms.int8, ms.int16, ms.int32, ms.int64, ms.uint8,ms.uint16, ms.uint32, ms.uint64, ms.float16, ms.float32, ms.float64, ms.bfloat16, ms.bool_]
    torch_dtypes = [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.uint16, torch.uint32, torch.uint64, torch.float16, torch.float32, torch.float64, torch.bfloat16, torch.bool]
    input_data = [[1,0],[0,0],[1,1]]

    for i in range(len(ms_dtypes)):
      ms_type=ms_dtypes[i]
      torch_type=torch_dtypes[i]

      ms_input=ms.Tensor(input_data,ms_type)
      torch_input=torch.tensor(input_data,dtype=torch_type)

      try:
        ms_res=ms.mint.atanh(ms_input)
      except Exception as e:
        print(f"Mindspore不支持{ms_type}类型")
            
      try:
        torch_res=torch.atanh(torch_input)
      except Exception as e:
        print(f"Torch不支持{torch_type}类型")

@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_any_random_input_fixed_dtype(mode):
    """测试固定数据类型下的随机输入值，对比输出误差"""
    ms.set_context(mode=mode)

    #100次随机生成数据，看是否有输出误差
    types=[np.float16, np.float32, np.float64]
    for _ in range(100):
      type=random.choice(types)
      shape=generate_shape()
      inputs=generate_data(shape,type)
      ms_inputs = ms.Tensor(inputs)
      torch_inputs = torch.tensor(inputs)
      ms_res = ms.mint.atanh(ms_inputs)
      torch_res=torch.atanh(torch_inputs)
      #转为numpy数组进行比较
      ms_res=ms_res.asnumpy()
      torch_res=np.asarray(torch_res)
      assert check(ms_res,torch_res)

@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_any_different_para(mode):
    """测试固定shape，固定输入值，不同输入参数（string\bool等类型），两个框架的支持度"""
    assert True#由于atanh只有一个输入参数，因此不再重复test_any_different_dtypes的测试

@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_any_wrong_input(mode):
    """测试随机混乱输入，报错信息的准确性"""
    ms.set_context(mode=mode)
    #4.1正常情况
    torch_input=torch.tensor([0.3,-0.66,0,0.2])  
    ms_input=ms.Tensor([0.3,-0.66,0,0.2])  
    print("正常输入：")
    try:
      print(torch.atanh(torch_input))  
      print(ms.mint.atanh(ms_input))  
    except Exception as e:
      print(e)

    #4.2特殊数据（函数要求输入在(-1,1)之间）
    torch_input=torch.tensor([float("nan"),float('inf'),float('-inf'),-1,1,3.3,-5.5])  # 字符串类型
    ms_input=ms.Tensor([float("nan"),float('inf'),float('-inf'),-1,1,3.3,-5.5])  # 字符串类型
    print("\n输入数据在区间外或nan、inf：")
    try:
      print(torch.atanh(torch_input))  
      print(ms.mint.atanh(ms_input))  
    except Exception as e:
      print(e)

    #4.3非法数据结构（不是torch）；可以看到MindSpore的报错信息不够详细，并未指出具体错误的输入类型
    inputs=['123',[0.1,-0.4],{'a':1,'b':2},0.1,True,np.array([0.1,-0.3])]
    print("\n输入非法数据结构：")
    for inp in inputs:
      print(f"当输入是{inp}时，")
      print("PyTorch：",end="")
      try:
        print(torch.atanh(inp))  
      except Exception as e:
        print(e)
      print("MindSpore：",end="")
      try:
        print(ms.mint.atanh(inp))  
      except Exception as e:
        print(e)

    #4.4空输入
    inp=None
    print("\n空输入：\nPytorch:",end="")
    try:
      print(torch.atanh(inp))  
    except Exception as e:
      print(e)
    print("MindSpore：",end="")
    try:
      print(ms.mint.atanh(inp))  
    except Exception as e:
      print(e)

@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_any_forward_back(mode):
    """使用Pytorch和MindSpore, 固定输入和权重, 测试正向推理结果和反向梯度"""
    ms.set_context(mode=mode)
    dtype_ms = ms.float32
    dtype_torch = torch.float32
    input_data = [[0.1,-0.2],[0.33,0.44],[0,-0.5]]
    torch_tensor = torch.tensor(input_data, dtype=dtype_torch, requires_grad=True)
    ms_tensor=ms.Tensor(input_data, dtype_ms)

    def forward_pt(x):
        return torch.atanh(x)

    def forward_ms(x):
        return ms.mint.atanh(x)
    
    #5.1测试正向推理结果是否小于1e-3
    torch_res=forward_pt(torch_tensor)
    ms_res=forward_ms(ms_tensor)
    assert check(np.asarray(torch_res.detach()),ms_res.asnumpy())

    #5.2测试反向传播梯度
    torch_res.backward(torch.ones_like(torch_res))
    torch_grad=torch_tensor.grad
    grad_fn = ms.value_and_grad(forward_ms)
    _, gradient_ms = grad_fn(ms_tensor)
    ms_grad=gradient_ms
    assert check(np.asarray(torch_grad.detach()),ms_grad.asnumpy())
      
if __name__=="__main__":
    test_any_forward_back(ms.GRAPH_MODE)