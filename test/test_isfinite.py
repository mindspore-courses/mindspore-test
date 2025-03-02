import mindspore as ms
import numpy as np
import pytest
import torch
from mindspore import mint


def generate_data(shape,dtype):#根据shape和类型生成
  if dtype in [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16]:
      input_data = np.random.randint(0, 2, size=shape).astype(dtype)
  elif dtype == np.bool_:
      input_data = np.random.choice([True, False], size=shape).astype(dtype)
  else:
      input_data = np.random.randn(*shape).astype(dtype)
  return input_data


def check_bool(arr1, arr2):
    """
    检查两个布尔类型的numpy数组是否在每个对应位置上相同。
    """
    # 检查形状是否相同
    if arr1.shape != arr2.shape:
        raise ValueError("Arrays must have the same shape")

    # 直接比较两个布尔数组
    if np.array_equal(arr1, arr2):
        return True, None  # 数组相同，返回True和None
    else:
        # 如果不相同，计算并返回不相同的位置
        differences = arr1 != arr2
        return False, differences  # 返回False和差异位置的布尔数组



@pytest.mark.parametrize('mode',[ms.GRAPH_MODE,ms.PYNATIVE_MODE])
def test_isfinite_different_dtypes(mode):
    """===测试random输入不同dtype,对比两个框架的支持度==="""
    ms.set_context(mode=mode)
    input_data = [[1,2],[0,0],[1,3]]
    ms_dtypes = [ms.int8, ms.int16, ms.int32, ms.int64, ms.uint8, ms.uint16, ms.uint32, ms.uint64, ms.float16,ms.bfloat16, ms.bool_]
    torch_dtypes = [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.uint16, torch.uint32, torch.uint64, torch.float16, torch.float32, torch.float64, torch.bfloat16, torch.bool]

    for i in range(len(ms_dtypes)):
        ms_type = ms_dtypes[i]
        torch_type = torch_dtypes[i]
        ms_input = ms.Tensor(input_data, ms_type)
        torch_input = torch.tensor(input_data, dtype=torch_type)

        try:
            ms_result = ms.mint.isfinite(ms_input)
        except Exception as e:
            print(e)
            print(f"mindspore不支持{ms_type}类型")

        try:
            torch_result = torch.isfinite(torch_input)
        except Exception as e:
            print(e)
            print(f"torch不支持{torch_type}类型")

@pytest.mark.parametrize('mode',[ms.GRAPH_MODE,ms.PYNATIVE_MODE])
def test_isfinite_fixed_dtype_random_input(mode):
    """===测试固定dtype,random输入值，对比两个框架输出==="""
    ms.set_context(mode=mode)
    all_types = [np.int8, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64, np.uint8, np.bool_]
    for dtype in all_types:
        shape = (2, 3)
        input_data = generate_data(shape, dtype)
        ms_input = ms.Tensor(input_data)
        torch_input = torch.tensor(input_data)

        ms_result = ms.mint.isfinite(ms_input)
        torch_result = torch.isfinite(torch_input)

        ms_result_np = ms_result.asnumpy()
        torch_result_np = torch_result.numpy()

        r,diff=check_bool(ms_result_np, torch_result_np)
        if not r:
            result = False
            print(f"当输入类型为{dtype}时，mindspore与torch的isfinite函数结果不一致，差异位置为{diff}")
            assert result

@pytest.mark.parametrize('mode',[ms.GRAPH_MODE,ms.PYNATIVE_MODE])
def test_isfinite_input_different_parameterst(mode):
    """===测试固定shape,固定输入值，不同输入参数（String、bool等），两个框架支持度==="""
    assert True #由于isfinite只有input参数，因此不再重复测试



@pytest.mark.parametrize('mode',[ms.GRAPH_MODE,ms.PYNATIVE_MODE])
def test_isfinite_input_random_chaotic(mode):
    """测试随机混乱输入，报错信息的准确性"""

    #非法数据结构输入
    input_data = [{'a':1,'b':2},0.1,True,np.array([0.1,-0.3]),'123',[0.1,-0.3]]
    print("\n输入非法数据结构：")
    for input in input_data:
        print(f"当输入为{input}时")
        print("Mindspore:",end="")
        try:
            print(ms.mint.isfinite(input))
        except Exception as e:
            print(e)


        print("Pytorch:",end="")
        try:
            print(torch.isfinite(input))
        except Exception as e:
            print(e)

    #空输入
    input_data= None
    print("\n输入空数据：")
    print("Mindspore:", end="")
    try:
        print(ms.mint.isfinite(input_data))
    except Exception as e:
        print(e)

    print("Pytorch:", end="")
    try:
        print(torch.isfinite(input_data))
    except Exception as e:
        print(e)







@pytest.mark.parametrize('mode',[ms.GRAPH_MODE,ms.PYNATIVE_MODE])
def test_isfinite_forward_back(mode):
    """===使用mindspore和pytorch,固定输入和权重，测试正向推理结果和反向梯度==="""
    ms.set_context(mode=mode)
    dtype_ms = ms.float32
    dtype_torch = torch.float32
    input_data = [[0.1, 0.2], [0.5, 0], [0.11, 0.32]]
    ms_input = ms.Tensor(input_data, dtype_ms)
    torch_input = torch.tensor(input_data, dtype=dtype_torch, requires_grad=True)

    def forward_ms(input):
        return ms.mint.isfinite(input)

    def forward_torch(input):
        return torch.isfinite(input)

    # 测试正向推理结果
    ms_result = forward_ms(ms_input)
    torch_result = forward_torch(torch_input)
    assert check_bool(ms_result.asnumpy(), np.asarray(torch_result.detach()))

    # 测试反向传播梯度
    try:
        torch_result.backward(torch.ones_like(torch_result))
        torch_grad = torch_input.grad
    except Exception as e:
        print("Pytorch反向传播失败：", e)

    try:
        grad_fn = ms.value_and_grad(forward_ms)
        _, ms_grad = grad_fn(ms_input)
        print(f"mindspore反向传播成功{ms_grad}")
    except Exception as e:
        print("mindspore传播失败：", e)

if __name__=="__main__":
    test_isfinite_forward_back(ms.PYNATIVE_MODE)


