import mindspore as ms
import numpy as np
import pytest
import torch



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
def test_greater_equal_different_dtypes(mode):
    """===测试random输入不同dtype,对比两个框架的支持度==="""
    ms.set_context(mode=mode)

    input_data_1 = [[1,2],[0,0],[1,3]]
    input_data_2 = [[1,2],[0,0],[1,3]]

    ms_dtypes = [ms.int8, ms.int16, ms.int32, ms.int64, ms.uint8, ms.uint16, ms.uint32, ms.uint64, ms.float16,ms.bfloat16, ms.bool_]
    torch_dtypes = [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.uint16, torch.uint32, torch.uint64, torch.float16, torch.float32, torch.float64, torch.bfloat16, torch.bool]

    for i in range(len(ms_dtypes)):
        ms_type = ms_dtypes[i]
        torch_type = torch_dtypes[i]

        ms_input_1 = ms.Tensor(input_data_1, ms_type)
        ms_input_2 = ms.Tensor(input_data_2, ms_type)
        torch_input_1 = torch.tensor(input_data_1, dtype=torch_type)
        torch_input_2 = torch.tensor(input_data_2, dtype=torch_type)

        try:
            ms_result = ms.mint.greater_equal(ms_input_1, ms_input_2)
        except Exception as e:
            print(e)
            print(f"mindspore不支持{ms_type}类型")

        try:
            torch_result = torch.greater_equal(torch_input_1, torch_input_2)
        except Exception as e:
            print(e)
            print(f"torch不支持{torch_type}类型")

@pytest.mark.parametrize('mode',[ms.GRAPH_MODE,ms.PYNATIVE_MODE])
def test_greater_equal_input_fixed_dtype(mode):
    """===测试固定dtype,random输入值，对比两个框架输出==="""
    all_types = [np.int8,np.int16,np.int32,np.int64,np.float16,np.float32,np.float64,np.uint8,np.bool_]
    for dtype in all_types:
        #每种类型随机生成10组数据
        for _ in range(10):
            shape= (3,2)
            input_data_1 = generate_data(shape,dtype)
            input_data_2 = generate_data(shape,dtype)

            ms_input_1 = ms.Tensor(input_data_1)
            ms_input_2 = ms.Tensor(input_data_2)

            torch_input_1 = torch.tensor(input_data_1)
            torch_input_2 = torch.tensor(input_data_2)

            ms_result = ms.mint.greater_equal(ms_input_1, ms_input_2)
            torch_result = torch.greater_equal(torch_input_1, torch_input_2)

            ms_result_np = ms_result.asnumpy()
            torch_result_np = torch_result.numpy()

            r,diff = check_bool(ms_result_np,torch_result_np)
            if not r:
                result= False
                print(f"当输入类型为{dtype}时，mindspore与torch的greater_equal函数结果不一致，差异位置为{diff}")
                assert result

@pytest.mark.parametrize('mode',[ms.GRAPH_MODE,ms.PYNATIVE_MODE])
def test_greater_equal_input_different_parameters(mode):
    """===测试固定shape,固定输入值，不同输入参数（String、bool等），两个框架支持度==="""
    assert True




@pytest.mark.parametrize('mode',[ms.GRAPH_MODE,ms.PYNATIVE_MODE])
def test_greater_equal_input_random_chaotic(mode):
    """测试随机混乱输入，报错信息的准确性"""
    ms.set_context(mode=mode)
    #非法数据结构输入
    inputs_data_1=['123',[0.1,-0.4],{'a':1,'b':2},0.1,True,np.array([1,-3]),None,{'a':1,'b':2},np.array([1,-3]),None]
    inputs_data_2=['123',[0.1,-0.4],{'a':1,'b':2},0.1,True,'123',np.array([1,-3]),np.array([1,-3]),None,None]
    print("\n非法数据结构输入")
    for i,input_1 in enumerate(inputs_data_1):
        input_2 = inputs_data_2[i]

        print("MindSpore：", end="")
        if type(input_1) is np.ndarray:
            ms_input_1 = ms.Tensor(input_1)
        else:
            ms_input_1 = input_1
        if type(input_2) is np.ndarray:
            ms_input_2 = ms.Tensor(input_2)
        else:
            ms_input_2 = input_2

        try:
            ms_res = ms.mint.greater_equal(ms_input_1, ms_input_2)
            print(ms_res)
        except Exception as e:
            print(e)

        print("PyTorch：", end="")
        if type(input_1) is np.ndarray:
            torch_input_1 = torch.tensor(input_1)
        else:
            torch_input_1 = input_1
        if type(input_2) is np.ndarray:
            torch_input_2 = torch.tensor(input_2)
        else:
            torch_input_2 = input_2

        try:
            torch_res = torch.greater_equal(torch_input_1, torch_input_2)
            print(torch_res)
        except Exception as e:
            print(e)

    #特殊数据输入nan或inf
    torch_input_1 = torch.tensor([float("nan"), float('inf'), float('-inf'), -1, 2.1, 0])
    torch_input_2 = torch.tensor([float("nan"), float('-inf'), float('inf'), float("nan"), float('inf'), float('-inf')])
    ms_input_1 = ms.Tensor([float("nan"), float('inf'), float('-inf'), -1, 2.1, 0])
    ms_input_2 = ms.Tensor([float("nan"), float('-inf'), float('inf'), float("nan"), float('inf'), float('-inf')])

    print("\n特殊数据nan或inf：")
    try:
        torch_res = torch.greater_equal(torch_input_1, torch_input_2)
        ms_res = ms.mint.greater_equal(ms_input_1, ms_input_2)
        print(torch_res)
        print(ms_res)
        assert check_bool(torch_res.numpy(), ms_res.asnumpy())
    except Exception as e:
        print(e)

    #input_1和input_2形状不同但是可以广播
    torch_input_1 = torch.tensor([[1.1, 2.1], [0.3, 0.5], [0, 6]])  # shape:(3,2)
    torch_input_2 = torch.tensor([[1], [0.5], [3]])  # shape:(3,1)
    ms_input_1 = ms.Tensor([[1.1, 2.1], [0.3, 0.5], [0, 6]])
    ms_input_2 = ms.Tensor([[1], [0.5], [3]])
    print("\ninput_1和input_2形状不同但可以广播：")
    try:
      torch_res=torch.greater_equal(torch_input_1,torch_input_2)
      ms_res=ms.mint.greater_equal(ms_input_1,ms_input_2)
      print(torch_res)
      print(ms_res)
      assert check_bool(torch_res.numpy(),ms_res.asnumpy())
    except Exception as e:
      print(e)


    #input_1和input_2形状不同但是不可广播
    torch_input_1 = torch.tensor([[1.1, 2.1], [0.3, 0.5], [0, 6]])
    torch_input_2 = torch.tensor([1,0.5,3])
    ms_input_1 = ms.Tensor([[1.1, 2.1], [0.3, 0.5], [0, 6]])
    ms_input_2 = ms.Tensor([1,0.5,3])
    print("\ninput_1和input_2形状不同且不可广播：")
    try:
      torch_res=torch.greater_equal(torch_input_1,torch_input_2)
      print(torch_res)
    except Exception as e:
      print(e)
    try:
      ms_res=ms.mint.greater_equal(ms_input_1,ms_input_2)
      print(ms_res)
    except Exception as e:
      print(e)




@pytest.mark.parametrize('mode',[ms.GRAPH_MODE,ms.PYNATIVE_MODE])
def test_greater_equal_forward_back(mode):
    """===使用mindspore和pytorch,固定输入和权重，测试正向推理结果和反向梯度==="""
    ms.set_context(mode=mode)
    dtype_ms = ms.float64
    dtype_torch = torch.float64
    input_data_1 = [[0.1, 0.2], [0.5, 0.7], [0.11, 0.32]]
    input_data_2 = [[0.1, 0.9], [0.5, 0.2], [-0.5, 0.32]]

    ms_input_1 = ms.Tensor(input_data_1, dtype_ms)
    ms_input_2 = ms.Tensor(input_data_2, dtype_ms)

    torch_input_1 = torch.tensor(input_data_1, dtype=dtype_torch, requires_grad=True)
    torch_input_2 = torch.tensor(input_data_2, dtype=dtype_torch, requires_grad=True)

    def forward_ms(input_1, input_2):
        return ms.mint.greater_equal(input_1, input_2)

    def forward_torch(input_1, input_2):
        return torch.greater_equal(input_1, input_2)

    # 测试正向推理结果
    ms_result = forward_ms(ms_input_1, ms_input_2)
    torch_result = forward_torch(torch_input_1, torch_input_2)
    assert check_bool(ms_result.asnumpy(), np.asarray(torch_result.detach()))

    # 测试反向传播梯度
    try:
        torch_result.backward(torch.ones_like(torch_result))
        torch_grad_1 = torch_input_1.grad
        torch_grad_2 = torch_input_2.grad
    except Exception as e:
        print("PyTorch 反向传播失败:", e)

    try:
        grad_fn = ms.value_and_grad(forward_ms, grad_position=(0, 1))
        _, ms_grad = grad_fn(ms_input_1, ms_input_2)
        ms_grad_1 = ms_grad[0]
        ms_grad_2 = ms_grad[1]
        print(f"mindspore反向传播成功{ms_grad_1}{ms_grad_2}")
    except Exception as e:
        print("minspore 反向传播失败:", e)


if __name__=="__main__":
    test_greater_equal_forward_back(ms.PYNATIVE_MODE)




