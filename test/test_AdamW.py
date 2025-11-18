import torch
import mindspore as ms
import numpy as np
from mindspore import value_and_grad

import mindspore
from mindspore import mint
from mindspore.mint import optim

import mindspore.nn as nn
from mindspore.common.initializer import Normal
import random
import pytest

class LeNet5_ms(nn.Cell):
    def __init__(self, num_class=10, num_channel=1, include_top=True):
        super(LeNet5_ms, self).__init__()
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.fc1 = nn.Dense(16 * 5 * 5, 120, weight_init='zeros', bias_init='zeros')
        self.fc2 = nn.Dense(120, 84, weight_init='zeros', bias_init='zeros')
        self.fc3 = nn.Dense(84, num_class, weight_init='zeros', bias_init='zeros')

    def construct(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class LeNet5_pt(torch.nn.Module):
    def __init__(self, num_class=10, num_channel=1, include_top=True):
        super(LeNet5_pt, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.relu = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        torch.nn.init.zeros_(self.fc1.weight)
        torch.nn.init.zeros_(self.fc1.bias)
        
        self.fc2 = torch.nn.Linear(120, 84)
        torch.nn.init.zeros_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc2.bias)
        
        self.fc3 = torch.nn.Linear(84, num_class)
        torch.nn.init.zeros_(self.fc3.weight)
        torch.nn.init.zeros_(self.fc3.bias)
            
    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def test_forward_backward():
    """
    (2) 测试两个框架的前向推理结果和反向梯度差异
    """
    np.random.seed(125)
    data_lst = np.random.rand(64, 1, 20, 20) * 100
    label_lst = np.random.randint(0, 9, (64, ))

    # mindspore 输入
    data_ms = ms.tensor(data_lst, dtype=mindspore.float32)
    label_ms = ms.tensor(label_lst, dtype=mindspore.int64)

    # pytorch 输入
    data_pt = torch.tensor(data_lst, dtype=torch.float32)
    label_pt = torch.tensor(label_lst, dtype=torch.int64)

    # 获取mindspore框架下网络的前项输出和梯度
    def ms_func():
        ms_net = LeNet5_ms()
        loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        optimizer = optim.AdamW(ms_net.trainable_params(), lr=0.01)
        def forward_fn(data, label):
            logits = ms_net(data)
            loss = loss_fn(logits, label)
            return loss, logits

        grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

        def train_step(data, label):
            (loss, logits), grads = grad_fn(data, label)
            optimizer(grads)
            return logits, grads
        
        for i in range(10):
            logits, grads = train_step(data_ms, label_ms)
            if i == 9:
                return logits, grads
        
        
    # 获取pytorch框架下网络的前项输出和参数
    def pt_func():
        pt_net = LeNet5_pt()
        loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        optimizer = torch.optim.AdamW(pt_net.parameters(), lr=0.01)
        def forward_fn(data, label):
            logits = pt_net(data)
            loss = loss_fn(logits, label)
            return loss, logits
        
        def train_step(data, label):
            optimizer.zero_grad()
            loss, logits = forward_fn(data, label)
            loss.backward()
            optimizer.step()
            return loss.sum(), logits
        
        for i in range(10):
            loss, logits = train_step(data_pt, label_pt)
            if i == 9:
                return logits, pt_net.parameters()
        
    ms_logits, ms_grad_tuple = ms_func()
    pt_logits, pt_params = pt_func()
    
    # 测试两个框架的梯度差异
    i = 0
    for param in pt_params:
        gradient_pt = param.grad.detach().numpy()
        gradient_ms = ms_grad_tuple[i].numpy()
        assert np.allclose(gradient_pt, gradient_ms, rtol=1e-3)
        i += 1
    
    # 测试两个框架的前向logits输出
    assert  np.allclose(pt_logits.detach().numpy(), ms_logits.numpy(), rtol=1e-3)

def test_random_input():
    """
    (1b) 测试随机输入值的情况下，测试两个框架下被优化器优化的参数的相似度
    """
    input_ = np.random.rand(4) * 100
    x_ = np.random.rand(4) * 100
    
    # torch 框架    
    input_pt = torch.tensor(input_, dtype=torch.float32, requires_grad=True)
    x_pt = torch.tensor(x_, dtype=torch.float32)
    optimizer_pt = torch.optim.AdamW([input_pt])
    def forward_fn(x, params):
        result = x * params * params + 2 * x * params + x
        return result.sum()
    
    for i in range(10):
        optimizer_pt.zero_grad()
        loss = forward_fn(x_pt, input_pt)
        loss.backward()
        optimizer_pt.step()
    
    # mindspore 框架
    input_ms = ms.tensor(input_, dtype=ms.float32)
    input_param = ms.Parameter(input_ms, requires_grad=True)
    x_ms = ms.tensor(x_, dtype=ms.float32)
    x_param = ms.Parameter(x_ms)
    
    optimizer_ms = ms.mint.optim.AdamW([input_param])
    grad_fn = ms.value_and_grad(fn=forward_fn, grad_position=1, weights=None)
    
    for i in range(10):
        loss, grad = grad_fn(x_param, input_param)
        optimizer_ms((grad,))
    
    assert np.allclose(input_pt.detach().numpy(), input_ms.numpy(), rtol=1e-3)
    
    
@pytest.mark.parametrize('mode', ['Tensor', 'None', 'List', 'Dict', 'Tuple', 'int_lr', 'minus_lr', 'minus_eps', 'betas', 'weight_decay', 'messy_input'])
def test_chaotic_input(mode):
    """
    (1d) 测试随机混乱输入，报错信息的准确性
    """
    # 测试输入参数为张量的情况
    if mode == 'Tensor':
        reported_flag = False
        input_ms = ms.tensor([1.1, 1.2, 1.3, 1.4], dtype=ms.float32)
        try:
            optimizer = ms.mint.optim.AdamW(params=input_ms)
        except Exception as e:
            reported_flag = True
            e = str(e)
            assert 'Tensor' in e
            print(e)
        if reported_flag == False:
            assert "No error message was reported when the input is tensor." == 1
    
    # 测试输入为空的情况
    if mode == 'None':
        reported_flag = False
        try:
            optimizer = ms.mint.optim.AdamW(params=None)
        except Exception as e:
            reported_flag = True
            e = str(e)
            assert 'None' in e
            print(e)
        if reported_flag == False:
            assert "No error message was reported when the input is None." == 1
        
    # 测试输入为列表的情况
    if mode == 'List':
        reported_flag = False
        try:
            optimizer = ms.mint.optim.AdamW(params=[1, 2, 3, 4, 5])
        except Exception as e:
            reported_flag = True
            e = str(e)
            assert 'List' in e
            print(e)
        if reported_flag == False:
            assert "No error message was reported when the input is List." == 1
    # 测试问题：报错存在异常: can only concatenate str (not "type") to str
    
    # 测试输入为字典的情况
    if mode == 'Dict':
        reported_flag = False
        try:
            optimizer = ms.mint.optim.AdamW(params={"a":1, "b":2})
        except Exception as e:
            reported_flag = True
            e = str(e)
            assert 'Dict' in e
            print(e)
        if reported_flag == False:
            assert "No error message was reported when the input is Dict." == 1
    # 测试问题：报错存在异常: can only concatenate str (not "type") to str
    
    # 测试输入为元祖的情况
    if mode == 'Tuple':
        reported_flag = False
        try:
            optimizer = ms.mint.optim.AdamW(params=(1, 2, 3))
        except Exception as e:
            reported_flag = True
            e = str(e)
            assert 'Tuple' in e
            print(e)
        if reported_flag == False:
            assert "No error message was reported when the input is Tuple." == 1
    # 测试问题：报错存在异常: can only concatenate str (not "type") to str
    
    # 测试学习率不为浮点数的情况
    ms_net = LeNet5_ms()
    if mode == 'int_lr':
        reported_flag = False
        try:
            optimizer = ms.mint.optim.AdamW(params=ms_net.trainable_params(), lr=1)
        except Exception as e:
            print(e)
            reported_flag = True
            e = str(e)
            assert 'float' in e and 'int' in e
        if reported_flag == False:
            assert "No error message was reported when the type of learning rate is int." == 1
    
    # 测试学习率为负数的情况
    if mode == 'minus_lr':
        reported_flag = False
        try:
            optimizer = ms.mint.optim.AdamW(params=ms_net.trainable_params(), lr=-0.5)
        except Exception as e:
            print(e)
            reported_flag = True
            e = str(e)
            assert 'learning rate' in e
        if reported_flag == False:
            assert "No error message was reported when learning rate is negative." == 1
    
    # 测试eps为负数的情况
    if mode == 'minus_eps':
        reported_flag = False
        try:
            optimizer = ms.mint.optim.AdamW(params=ms_net.trainable_params(), eps=-1e-5)
        except Exception as e:
            print(e)
            reported_flag = True
            e = str(e)
            assert 'epsilon' in e
        if reported_flag == False:
            assert "No error message was reported when epsilon is negative." == 1
    
    # 测试betas范围不在[0,1)区间的情况
    if mode == 'betas':
        betas_lst = [(-0.5, 1.5), (1.0, 1.5), (0.5, -0.5), (0.5, 1.0)]
        for betas in betas_lst:
            reported_flag = False
            try:
                optimizer = ms.mint.optim.AdamW(params=ms_net.trainable_params(), betas=betas)
            except Exception as e:
                print(e)
                reported_flag = True
                e = str(e)
                assert 'beta' in e
            if reported_flag == False:
                assert "No error message was reported when beta parameter is invalid." == 1
        
    # 测试权重衰减参数为负数的情况
    if mode == 'weight_decay':
        reported_flag = False
        try:
            optimizer = ms.mint.optim.AdamW(params=ms_net.trainable_params(), weight_decay=-0.5)
        except Exception as e:
            reported_flag = True
            print(e)
            e = str(e)
            assert 'weight_decay' in e
        if reported_flag == False:
            assert "No error message was reported when weight_decay is negative." == 1
    
    # 测试乱序输入
    if mode == 'messy_input':
        reported_flag = False
        try:
            optimizer = ms.mint.optim.AdamW(params=(0.5, 0.5), betas=ms_net.trainable_params(), weight_decay=False, maximize=True)
        except Exception as e:
            reported_flag = True
            e = str(e)
            print(e)
            assert 'tuple' in e and 'list' in e
        if reported_flag == False:
            assert "No error message was reported when the input is messy" == 1

def test_param_type():
    """
    (1a) 测试优化器的params参数类型
    """
    # mindspore框架
    input_ms1 = ms.tensor([1.0, 1.1, 1.2, 1.3], dtype=ms.float32)
    input_ms2 = ms.tensor([2.0, 2.1, 2.2, 2.3], dtype=ms.float32)
    param1 = ms.Parameter(input_ms1, requires_grad=True)
    param2 = ms.Parameter(input_ms2, requires_grad=True)
    # 测试Parameter组成的列表
    try:
        optimizer = ms.mint.optim.AdamW([param1, param2])
    except Exception as e:
        print(e)
    
    # 测试Parameter字典列表
    param_dict = [{"params":param1}, {"params": param2}]
    try:
        optimizer = ms.mint.optim.AdamW(param_dict)
    except Exception as e:
        print(e)
    
    # pytorch框架
    input_pt1 = torch.tensor([1.0, 1.1, 1.2, 1.3], dtype=torch.float32, requires_grad=True)
    input_pt2 = torch.tensor([2.0, 2.1, 2.2, 2.3], dtype=torch.float32, requires_grad=True)
    
    # 测试张量列表
    try:
        optimizer = torch.optim.AdamW([input_pt1, input_pt2])
    except Exception as e:
        print(e)
        
    # 测试字典列表
    try:
        optimizer = torch.optim.AdamW([{"params": input_pt1}, {"params": input_pt2}])
    except Exception as e:
        print(e)
    
ms.set_context(device_target='Ascend')
pytest.main(['-vs', 'test_AdamW.py', '--html', './report/report.html'])