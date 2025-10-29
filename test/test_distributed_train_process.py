import os
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import ops
from mindspore import set_context
import mindspore.dataset as ds
import mindspore.mint.distributed as ms_dist

# 定义一个简单的MLP网络
class SimpleMLP(nn.Cell):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Dense(28*28, 1024*2)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Dense(1024*2, 64*5)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Dense(64*5, 10)

    def construct(self, x):
        x = self.flatten(x)
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        return self.layer3(x)

def create_dataset(batch_size):
    # 创建随机数据用于测试
    data = np.random.randn(1000, 28, 28).astype(np.float32)
    label = np.random.randint(0, 10, (1000,)).astype(np.int32)
    
    dataset = ds.NumpySlicesDataset(
        {"data": data, "label": label}, 
        shuffle=True
    )
    dataset = dataset.batch(batch_size)
    return dataset

def main():
    
    # 设置运行环境
    set_context(mode=ms.GRAPH_MODE, device_target="Ascend")
    
    # 使用mint接口初始化分布式环境
    ms_dist.init_process_group(
        backend='hccl',  # Ascend使用hccl后端
        world_size=4,    # 总进程数
    )
    rank_id = ms_dist.get_rank()
    rank_size = ms_dist.get_world_size()
    print(f"当前进程 rank_id: {rank_id}, 总进程数 rank_size: {rank_size}")

    # 创建数据集
    dataset = create_dataset(batch_size=32)
    
    # 创建网络、损失函数和优化器
    network = SimpleMLP()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = nn.Adam(network.trainable_params())

    def forward_fn(data, label):
        logits = network(data)
        loss = loss_fn(logits, label)
        return loss

    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters)

    def train_step(data, label):
        loss, grads = grad_fn(data, label)
        optimizer(grads)
        return loss

    try:
        # 训练循环
        epochs = 5
        for epoch in range(epochs):
            total_loss = 0
            steps = 0
            for data in dataset.create_dict_iterator():
                loss = train_step(data["data"], data["label"])
                total_loss += loss
                steps += 1
                
                print(f"Epoch: {epoch}, Step: {steps}, Loss: {loss}")
            
            print(f"Epoch: {epoch}, 平均损失: {total_loss/steps}")
    
    finally:
        # 清理分布式环境
        ms_dist.destroy_process_group()
        print("分布式环境已清理")

if __name__ == "__main__":
    main() 