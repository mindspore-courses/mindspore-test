import os
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import ops
from mindspore import set_context
import mindspore.dataset as ds
import torch_npu
import torch.distributed as dist
from mindspore.mint.distributed import init_process_group, destroy_process_group, get_world_size, get_rank
import pytest


@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():
    """测试模块的初始化和清理"""
    # 将 msrun 设置的环境变量映射到 PyTorch 所需的环境变量
    os.environ["RANK"] = os.environ.get("RANK_ID", "0")
    os.environ["WORLD_SIZE"] = os.environ.get("RANK_SIZE", "1")
    os.environ["MASTER_ADDR"] = os.environ.get("MS_SCHED_HOST", "127.0.0.1")
    os.environ["MASTER_PORT"] = os.environ.get("MS_SCHED_PORT", "8491")

    print(f"当前 RANK: {os.environ['RANK']}")
    print(f"当前 WORLD_SIZE: {os.environ['WORLD_SIZE']}")
    print(f"MASTER_ADDR: {os.environ['MASTER_ADDR']}")
    print(f"MASTER_PORT: {os.environ['MASTER_PORT']}")
    
    set_context(device_target="Ascend", mode=ms.GRAPH_MODE)
    
    yield  # 测试执行
    
    print("测试执行完毕")

class TestMindSporeDistributed:
    """专门测试 MindSpore 分布式接口的测试类"""
    
    def test_ms_init_and_destroy(self):
        """测试 MindSpore 的分布式初始化和销毁"""
        print("\n测试 MindSpore 分布式初始化和销毁")
        # 初始化
        init_process_group()
        rank = get_rank()
        print(f"当前 MindSpore rank: {rank}")
        assert rank is not None, "MindSpore 初始化失败"
        
        # 销毁
        destroy_process_group()
        with pytest.raises(RuntimeError):
            get_rank()
        print("MindSpore 分布式环境已成功销毁")

    def test_ms_get_rank(self):
        """测试 MindSpore 的 rank 获取"""
        print("\n测试 MindSpore rank 获取")
        init_process_group()
        rank = get_rank()
        print(f"获取到的 MindSpore rank: {rank}")
        assert isinstance(rank, int), "MindSpore rank 不是整数"
        assert rank >= 0, "MindSpore rank 应该大于等于 0"
        destroy_process_group()

    def test_ms_get_world_size(self):
        """测试 MindSpore 的 world_size 获取"""
        print("\n测试 MindSpore world_size 获取")
        init_process_group()
        world_size = get_world_size()
        print(f"获取到的 MindSpore world_size: {world_size}")
        assert isinstance(world_size, int), "MindSpore world_size 不是整数"
        assert world_size > 0, "MindSpore world_size 应该大于 0"
        destroy_process_group()

    def test_ms_all_reduce(self):
        """测试 MindSpore 的 all_reduce 操作"""
        print("\n测试 MindSpore all_reduce 操作")
        init_process_group()
        tensor = ms.Tensor(np.array([1.0]), ms.float32)
        result = ops.AllReduce()(tensor)
        expected = ms.Tensor(np.array([get_world_size() * 1.0]), ms.float32)
        print(f"AllReduce 结果: {result.asnumpy()}, 期望值: {expected.asnumpy()}")
        assert np.allclose(result.asnumpy(), expected.asnumpy()), "AllReduce 结果不正确"
        destroy_process_group()

# 混合测试部分
class TestMixedDistributed:
    """测试 MindSpore 和 PyTorch 混合分布式接口"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """在每个测试前初始化 PyTorch 分布式环境"""
        dist.init_process_group(backend='hccl')
        yield
        dist.destroy_process_group()

    def test_init_and_destroy_process_group(self):
        """对比 MindSpore 和 PyTorch 的分布式初始化与销毁"""
        print("\n对比 MindSpore 和 PyTorch 的分布式初始化")
        # 验证初始化是否成功
        ms_rank = get_rank()
        torch_rank = dist.get_rank()
        print(f"MindSpore rank: {ms_rank}, PyTorch rank: {torch_rank}")
        assert ms_rank is not None, "MindSpore 初始化失败"
        assert torch_rank is not None, "PyTorch 初始化失败"
        
        # 验证销毁是否成功
        destroy_process_group()
        dist.destroy_process_group()
        
        # 验证销毁后获取 rank 是否报错
        with pytest.raises(RuntimeError):
            get_rank()
        with pytest.raises(RuntimeError):
            dist.get_rank()
        print("分布式环境已成功销毁")

    def test_get_rank(self):
        """对比 MindSpore 和 PyTorch 的 rank 获取"""
        print("\n对比 MindSpore 和 PyTorch 的 rank 获取")
        ms_rank = get_rank()
        torch_rank = dist.get_rank()
        print(f"MindSpore rank: {ms_rank}, PyTorch rank: {torch_rank}")
        assert ms_rank == torch_rank, f"Rank 不匹配: MindSpore {ms_rank} vs PyTorch {torch_rank}"
        assert isinstance(ms_rank, int), "MindSpore rank 不是整数"
        assert isinstance(torch_rank, int), "PyTorch rank 不是整数"

    def test_get_world_size(self):
        """对比 MindSpore 和 PyTorch 的 world_size 获取"""
        print("\n对比 MindSpore 和 PyTorch 的 world_size 获取")
        ms_world_size = get_world_size()
        torch_world_size = dist.get_world_size()
        print(f"MindSpore world_size: {ms_world_size}, PyTorch world_size: {torch_world_size}")
        assert ms_world_size == torch_world_size, f"World Size 不匹配: MindSpore {ms_world_size} vs PyTorch {torch_world_size}"
        assert isinstance(ms_world_size, int), "MindSpore world_size 不是整数"
        assert isinstance(torch_world_size, int), "PyTorch world_size 不是整数"

    def test_distributed_communication(self):
        """对比 MindSpore 和 PyTorch 的 all_reduce 操作"""
        print("\n对比 MindSpore 和 PyTorch 的 all_reduce 操作")
        # 创建测试数据
        ms_tensor = ms.Tensor(np.array([1.0]), ms.float32)
        torch_tensor = dist.new_group().new_tensor([1.0])
        
        # 执行 all_reduce 操作
        ms_result = ops.AllReduce()(ms_tensor)
        dist.all_reduce(torch_tensor, op=dist.ReduceOp.SUM)
        
        # 验证结果
        print(f"MindSpore AllReduce 结果: {ms_result.asnumpy()}, PyTorch AllReduce 结果: {torch_tensor.numpy()}")
        assert ms_result.asnumpy() == torch_tensor.numpy(), "AllReduce 结果不匹配"

if __name__ == "__main__":
    pytest.main()