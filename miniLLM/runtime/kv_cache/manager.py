"""管理真实缓存张量和空闲块"""

import torch


class KVCache:
    def __init__(self, layers, num_blocks, block_size, kv_heads, head_dim,
                 dtype=torch.float32, device="cpu"):
        if min(layers, num_blocks, block_size, kv_heads, head_dim) < 1:
            raise ValueError("缓存维度必须大于零")
        self.block_size = block_size
        self.num_blocks = num_blocks
        # 依次是层数块数块内位置注意力头数头维度
        shape = (layers, num_blocks, block_size, kv_heads, head_dim)
        self.k = torch.empty(shape, dtype=dtype, device=device)
        self.v = torch.empty_like(self.k)
        self.free_blocks = list(range(num_blocks))

    def blocks_needed(self, length):
        return (length + self.block_size - 1) // self.block_size

    def allocate(self, length):
        count = self.blocks_needed(length)
        if length < 1 or count > len(self.free_blocks):
            raise ValueError("缓存容量不足")
        return [self.free_blocks.pop() for _ in range(count)]

    def release(self, table):
        if (len(set(table)) != len(table)
                or any(b < 0 or b >= self.num_blocks or b in self.free_blocks
                       for b in table)):
            raise ValueError("无效块号或重复释放")
        self.free_blocks.extend(table)
        table.clear()

    def write(self, layer, table, start, k, v):
        # 输入形状是词数注意力头数头维度
        end = start + k.shape[0]
        if start < 0 or end > len(table) * self.block_size:
            raise ValueError("写入位置超过已分配容量")
        if k.shape != v.shape or k.shape[1:] != self.k.shape[-2:]:
            raise ValueError("键和值的形状不匹配")
        positions = torch.arange(start, end, device=self.k.device)
        table_tensor = torch.tensor(table, device=self.k.device)
        blocks = table_tensor[positions // self.block_size]
        offsets = positions % self.block_size
        self.k[layer, blocks, offsets] = k
        self.v[layer, blocks, offsets] = v
