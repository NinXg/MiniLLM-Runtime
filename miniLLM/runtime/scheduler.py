"""按轮推进请求"""

from collections import deque


class Scheduler:
    def __init__(self, cache, max_num_seqs=4):
        if type(max_num_seqs) is not int or max_num_seqs < 1:
            raise ValueError("并发请求数必须是正整数")
        self.cache = cache
        self.max_num_seqs = max_num_seqs
        self.waiting = deque()
        self.running = []

    def add(self, seq):
        if not seq.prompt or seq.capacity > self.cache.num_blocks * self.cache.block_size:
            raise ValueError("输入为空/请求超过缓存总容量")
        self.waiting.append(seq)

    def schedule(self):
        # 提前预留本次请求所需的块以避免中途内存不足
        while self.waiting and len(self.running) < self.max_num_seqs:
            seq = self.waiting[0]
            if self.cache.blocks_needed(seq.capacity) > len(self.cache.free_blocks):
                break
            self.waiting.popleft()
            seq.block_table = self.cache.allocate(seq.capacity)
            self.running.append(seq)
        return list(self.running)

    def postprocess(self, seq, token, eos_ids):
        seq.output.append(token)
        if token in eos_ids:
            seq.finish_reason = "eos"
        elif len(seq.output) >= seq.params.max_tokens:
            seq.finish_reason = "length"
        if seq.finish_reason:
            # 按对象身份删除以区分内容相同的请求
            self.running = [item for item in self.running if item is not seq]
            self.cache.release(seq.block_table)

    def clear(self):
        for seq in self.running:
            self.cache.release(seq.block_table)
        self.running.clear()
        self.waiting.clear()

    def is_finished(self):
        return not self.waiting and not self.running
