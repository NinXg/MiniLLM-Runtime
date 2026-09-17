# miniLLM

一个纯 PyTorch 分页注意力教学项目

## 这版做了什么

- 使用真实 K 和 V 张量替代原来的缓存元数据占位
- 通过每个请求的 block table 读取不同物理块
- Prefill 和 Decode 都经过自己的 paged_attention
- 保留请求轮转调度和 EOS 及长度终止
- 保留贪心采样和 temperature 及 top_k 和 top_p
- 保留测试并将延迟吞吐显存测试合并到 benchmarks.py
- 删除 kernels 目录及全部自定义 CUDA 扩展

Python 文件从原来的 26 个缩减到 13 个
config.py 的配置合并到引擎构造参数
allocator.py 和 block.py 合并到 kv_cache/manager.py
删除未接入模型计算的 Prefix Cache 原型
删除重复的结构说明和进度条包装

## 安装和运行

先进入解压后的 miniLLM_scaffold 目录
建议使用独立环境以免固定版本影响其他项目
如果已有可用的 GPU 版 PyTorch 可以继续使用

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m miniLLM.model --model ~/huggingface/Qwen2.5-0.5B-Instruct --max-tokens 64
python -m miniLLM.benchmarks --model ~/huggingface/Qwen2.5-0.5B-Instruct
```

没有显卡时加上 --device cpu
脚本用 python -m 运行而不是直接执行文件路径
测试使用随机小模型无需下载真实权重

## 阅读顺序

| 文件 | 作用 |
| --- | --- |
| miniLLM/runtime/kv_cache/manager.py | 分配物理块并按位置写入真实缓存 |
| miniLLM/runtime/attention.py | 根据块表计算注意力 |
| miniLLM/runtime/model_runner.py | 复用权重完成整个模型前向 |
| miniLLM/runtime/sequence.py | 保存请求及已经计算的长度 |
| miniLLM/runtime/scheduler.py | 请求排队与块回收 |
| miniLLM/runtime/engine.py | 编码生成与解码 |
| miniLLM/sampling_params.py | 生成长度和采样参数 |
| miniLLM/model.py | 命令行入口 |
| miniLLM/benchmarks.py | 延迟吞吐和显存测试 |
| tests/test_paged.py | 分页注意力与模型对齐测试 |

## 分页存储

K 和 V 分别存为五维张量
`[num_layers, num_blocks, block_size, num_kv_heads, head_dim]`

逻辑位置先除以 block_size 得到逻辑块号
再用 block_table 找到物理块号
位置对 block_size 取余得到块内偏移
所以不同请求可以使用同一个物理池而不要求各自的块相邻

Qwen2.5 的查询头数和缓存头数不同
缓存只保存 KV 头
计算时用 repeat_interleave 将它们对应到查询头

Prefill 计算全部输入并缓存它们的 K 和 V
采样出的第一个词此时还没有 K 和 V
下一轮 Decode 才计算这个词并追加缓存
cached 记录已经完成前向的词数
它不等于当前输入和所有输出的总长度

## 注意力计算

先逐块读取 K 计算分数
用绝对位置遮住未来的词
拼接各块的分数后统一做 Softmax
再逐块读取 V 得到加权和
这里拼接的是分数而不是整段历史 K 和 V
不能让每个块独立做 Softmax 后直接相加
最后一个块只读取有效位置

这是简单的两遍计算范例
Prefill 仍会保存完整注意力分数
不是使用在线 Softmax 的高性能实现

## 使用边界

- 模型前向针对 Qwen2 架构编写
- 目标是普通全注意力的 Qwen2.5 模型
- 不支持 Qwen3 以及滑动窗口和缩放 RoPE
- 固定 Transformers 4.51.3 以避免内部模型结构变化
- Transformers 用于加载权重和提供线性层及归一化层
- 生成不调用 model.generate 或模型原有的 Attention
- 不使用 Transformers 的 past_key_values
- 多请求按轮执行但仍然逐请求前向
- 未实现融合 batch kernel 以及抢占和 Prefix Cache
- 接纳请求时预留其最大生成长度所需的块
- 预留策略简单且不会中途缺块但利用率低于按需分配
- 默认缓存池含 256 个块且每块可放 16 个词
- 池容量不足时可以调大 LLMEngine 的 num_blocks 参数
- 纯 PyTorch 可使用 CPU 或 GPU 但不需要自行编译 CUDA
- 这版用于理解数据流不承诺比原版更快

## 测试和指标

测试包含非连续块读取和末块未填满
也包含因果遮罩和不同查询头数
随机小型 Qwen 模型的 Prefill 与多步 Decode 对齐官方前向的 logits
多请求测试检查请求隔离以及完成和异常后的块回收
命令行入口和 benchmark 也有冒烟测试
GPU 测试在没有显卡时自动跳过

benchmark 先预热后多次计时
GPU 计时前后均同步
延迟表示整批请求从提交到全部完成的耗时
设置 --num-prompts 1 才是单请求延迟
吞吐统计实际生成的词数
显存指标包含权重和预分配缓存池及临时张量
不同长度及提前遇到 EOS 的结果不宜直接横向比较

## 参考

[Qwen2.5 官方配置](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/config.json)
[Transformers Qwen2 源码](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen2/modeling_qwen2.py)
