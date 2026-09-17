# miniLLM

使用PyTorch的分页注意力练习项目

## 安装和运行

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m miniLLM.model --model ~/huggingface/Qwen2.5-0.5B-Instruct --max-tokens 64
python -m miniLLM.benchmarks --model ~/huggingface/Qwen2.5-0.5B-Instruct
```

## 参考
[nanovllm 源代码](https://github.com/GeeeekExplorer/nano-vllm)
[Qwen2.5 官方配置](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/config.json)
[Transformers Qwen2 源代码](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen2/modeling_qwen2.py)
