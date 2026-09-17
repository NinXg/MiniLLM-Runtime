"""一起测量延迟吞吐和显存"""

import argparse
import statistics
from time import perf_counter
import torch
from . import LLMEngine, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="请简单介绍大语言模型")
    parser.add_argument("--num-prompts", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if min(args.num_prompts, args.max_tokens, args.repeat) < 1 or args.warmup < 0:
        parser.error("测试次数和请求规模必须有效")
    engine = LLMEngine(args.model, device=args.device)
    device = engine.runner.device

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    try:
        prompt = engine.tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=False, add_generation_prompt=True)
        prompts = [prompt] * args.num_prompts
        params = SamplingParams(max_tokens=args.max_tokens)
        for _ in range(args.warmup):
            engine.generate(prompts, params)
        synchronize()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        times, tokens = [], 0
        for _ in range(args.repeat):
            synchronize()
            start = perf_counter()
            results = engine.generate(prompts, params)
            synchronize()
            times.append(perf_counter() - start)
            tokens += sum(len(result["token_ids"]) for result in results)

        # 延迟表示整批请求从提交到全部完成的时间
        print(f"batch mean latency: {statistics.mean(times) * 1000:.2f} ms")
        print(f"batch p50 latency: {statistics.median(times) * 1000:.2f} ms")
        print(f"requests/s: {args.repeat * args.num_prompts / sum(times):.2f}")
        print(f"output tokens/s: {tokens / sum(times):.2f}")
        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated(device) / 1024 ** 3
            print(f"peak allocated memory: {peak:.3f} GiB")
        print(f"device: {device}")
    finally:
        engine.close()


if __name__ == "__main__":
    main()
