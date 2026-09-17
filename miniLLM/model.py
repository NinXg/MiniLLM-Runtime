"""运行模型的入口"""

import argparse
from .runtime.engine import LLMEngine
from .sampling_params import SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="~/huggingface/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--prompt", default="请用三句话介绍大语言模型")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)
    engine = LLMEngine(args.model, device=args.device)
    try:
        prompt = engine.tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=False, add_generation_prompt=True)
        print(engine.generate(prompt, params)[0]["text"])
    finally:
        engine.close()


if __name__ == "__main__":
    main()
