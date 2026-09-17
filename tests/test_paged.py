"""检查注意力和推理是否正确"""

import pytest
import torch
import subprocess
import sys
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import Qwen2Config, Qwen2ForCausalLM, PreTrainedTokenizerFast
from miniLLM import LLMEngine, SamplingParams
from miniLLM.runtime.attention import paged_attention
from miniLLM.runtime.kv_cache.manager import KVCache
from miniLLM.runtime.model_runner import ModelRunner
from miniLLM.runtime.scheduler import Scheduler
from miniLLM.runtime.sequence import Sequence

torch.set_num_threads(1)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("start,count,heads", [(0, 1, 2), (0, 7, 4), (6, 1, 4), (5, 3, 4)])
def test_attention_matches_dense(device, start, count, heads):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("没有可用显卡")
    torch.manual_seed(7)
    cache = KVCache(1, 8, 3, 2, 8, device=device)
    cache.k.fill_(float("nan"))
    cache.v.fill_(float("nan"))
    table = [6, 1, 4]
    length = start + count
    k = torch.randn(length, 2, 8, device=device)
    v = torch.randn_like(k)
    q = torch.randn(count, heads, 8, device=device)
    cache.write(0, table, 0, k, v)
    actual = paged_attention(q, cache, 0, table, start)

    keys = k.transpose(0, 1).repeat_interleave(heads // 2, dim=0)
    values = v.transpose(0, 1).repeat_interleave(heads // 2, dim=0)
    score = q.transpose(0, 1) @ keys.transpose(-1, -2) / 8 ** 0.5
    mask = torch.arange(length, device=device)[None, :] > torch.arange(start, length, device=device)[:, None]
    expected = (score.masked_fill(mask, float("-inf")).softmax(-1) @ values).transpose(0, 1)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_allocation_and_reuse():
    cache = KVCache(1, 4, 2, 1, 4)
    a, b = cache.allocate(3), cache.allocate(3)
    assert set(a).isdisjoint(b)
    with pytest.raises(ValueError):
        cache.allocate(1)
    freed = list(a)
    cache.release(a)
    c = cache.allocate(3)
    assert set(c) == set(freed)
    cache.release(c)
    with pytest.raises(ValueError):
        cache.release(freed)
    cache.release(b)
    assert len(cache.free_blocks) == 4


def test_scheduler_waits_and_releases():
    cache = KVCache(1, 2, 2, 1, 4)
    scheduler = Scheduler(cache, max_num_seqs=2)
    a = Sequence([1, 3, 4], SamplingParams(max_tokens=2))
    b = Sequence([1, 3, 4], SamplingParams(max_tokens=1))
    scheduler.add(a)
    scheduler.add(b)
    assert scheduler.schedule() == [a]
    scheduler.postprocess(a, 5, set())
    assert scheduler.schedule() == [a]
    scheduler.postprocess(a, 6, set())
    assert a.finish_reason == "length"
    assert scheduler.schedule() == [b]
    scheduler.postprocess(b, 2, {2})
    assert b.finish_reason == "eos"
    assert scheduler.is_finished()
    assert len(cache.free_blocks) == 2


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    # 使用随机小模型避免测试时下载权重
    torch.manual_seed(12)
    config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=2, num_attention_heads=4,
                        num_key_value_heads=2, max_position_embeddings=128,
                        bos_token_id=1, eos_token_id=2, pad_token_id=0,
                        tie_word_embeddings=True, use_sliding_window=False)
    model = Qwen2ForCausalLM(config).eval()
    directory = tmp_path_factory.mktemp("qwen")
    model.save_pretrained(directory)
    words = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3}
    words.update({f"word{i}": i for i in range(4, 64)})
    backend = Tokenizer(WordLevel(words, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]",
                                        bos_token="[BOS]", eos_token="[EOS]",
                                        unk_token="[UNK]")
    tokenizer.chat_template = "{{ messages[0]['content'] }}"
    tokenizer.save_pretrained(directory)
    return str(directory)


@pytest.mark.parametrize("prompt", [[1], [1, 4, 5], [1, 4, 5, 6], [1, 4, 5, 6, 7]])
def test_model_logits_match_hf(model_dir, prompt):
    model = Qwen2ForCausalLM.from_pretrained(model_dir, attn_implementation="eager")
    runner = ModelRunner(model, num_blocks=16, block_size=3)
    seq = Sequence(prompt, SamplingParams(max_tokens=5))
    seq.block_table = runner.cache.allocate(seq.capacity)
    with torch.inference_mode():
        for step in range(5):
            actual = runner.forward(seq)
            full = torch.tensor([seq.prompt + seq.output])
            expected = model(full, use_cache=False).logits[0, -1].float()
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
            assert actual.argmax().item() == expected.argmax().item()
            assert seq.cached == len(prompt) + step
            seq.output.append(actual.argmax().item())
    runner.cache.release(seq.block_table)


def test_engine_batch_reuse_and_cleanup(model_dir, monkeypatch):
    engine = LLMEngine(model_dir, device="cpu", num_blocks=8, block_size=3, max_num_seqs=2)
    prompts = [[1, 4, 5], [1, 7], [1, 4, 5]]
    params = SamplingParams(max_tokens=4)
    try:
        batch = engine.generate(prompts, params)
        singles = [engine.generate([prompt], params)[0] for prompt in prompts]
        assert batch == singles
        assert len(engine.runner.cache.free_blocks) == 8
        assert engine.generate([]) == []
        assert len(engine.generate(["word4 word5"], params)) == 1

        def fail(seq):
            raise RuntimeError("模拟模型计算失败")

        with monkeypatch.context() as patch:
            patch.setattr(engine.runner, "forward", fail)
            with pytest.raises(RuntimeError):
                engine.generate(prompts, params)
        assert len(engine.runner.cache.free_blocks) == 8
        assert engine.generate(prompts, params) == batch
    finally:
        engine.close()
    engine.close()
    with pytest.raises(RuntimeError):
        engine.generate(prompts)


@pytest.mark.parametrize("prompt", [[], [-1], [64], [True]])
def test_invalid_requests(model_dir, prompt):
    engine = LLMEngine(model_dir, device="cpu", num_blocks=4, block_size=4)
    try:
        with pytest.raises(ValueError):
            engine.generate([prompt], SamplingParams(max_tokens=1))
        assert len(engine.runner.cache.free_blocks) == 4
        with pytest.raises(ValueError):
            engine.generate(["word4"], SamplingParams(max_tokens=100))
        with pytest.raises(ValueError):
            engine.generate(["word4"], [])
    finally:
        engine.close()


@pytest.mark.parametrize("kwargs", [{"max_tokens": 0}, {"temperature": -1},
                                    {"temperature": float("nan")}, {"top_p": 0},
                                    {"top_k": -1}])
def test_invalid_sampling(kwargs):
    with pytest.raises(ValueError):
        SamplingParams(**kwargs)


def test_sampling():
    logits = torch.tensor([-2.0, 0.0, 8.0])
    for params in [SamplingParams(), SamplingParams(temperature=1, top_k=1),
                   SamplingParams(temperature=1, top_p=0.1)]:
        assert ModelRunner.sample(logits.clone(), params) == 2


@pytest.mark.parametrize("module", ["miniLLM.model", "miniLLM.benchmarks"])
def test_command_line(model_dir, module):
    command = [sys.executable, "-m", module, "--model", model_dir,
               "--device", "cpu", "--max-tokens", "2", "--prompt", "word4 word5"]
    if module.endswith("benchmarks"):
        command += ["--repeat", "1", "--warmup", "0", "--num-prompts", "2"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    if module.endswith("benchmarks"):
        assert "output tokens/s:" in result.stdout
