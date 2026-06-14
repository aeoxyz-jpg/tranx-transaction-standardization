def make_mlx_chat(model_path: str, adapter_path: str, max_tokens: int = 64):
    """Return a chat_fn(prompt)->str backed by an MLX model + LoRA adapter,
    running in-process (no server, no Ollama). Deterministic (temperature 0)."""
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(model_path, adapter_path=adapter_path)
    sampler = make_sampler(temp=0.0)

    def chat(prompt: str) -> str:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, tokenize=False)
        return generate(model, tokenizer, prompt=text, max_tokens=max_tokens,
                        sampler=sampler, verbose=False)

    return chat
