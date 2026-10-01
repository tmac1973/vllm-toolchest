<!-- Excerpt of the model card of Qwen/Qwen3-14B-AWQ, fetched 2026-09-30, cut down to what autoconfigure reads. -->

## Deployment


For deployment, you can use `sglang>=0.4.6.post1` or `vllm>=0.8.5` or to create an OpenAI-compatible API endpoint:
- SGLang:
    ```shell
    python -m sglang.launch_server --model-path Qwen/Qwen3-14B-AWQ --reasoning-parser qwen3
    ```
- vLLM:
    ```shell
    vllm serve Qwen/Qwen3-14B-AWQ --enable-reasoning --reasoning-parser deepseek_r1
    ```

Also check out our [AWQ documentation](https://qwen.readthedocs.io/en/latest/quantization/awq.html) for more usage guide.

## Processing Long Texts

  For `vllm`, you can use
    ```shell
    vllm serve ... --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' --max-model-len 131072  
    ```

