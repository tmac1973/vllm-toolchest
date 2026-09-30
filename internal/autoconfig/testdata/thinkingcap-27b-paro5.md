<!-- Excerpt of the model card of tcclaviger/ThinkingCap-3.8-27B-PARO5, fetched 2026-09-30, cut down to what autoconfigure reads. -->

</table>
<pre><code>docker run --rm -it \
  --network host --shm-size 32g \
  --ulimit memlock=-1:-1 \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  --device /dev/kfd --device /dev/dri \
  --group-add video \
  -v &lt;path&gt;/ThinkingCap-3.8-27B-PARO5:/app/models \
  -v &lt;path&gt;/cache/triton:/cache/triton \
  -v &lt;path&gt;/cache/vllm:/cache/vllm \
  -v &lt;path&gt;/cache/inductor:/cache/inductor \
  -e TRITON_CACHE_DIR=/cache/triton \
  -e VLLM_CACHE_ROOT=/cache/vllm \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor \
  -e OMP_NUM_THREADS=8 \
  -e VLLM_ROCM_USE_AITER=0 \
  -e GPU_MAX_HW_QUEUES=2 \
  -e HSA_ENABLE_INTERRUPT=1 \
  -e HSA_ENABLE_MWAITX=1 \
  -e ROCR_VISIBLE_DEVICES=0,1,2,3 \
  tcclaviger/vllm:latest \
  /app/models \
  --tensor-parallel-size 4 \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice \
  --max-num-seqs 8 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.92 \
  --host 0.0.0.0 \
  --port 8077 \
  --kv-cache-dtype fp8 \
  --served-model-name ThinkingCap-3.8-27B-PARO5 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --override-generation-config '{"max_tokens": 65536, "temperature": 0.6, "top_p": 0.95, "top_k": 20}' \
  --compilation-config '{"cudagraph_mode": "PIECEWISE", "cudagraph_capture_sizes": [1, 8, 16, 24, 32, 40, 48, 56, 64], "max_cudagraph_capture_size": 64}'
</code></pre>
<p>Speculative decoding with a Qwen3.8-27B DFlash draft works unchanged: add <code>--speculative-config '{"method": "dflash", "model": "/app/draft", "num_speculative_tokens": 7}'</code> with the draft mounted at <code>/app/draft</code>.</p>

## Base model

### vLLM / SGLang

Serve the bf16 model with either engine. The reasoning parser returns the thinking in a separate `reasoning` / `reasoning_content` field instead of inline in `content` before `</think>`, and the tool-call parser turns the model's XML tool calls into structured `tool_calls` — the same flags the base model's serving recipes use. The model's own MTP (multi-token-prediction / NextN) head gives self-speculative decoding with no separate draft model:

```bash
# vLLM — standard
vllm serve bottlecapai/ThinkingCap-Qwen3.8-27B \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml
# vLLM — with MTP self-speculative decoding
vllm serve bottlecapai/ThinkingCap-Qwen3.8-27B \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'

# SGLang — standard
python -m sglang.launch_server --model-path bottlecapai/ThinkingCap-Qwen3.8-27B --trust-remote-code \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder
# SGLang — with MTP self-speculative decoding
python -m sglang.launch_server --model-path bottlecapai/ThinkingCap-Qwen3.8-27B --trust-remote-code \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --speculative-algorithm EAGLE --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```
