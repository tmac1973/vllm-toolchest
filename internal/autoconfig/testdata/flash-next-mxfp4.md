<!-- Excerpt of the model card of tcclaviger/Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ, fetched 2026-09-30, cut down to what autoconfigure reads. -->

<p>AMD RAM offload has been enabled.</p>
<p>FP8 KV cache has been enabled, as well as FP8 KV scale use.</p>
<pre><code>docker run --rm -it \
  --network host --shm-size 32g \
  --ulimit memlock=-1:-1 \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  --device /dev/kfd --device /dev/dri \
  --group-add video \
  -v &lt;path&gt;/Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ:/app/models \
  -v &lt;path&gt;/tunableop:/tunableop \
  -v &lt;path&gt;/cache/triton:/cache/triton \
  -v &lt;path&gt;/cache/vllm:/cache/vllm \
  -v &lt;path&gt;/cache/inductor:/cache/inductor \
  -e TRITON_CACHE_DIR=/cache/triton \
  -e VLLM_CACHE_ROOT=/cache/vllm \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor \
  -e OMP_NUM_THREADS=8 \
  -e VLLM_ROCM_USE_AITER=0 \
  -e GPU_MAX_HW_QUEUES=1 \
  -e HSA_ENABLE_INTERRUPT=1 \
  -e HSA_ENABLE_MWAITX=1 \
  -e ROCR_VISIBLE_DEVICES=0,1,2,3 \
  tcclaviger/vllm:latest \
  /app/models \
  --tensor-parallel-size 4 \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice \
  --max-num-seqs 16 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.97 \
  --host 0.0.0.0 \
  --port 8078 \
  --kv-cache-dtype fp8 \
  --served-model-name Qwen3.8-Flash-Next \
  --max-model-len 524288 \
  --reasoning-parser qwen3 \
  --override-generation-config '{"max_tokens": 65536, "temperature": 0.8, "top_p": 0.95, "top_k": 40, "presence_penalty": 1}' \
  --compilation-config '{"cudagraph_capture_sizes": [4], "max_cudagraph_capture_size": 4}' \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}' \
  --hf-overrides '{"text_config": {"rope_parameters": {"rope_type": "yarn", "factor": 2.0, "original_max_position_embeddings": 262144, "mrope_section": [11, 11, 10], "mrope_interleaved": true, "partial_rotary_factor": 0.25, "rope_theta": 10000000}}}'
</code></pre>

<p><strong>2 x R9700</strong> — the image's best measured TP2 recipe (<code>--recipes tp2-expert-mem-60gb-ple-cache-8gb</code>): routed experts that do not fit VRAM live in 60 GiB of system RAM behind a device-side LRU, the n-gram table streams from NVMe with an 8 GiB row cache; 82 GiB of system RAM at runtime, 100 tok/s single-request decode, 323 tok/s response at 16 concurrent.</p>
<pre><code>docker run --rm -it \
  --network host --shm-size 32g \
  --ulimit memlock=-1:-1 \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  --device /dev/kfd --device /dev/dri \
  --group-add video \
  -v &lt;path&gt;/Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ:/app/models \
  -v &lt;path&gt;/tunableop:/tunableop \
  -v &lt;path&gt;/lru_store:/lru_store \
  -v &lt;nvme path&gt;/plecache:/app/pleoffload \
  -e OMP_NUM_THREADS=8 \
  -e VLLM_ROCM_USE_AITER=0 \
  -e GPU_MAX_HW_QUEUES=1 \
  -e HSA_ENABLE_INTERRUPT=1 \
  -e HSA_ENABLE_MWAITX=1 \
  -e ROCR_VISIBLE_DEVICES=0,1 \
  tcclaviger/vllm:latest \
  /app/models \
  --tensor-parallel-size 2 \
  --enable-expert-offload \
  --expert-offload-mem 60 \
  --ple-nvme-offload \
  --ple-nvme-dir /app/pleoffload \
  --ple-cache-gb 8 \
  --ple-cache-reuse true \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice \
  --max-num-seqs 16 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.95 \
  --host 0.0.0.0 \
  --port 8078 \
  --kv-cache-dtype fp8 \
  --served-model-name Qwen3.8-Flash-Next \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --override-generation-config '{"max_tokens": 65536, "temperature": 0.8, "top_p": 0.95, "top_k": 40, "presence_penalty": 1}' \
  --compilation-config '{"cudagraph_capture_sizes": [4], "max_cudagraph_capture_size": 4}' \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}' \
  --hf-overrides '{"text_config": {"rope_parameters": {"rope_type": "yarn", "factor": 1.0, "original_max_position_embeddings": 262144, "mrope_section": [11, 11, 10], "mrope_interleaved": true, "partial_rotary_factor": 0.25, "rope_theta": 10000000}}}'
</code></pre>

## Processing Ultra-Long Texts

    - Passing command line arguments:

        For vLLM, you can use
        ```shell
        VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve ... --hf-overrides '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}' --max-model-len 1000000  
        ```

