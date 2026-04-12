#!/usr/bin/env bash
# Quick inference test against the vLLM API proxy
#
# Usage:
#   ./scripts/test-inference.sh                    # default prompt
#   ./scripts/test-inference.sh "Your prompt here"  # custom prompt
#   ./scripts/test-inference.sh -s                  # streaming mode

set -euo pipefail

BASE_URL="${VLLMCTL_URL:-http://localhost:3000}"
API_KEY="${VLLMCTL_API_KEY:-}"
STREAM=false
PROMPT="Explain what vLLM is in 2-3 sentences."

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--stream) STREAM=true; shift ;;
        -u|--url) BASE_URL="$2"; shift 2 ;;
        -k|--key) API_KEY="$2"; shift 2 ;;
        *) PROMPT="$1"; shift ;;
    esac
done

AUTH_HEADER=""
if [[ -n "$API_KEY" ]]; then
    AUTH_HEADER="-H Authorization: Bearer $API_KEY"
fi

echo "=== vLLM Inference Test ==="
echo "URL: ${BASE_URL}/v1"
echo ""

# Check health
echo "--- Checking health ---"
HEALTH=$(curl -sf "${BASE_URL}/api/service/health" 2>/dev/null || echo '{}')
HEALTHY=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin).get('healthy',False))" 2>/dev/null || echo "False")

if [[ "$HEALTHY" != "True" ]]; then
    echo "vLLM is not running. Start a model from the Server page first."
    exit 1
fi
echo "vLLM is healthy."

# Get the actual model name that vLLM is serving (the full path)
# vLLM uses the model path as the served model name, not the HF repo ID
VLLM_PORT="${VLLMCTL_INFERENCE_PORT:-8000}"
MODEL=$(curl -sf "${BASE_URL%%:*}://localhost:${VLLM_PORT}/v1/models" 2>/dev/null \
    | python3 -c "import sys,json; d=json.load(sys.stdin).get('data',[]); print(d[0]['id'] if d else '')" 2>/dev/null || echo "")

if [[ -z "$MODEL" ]]; then
    # Try through the proxy (port 3000) - vLLM's models endpoint
    # Our proxy passes /v1/* to vLLM except /v1/models which we override
    # So query the service status instead
    MODEL_ID=$(curl -sf "${BASE_URL}/api/service/health" 2>/dev/null \
        | python3 -c "import sys,json; print(json.load(sys.stdin).get('model',''))" 2>/dev/null || echo "")
    if [[ -n "$MODEL_ID" ]]; then
        MODEL="/data/models/${MODEL_ID}"
    fi
fi

if [[ -z "$MODEL" ]]; then
    echo "Could not determine running model."
    exit 1
fi

echo "Model: $MODEL"
echo "Prompt: $PROMPT"
echo ""

# Non-streaming request
if [[ "$STREAM" == "false" ]]; then
    echo "--- Chat completion ---"
    RESPONSE=$(curl -sf "${BASE_URL}/v1/chat/completions" \
        ${API_KEY:+-H "Authorization: Bearer $API_KEY"} \
        -H "Content-Type: application/json" \
        -d "$(python3 -c "
import json
print(json.dumps({
    'model': '$MODEL',
    'messages': [{'role': 'user', 'content': '''$PROMPT'''}],
    'max_tokens': 512,
    'temperature': 0.7
}))
")" 2>&1)

    if echo "$RESPONSE" | python3 -c "
import sys,json
d=json.load(sys.stdin)
print(d['choices'][0]['message']['content'])
print()
print('--- Usage ---')
u=d.get('usage',{})
print(f'  Prompt tokens:     {u.get(\"prompt_tokens\",\"?\")}')
print(f'  Completion tokens: {u.get(\"completion_tokens\",\"?\")}')
print(f'  Total tokens:      {u.get(\"total_tokens\",\"?\")}')
" 2>/dev/null; then
        :
    else
        echo "Error:"
        echo "$RESPONSE"
    fi
else
    # Streaming request
    echo "--- Chat completion (streaming) ---"
    curl -sfN "${BASE_URL}/v1/chat/completions" \
        ${API_KEY:+-H "Authorization: Bearer $API_KEY"} \
        -H "Content-Type: application/json" \
        -d "$(python3 -c "
import json
print(json.dumps({
    'model': '$MODEL',
    'messages': [{'role': 'user', 'content': '''$PROMPT'''}],
    'max_tokens': 512,
    'temperature': 0.7,
    'stream': True
}))
")" 2>&1 | while IFS= read -r line; do
            line="${line#data: }"
            [[ -z "$line" || "$line" == "[DONE]" ]] && continue
            echo "$line" | python3 -c "
import sys,json
try:
    d=json.load(sys.stdin)
    c=d.get('choices',[{}])[0].get('delta',{}).get('content','')
    if c: print(c, end='', flush=True)
except: pass
" 2>/dev/null || true
        done
    echo ""
fi

echo ""
echo "--- Tool use test ---"
TOOL_RESPONSE=$(curl -sf "${BASE_URL}/v1/chat/completions" \
    ${API_KEY:+-H "Authorization: Bearer $API_KEY"} \
    -H "Content-Type: application/json" \
    -d "$(python3 -c "
import json
print(json.dumps({
    'model': '$MODEL',
    'messages': [{'role': 'user', 'content': 'What is the weather in San Francisco?'}],
    'tools': [{
        'type': 'function',
        'function': {
            'name': 'get_weather',
            'description': 'Get the current weather for a location',
            'parameters': {
                'type': 'object',
                'properties': {
                    'location': {'type': 'string', 'description': 'City name'}
                },
                'required': ['location']
            }
        }
    }],
    'tool_choice': 'auto',
    'max_tokens': 512
}))
")" 2>&1)

echo "$TOOL_RESPONSE" | python3 -c "
import sys,json
try:
    d=json.load(sys.stdin)
    msg=d['choices'][0]['message']
    if msg.get('tool_calls'):
        for tc in msg['tool_calls']:
            print(f'  Tool call: {tc[\"function\"][\"name\"]}({tc[\"function\"][\"arguments\"]})')
    elif msg.get('content'):
        print(f'  Response: {msg[\"content\"][:200]}')
    else:
        print('  No tool call or content in response')
except Exception as e:
    print(f'  Error parsing: {e}')
    print(f'  Raw: {sys.stdin.read()[:200]}')
" 2>/dev/null || echo "  Error: $TOOL_RESPONSE"

echo ""
echo "=== Done ==="
