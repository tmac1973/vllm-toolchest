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

AUTH=""
if [[ -n "$API_KEY" ]]; then
    AUTH="-H \"Authorization: Bearer $API_KEY\""
fi

echo "=== vLLM Inference Test ==="
echo "URL:    ${BASE_URL}/v1"
echo "Prompt: ${PROMPT}"
echo ""

# Check if vLLM is running
echo "--- Checking health ---"
STATUS=$(curl -sf "${BASE_URL}/api/service/health" 2>/dev/null || echo '{"healthy":false}')
HEALTHY=$(echo "$STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('healthy',False))" 2>/dev/null || echo "False")

if [[ "$HEALTHY" != "True" ]]; then
    echo "vLLM is not running. Start a model from the Server page first."
    echo "Status: $STATUS"
    exit 1
fi
echo "vLLM is healthy."

# Get available models
echo ""
echo "--- Available models ---"
curl -sf "${BASE_URL}/v1/models" ${API_KEY:+-H "Authorization: Bearer $API_KEY"} 2>/dev/null \
    | python3 -c "import sys,json; [print(f'  {m[\"id\"]}') for m in json.load(sys.stdin).get('data',[])]" 2>/dev/null \
    || echo "  (could not list models)"

# Non-streaming request
if [[ "$STREAM" == "false" ]]; then
    echo ""
    echo "--- Chat completion (non-streaming) ---"
    RESPONSE=$(curl -sf "${BASE_URL}/v1/chat/completions" \
        ${API_KEY:+-H "Authorization: Bearer $API_KEY"} \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"auto\",
            \"messages\": [{\"role\": \"user\", \"content\": \"$PROMPT\"}],
            \"max_tokens\": 512,
            \"temperature\": 0.7
        }" 2>&1)

    if echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'])" 2>/dev/null; then
        echo ""
        echo "--- Usage ---"
        echo "$RESPONSE" | python3 -c "
import sys,json
d=json.load(sys.stdin)
u=d.get('usage',{})
print(f'  Prompt tokens:     {u.get(\"prompt_tokens\",\"?\")}')
print(f'  Completion tokens: {u.get(\"completion_tokens\",\"?\")}')
print(f'  Total tokens:      {u.get(\"total_tokens\",\"?\")}')
" 2>/dev/null
    else
        echo "Error response:"
        echo "$RESPONSE"
    fi
else
    # Streaming request
    echo ""
    echo "--- Chat completion (streaming) ---"
    curl -sfN "${BASE_URL}/v1/chat/completions" \
        ${API_KEY:+-H "Authorization: Bearer $API_KEY"} \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"auto\",
            \"messages\": [{\"role\": \"user\", \"content\": \"$PROMPT\"}],
            \"max_tokens\": 512,
            \"temperature\": 0.7,
            \"stream\": true
        }" 2>&1 | while IFS= read -r line; do
            # Strip "data: " prefix
            line="${line#data: }"
            [[ -z "$line" || "$line" == "[DONE]" ]] && continue
            # Extract content delta
            DELTA=$(echo "$line" | python3 -c "
import sys,json
try:
    d=json.load(sys.stdin)
    c=d.get('choices',[{}])[0].get('delta',{}).get('content','')
    if c: print(c, end='', flush=True)
except: pass
" 2>/dev/null || true)
            [[ -n "$DELTA" ]] && printf '%s' "$DELTA"
        done
    echo ""
fi

echo ""
echo "--- Tool use test ---"
TOOL_RESPONSE=$(curl -sf "${BASE_URL}/v1/chat/completions" \
    ${API_KEY:+-H "Authorization: Bearer $API_KEY"} \
    -H "Content-Type: application/json" \
    -d '{
        "model": "auto",
        "messages": [{"role": "user", "content": "What is the weather in San Francisco?"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"}
                    },
                    "required": ["location"]
                }
            }
        }],
        "tool_choice": "auto",
        "max_tokens": 512
    }' 2>&1)

if echo "$TOOL_RESPONSE" | python3 -c "
import sys,json
d=json.load(sys.stdin)
msg=d['choices'][0]['message']
if msg.get('tool_calls'):
    for tc in msg['tool_calls']:
        print(f'  Tool call: {tc[\"function\"][\"name\"]}({tc[\"function\"][\"arguments\"]})')
elif msg.get('content'):
    print(f'  No tool call, got text: {msg[\"content\"][:100]}')
" 2>/dev/null; then
    :
else
    echo "  Error: $TOOL_RESPONSE"
fi

echo ""
echo "=== Done ==="
