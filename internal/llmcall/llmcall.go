// Package llmcall asks an OpenAI-compatible engine for an answer constrained
// to a JSON schema, and decodes it.
package llmcall

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// Message is one chat message.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Client makes the calls. The zero value is ready to use.
type Client struct {
	// HTTP is the client used; nil means one with a five-minute timeout,
	// which is generous for a 4B model answering a short form.
	HTTP *http.Client
}

func (c *Client) http() *http.Client {
	if c != nil && c.HTTP != nil {
		return c.HTTP
	}
	return &http.Client{Timeout: 5 * time.Minute}
}

// JSON asks the engine at baseURL for an answer constrained to schema and
// decodes it into out.
//
// Two recoveries are tried, once each. An engine too old for response_format's
// json_schema, which answers 400 naming it, is asked again with the older
// guided_json spelling -- two of the images here pin vLLM 0.23. And an answer
// that does not decode is asked for once more, with the instruction to answer
// with the JSON object only.
func (c *Client) JSON(ctx context.Context, baseURL, model, schemaName string,
	schema map[string]any, msgs []Message, out any) error {
	body := map[string]any{
		"model":       model,
		"messages":    msgs,
		"temperature": 0,
		"max_tokens":  2048,
		"stream":      false,
		"response_format": map[string]any{
			"type": "json_schema",
			"json_schema": map[string]any{
				"name": schemaName, "schema": schema, "strict": true,
			},
		},
	}

	content, err := c.complete(ctx, baseURL, body)
	var old *oldEngineError
	if errors.As(err, &old) {
		delete(body, "response_format")
		body["guided_json"] = schema
		content, err = c.complete(ctx, baseURL, body)
	}
	if err != nil {
		return err
	}
	if decode(content, out) == nil {
		return nil
	}

	retry := append(append([]Message{}, msgs...),
		Message{Role: "assistant", Content: content},
		Message{Role: "user", Content: "Answer again with only the JSON object."})
	body["messages"] = retry
	if content, err = c.complete(ctx, baseURL, body); err != nil {
		return err
	}
	if err := decode(content, out); err != nil {
		return fmt.Errorf("the answer was not the JSON asked for: %w", err)
	}
	return nil
}

// oldEngineError is a 400 that names response_format or json_schema: an
// engine that predates it.
type oldEngineError struct{ body string }

func (e *oldEngineError) Error() string { return "the engine does not accept json_schema: " + e.body }

func (c *Client) complete(ctx context.Context, baseURL string, body map[string]any) (string, error) {
	data, err := json.Marshal(body)
	if err != nil {
		return "", err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		strings.TrimRight(baseURL, "/")+"/v1/chat/completions", bytes.NewReader(data))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.http().Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))

	if resp.StatusCode == http.StatusBadRequest {
		lower := strings.ToLower(string(raw))
		if _, has := body["response_format"]; has &&
			(strings.Contains(lower, "response_format") || strings.Contains(lower, "json_schema")) {
			return "", &oldEngineError{clip(string(raw))}
		}
	}
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("the engine answered HTTP %d: %s", resp.StatusCode, clip(string(raw)))
	}

	var parsed struct {
		Choices []struct {
			Message      struct{ Content string } `json:"message"`
			FinishReason string                   `json:"finish_reason"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(raw, &parsed); err != nil || len(parsed.Choices) == 0 {
		return "", fmt.Errorf("the engine's answer could not be read: %s", clip(string(raw)))
	}
	ch := parsed.Choices[0]
	if ch.FinishReason == "length" {
		return "", errors.New("the answer was cut off")
	}
	return ch.Message.Content, nil
}

// decode reads the answer into out. A model sometimes wraps JSON in a code
// fence even when constrained; the fence is not part of the answer.
func decode(content string, out any) error {
	c := strings.TrimSpace(content)
	c = strings.TrimPrefix(c, "```json")
	c = strings.TrimPrefix(c, "```")
	c = strings.TrimSuffix(c, "```")
	return json.Unmarshal([]byte(strings.TrimSpace(c)), out)
}

func clip(s string) string {
	s = strings.TrimSpace(s)
	if len(s) > 300 {
		return s[:300]
	}
	return s
}
