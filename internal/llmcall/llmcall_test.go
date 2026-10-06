package llmcall

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

type answer struct {
	Temperature *float64 `json:"temperature"`
}

// engine answers each request with the next of replies: a status and a body.
type engine struct {
	mu      sync.Mutex
	replies [][2]string
	bodies  []map[string]any
}

func (e *engine) server(t *testing.T) string {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			http.NotFound(w, r)
			return
		}
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		json.Unmarshal(raw, &body)
		e.mu.Lock()
		e.bodies = append(e.bodies, body)
		reply := e.replies[0]
		if len(e.replies) > 1 {
			e.replies = e.replies[1:]
		}
		e.mu.Unlock()
		code := http.StatusOK
		if reply[0] == "400" {
			code = http.StatusBadRequest
		}
		w.WriteHeader(code)
		w.Write([]byte(reply[1]))
	}))
	t.Cleanup(srv.Close)
	return srv.URL
}

func chat(content, finish string) string {
	b, _ := json.Marshal(map[string]any{"choices": []any{map[string]any{
		"message": map[string]any{"content": content}, "finish_reason": finish}}})
	return string(b)
}

var schema = map[string]any{"type": "object"}

func ask(t *testing.T, e *engine) (answer, error) {
	var out answer
	err := (&Client{}).JSON(context.Background(), Endpoint{BaseURL: e.server(t)}, "helper", "form", schema,
		[]Message{{Role: "user", Content: "read this"}}, &out)
	return out, err
}

func TestJSONDecodes(t *testing.T) {
	e := &engine{replies: [][2]string{{"200", chat(`{"temperature": 0.7}`, "stop")}}}
	out, err := ask(t, e)
	if err != nil || out.Temperature == nil || *out.Temperature != 0.7 {
		t.Fatalf("out=%+v err=%v", out, err)
	}
	b := e.bodies[0]
	if b["temperature"] != float64(0) || b["model"] != "helper" {
		t.Errorf("request: %v", b)
	}
	rf, _ := b["response_format"].(map[string]any)
	if rf["type"] != "json_schema" {
		t.Errorf("response_format = %v", rf)
	}
}

func TestJSONFallsBackForAnOldEngine(t *testing.T) {
	e := &engine{replies: [][2]string{
		{"400", `{"error": "unknown field response_format.json_schema"}`},
		{"200", chat(`{"temperature": 0.6}`, "stop")},
	}}
	out, err := ask(t, e)
	if err != nil || *out.Temperature != 0.6 {
		t.Fatalf("out=%+v err=%v", out, err)
	}
	if _, has := e.bodies[1]["guided_json"]; !has {
		t.Error("the retry did not use guided_json")
	}
	if _, has := e.bodies[1]["response_format"]; has {
		t.Error("the retry still sent response_format")
	}
}

func TestJSONAsksOnceMoreForBadJSON(t *testing.T) {
	e := &engine{replies: [][2]string{
		{"200", chat("Sure! Here it is", "stop")},
		{"200", chat("```json\n{\"temperature\": 1}\n```", "stop")},
	}}
	if out, err := ask(t, e); err != nil || *out.Temperature != 1 {
		t.Fatalf("out=%+v err=%v", out, err)
	}
	e = &engine{replies: [][2]string{{"200", chat("no", "stop")}}}
	if _, err := ask(t, e); err == nil || len(e.bodies) != 2 {
		t.Errorf("err=%v after %d requests; want an error after exactly one retry", err, len(e.bodies))
	}
}

func TestJSONErrors(t *testing.T) {
	e := &engine{replies: [][2]string{{"200", chat(`{"temperature": `, "length")}}}
	if _, err := ask(t, e); err == nil || !strings.Contains(err.Error(), "cut off") {
		t.Errorf("a truncated answer: %v", err)
	}
	e = &engine{replies: [][2]string{{"400", `{"error": "model not found"}`}}}
	if _, err := ask(t, e); err == nil || !strings.Contains(err.Error(), "HTTP 400") {
		t.Errorf("an unrelated 400: %v", err)
	}

	slow := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(600 * time.Millisecond)
	}))
	defer slow.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	start := time.Now()
	var out answer
	if err := (&Client{}).JSON(ctx, Endpoint{BaseURL: slow.URL}, "m", "f", schema, nil, &out); err == nil || time.Since(start) > time.Second {
		t.Errorf("a cancelled call took %s and returned %v", time.Since(start), err)
	}
}
