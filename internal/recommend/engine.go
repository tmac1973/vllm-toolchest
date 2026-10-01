package recommend

import (
	"context"
	"slices"
	"sync"
	"time"
)

// staleAfter is how old a pool is served before it says so. It is still
// served: a known-good order a few hours old beats a wait nobody asked for.
const staleAfter = 6 * time.Hour

// Engine builds and holds the scored pool.
type Engine struct {
	hub     Hub
	dataDir string
	now     func() time.Time

	mu   sync.Mutex
	pool *pool
}

type pool struct {
	key         string
	profile     ProfileView
	generatedAt time.Time
	candidates  []Candidate
	orders      map[string][]int
	unverified  []int
	unavailable string
}

// Result is one aim's view of the pool.
type Result struct {
	Profile     ProfileView `json:"profile"`
	Intent      string      `json:"intent"`
	GeneratedAt time.Time   `json:"generated_at"`
	Stale       bool        `json:"stale"`
	Unavailable string      `json:"unavailable"`
	Verified    []Candidate `json:"verified"`
	Unverified  []Candidate `json:"unverified"`
}

// NewEngine is an engine asking hub, caching configs under dataDir.
func NewEngine(hub Hub, dataDir string) *Engine {
	return &Engine{hub: hub, dataDir: dataDir, now: time.Now}
}

// Result builds the pool for p, or reuses one built for the same machine,
// and returns it ordered for intent. An unknown intent is quality.
func (e *Engine) Result(ctx context.Context, p Profile, intent string) Result {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.pool == nil || e.pool.key != p.Key() {
		e.pool = e.build(ctx, p)
	}
	return e.view(intent)
}

// Refresh discards the pool and builds it again, ordered for quality.
func (e *Engine) Refresh(ctx context.Context, p Profile) Result {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.pool = e.build(ctx, p)
	return e.view(IntentQuality)
}

// Recommended reports a repository the current pool verified: a download of
// it is seeded at completion. Looked up then rather than carried on the
// request, because a download outlives the request that started it. A pool
// rebuilt since, without it, means no seed -- a lesser outcome, not an error.
func (e *Engine) Recommended(modelID string) bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.pool == nil {
		return false
	}
	for _, i := range e.pool.orders[IntentQuality] {
		if e.pool.candidates[i].ID == modelID {
			return true
		}
	}
	return false
}

func (e *Engine) view(intent string) Result {
	if !slices.Contains(intents, intent) {
		intent = IntentQuality
	}
	p := e.pool
	r := Result{
		Profile: p.profile, Intent: intent, GeneratedAt: p.generatedAt,
		Stale:       p.unavailable == "" && e.now().Sub(p.generatedAt) > staleAfter,
		Unavailable: p.unavailable,
		Verified:    []Candidate{}, Unverified: []Candidate{},
	}
	for _, i := range p.orders[intent] {
		r.Verified = append(r.Verified, p.candidates[i])
	}
	for _, i := range p.unverified {
		r.Unverified = append(r.Unverified, p.candidates[i])
	}
	return r
}

// build makes a pool from nothing. A failure is a pool with nothing in it and
// the reason, served like any other, so the page renders one state.
func (e *Engine) build(ctx context.Context, p Profile) *pool {
	pl := &pool{key: p.Key(), profile: p.View(), generatedAt: e.now(), orders: map[string][]int{}}
	if !p.Inventory.Known || p.Inventory.Count < 1 || p.Inventory.PerCardGB <= 0 {
		pl.unavailable = "the GPUs have not been read yet"
		return pl
	}
	if e.hub == nil {
		pl.unavailable = "Hugging Face is not reachable from here"
		return pl
	}
	raw, err := fetchPool(ctx, e.hub, p)
	if err != nil {
		pl.unavailable = "Hugging Face could not be reached: " + err.Error()
		return pl
	}
	var cands []Candidate
	for _, r := range coarseFilter(raw, p) {
		if servable(r) {
			cands = append(cands, toCandidate(r, p))
		}
	}
	cands = coarseRank(cands)
	evaluate(ctx, e.hub, e.dataDir, p, cands)

	var verified []int
	for i, c := range cands {
		switch c.Verdict {
		case Verified:
			verified = append(verified, i)
		case Unverified:
			pl.unverified = append(pl.unverified, i)
		}
	}
	pl.candidates = cands
	pl.orders = orders(cands, verified, p.Inventory.Count)
	return pl
}
