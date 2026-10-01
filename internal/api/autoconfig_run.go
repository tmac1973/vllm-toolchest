package api

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"time"

	"github.com/tmac1973/vllm-toolchest/internal/autoconfig"
	"github.com/tmac1973/vllm-toolchest/internal/llmcall"
	"github.com/tmac1973/vllm-toolchest/internal/models"
	"github.com/tmac1973/vllm-toolchest/internal/process"
)

// autoconfigTimeout bounds one run: the card, a helper start that may have to
// load and compile, the reading, and giving the engine back.
const autoconfigTimeout = 25 * time.Minute

// autoconfigState holds the one run the server allows at a time, and keeps
// its result until it is saved or discarded.
type autoconfigState struct {
	mu  sync.Mutex
	run *autoconfigRun
	// lastAdvice is the last reading of each model's card this process
	// obtained, so an unsaved or discarded run's reading is reused while
	// the card is unchanged. Persisted nowhere: a saved profile carries its
	// own.
	lastAdvice map[string]storedAdvice
}

type storedAdvice struct {
	CardHash string
	Advice   json.RawMessage
}

type autoconfigRun struct {
	modelID     string
	class       models.ContextClass
	progress    string
	restoreNote string
	done        bool
	result      *autoconfig.Result
	err         error
}

// autoconfigSnapshot is a copy of the current run, if any.
func (s *Server) autoconfigSnapshot() (autoconfigRun, bool) {
	s.autoconf.mu.Lock()
	defer s.autoconf.mu.Unlock()
	if s.autoconf.run == nil {
		return autoconfigRun{}, false
	}
	return *s.autoconf.run, true
}

// clearAutoconfigRun drops a finished run for the model.
func (s *Server) clearAutoconfigRun(id string) {
	s.autoconf.mu.Lock()
	defer s.autoconf.mu.Unlock()
	if r := s.autoconf.run; r != nil && r.modelID == id && r.done {
		s.autoconf.run = nil
	}
}

func (s *Server) setAutoconfigProgress(run *autoconfigRun, line string) {
	s.autoconf.mu.Lock()
	run.progress = line
	s.autoconf.mu.Unlock()
}

// startAutoconfig starts a run for one model in the background.
//
// It does not refuse on a busy engine: whether the helper is needed is known
// only once the card has been fetched and compared with the last reading,
// and when it is needed while the engine is busy, the loan fails with the
// reason and the run completes on the card's command and the machine.
func (s *Server) startAutoconfig(id string, class models.ContextClass, reread bool) error {
	m, ok := s.registry.Get(id)
	switch {
	case !ok:
		return fmt.Errorf("model not found: %s", id)
	case m.Orphaned:
		return errors.New("the model's files are missing")
	case m.IsDraft():
		return errors.New("a draft model is configured through the model it drafts for")
	case m.Helper:
		return errors.New("the helper model has fixed settings")
	}

	s.autoconf.mu.Lock()
	if cur := s.autoconf.run; cur != nil && !cur.done {
		s.autoconf.mu.Unlock()
		return fmt.Errorf("autoconfigure is already running for %s; wait for it to finish", s.displayName(cur.modelID))
	}
	run := &autoconfigRun{modelID: id, class: class, progress: "Starting"}
	s.autoconf.run = run
	s.autoconf.mu.Unlock()

	deps := s.autoconfigDeps(m, run, class, reread)
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), autoconfigTimeout)
		defer cancel()
		res, err := autoconfig.Run(ctx, deps, class)
		if errors.Is(err, context.DeadlineExceeded) {
			err = fmt.Errorf("autoconfigure took longer than %s and was stopped", autoconfigTimeout)
		}
		if err != nil {
			slog.Warn("autoconfigure failed", "model", id, "error", err)
		}

		s.autoconf.mu.Lock()
		defer s.autoconf.mu.Unlock()
		if res != nil {
			if run.restoreNote != "" {
				res.Notes = append(res.Notes, models.ProfileNote{Reason: capitalise(run.restoreNote) + ".", Origin: "default"})
			}
			if res.AdviceFrom == "helper" && len(res.Advice) > 0 {
				if s.autoconf.lastAdvice == nil {
					s.autoconf.lastAdvice = map[string]storedAdvice{}
				}
				s.autoconf.lastAdvice[id] = storedAdvice{res.CardHash, res.Advice}
			}
		}
		run.done, run.result, run.err = true, res, err
	}()
	return nil
}

// autoconfigDeps wires a run to this server.
func (s *Server) autoconfigDeps(m *models.Model, run *autoconfigRun, class models.ContextClass, reread bool) autoconfig.Deps {
	d := autoconfig.Deps{
		Model:      m,
		Base:       m.VLLMConfig,
		Reread:     reread,
		Flags:      s.serveFlags(),
		MachineEnv: s.configuredEnvPairs(nil),
		CardChars:  autoconfig.CardCharsForContext(models.HelperContext),
		Progress:   func(line string) { s.setAutoconfigProgress(run, line) },
		Plan: func(base models.VLLMConfig, kv string) models.FitPlan {
			return models.PlanFit(s.planInput(m, base, class, kv))
		},
		Previous: func(hash string) (json.RawMessage, bool) { return s.previousAdvice(m.ID, hash) },
	}
	if s.hfClient != nil {
		d.Fetcher = s.cardCache()
		d.Hub = s.hfClient
	}
	d.Installed = func(repo string) *models.Model {
		if x, ok := s.registry.Get(repo); ok && !x.Orphaned {
			return x
		}
		return nil
	}
	if desc, known := s.vllmEnv.Descriptor(); known && len(desc.AttentionBackends) > 0 {
		for _, b := range desc.AttentionBackends {
			d.Backends = append(d.Backends, b.Value)
		}
	}
	for _, x := range s.registry.List() {
		if x.IsDraft() && !x.Orphaned {
			d.Drafts = append(d.Drafts, x)
		}
	}
	d.Drafts = append(d.Drafts, models.BundledDrafts(m)...)

	switch helper := s.helperModel(); {
	case helper == nil:
		d.NoHelperWhy = "No helper model is installed, so the card's text was not read; its command still was. Install the helper in Settings to include the rest."
	default:
		if ok, why := models.HelperFits(s.gpuInventory(), s.helperUtil()); !ok {
			d.NoHelperWhy = why + " The card's text was not read; its command still was."
			break
		}
		d.Helper = func(ctx context.Context, name string, schema map[string]any, system, user string, out any) error {
			return s.askHelper(ctx, run, helper, name, schema, system, user, out)
		}
	}
	return d
}

// previousAdvice is an earlier reading of this card, from this process's
// last run or from the model's saved Autoconfig profile, whichever matches
// the card as it is now.
func (s *Server) previousAdvice(modelID, hash string) (json.RawMessage, bool) {
	s.autoconf.mu.Lock()
	last, ok := s.autoconf.lastAdvice[modelID]
	s.autoconf.mu.Unlock()
	if ok && last.CardHash == hash {
		return last.Advice, true
	}
	if p, ok := s.registry.Profile(modelID, models.AutoconfigProfileName); ok &&
		p.Autoconfig != nil && p.Autoconfig.CardHash == hash && len(p.Autoconfig.Advice) > 0 {
		return p.Autoconfig.Advice, true
	}
	return nil, false
}

// askHelper borrows the engine for the helper, asks it, and gives the engine
// back. run is how the loan's progress lines reach the right run.
func (s *Server) askHelper(ctx context.Context, run *autoconfigRun, helper *models.Model,
	schemaName string, schema map[string]any, system, user string, out any) error {
	hc := *helper
	hc.VLLMConfig = models.HelperConfig(s.helperUtil())
	start := hc.VLLMConfig.StartConfig()
	start.ServedModelName = models.HelperServedName

	loan := engineLoan{
		ModelID:   helper.ID,
		ModelPath: process.ResolveModelPath(helper.LocalPath),
		Args:      process.BuildArgs(start),
		Env:       s.launchEnv(&hc),
		StartWait: 10 * time.Minute,
	}
	progress := func(line string) { s.setAutoconfigProgress(run, line) }
	err, restore := s.borrowEngine(ctx, "autoconfigure", loan, progress,
		func(ctx context.Context, baseURL string) error {
			progress("Asking the helper model to read the card")
			return s.llmClient().JSON(ctx, baseURL, models.HelperServedName, schemaName, schema,
				[]llmcall.Message{{Role: "system", Content: system}, {Role: "user", Content: user}}, out)
		})
	if restore != "" {
		s.autoconf.mu.Lock()
		run.restoreNote = restore
		s.autoconf.mu.Unlock()
	}
	return err
}

func (s *Server) llmClient() *llmcall.Client {
	if s.llm == nil {
		s.llm = &llmcall.Client{}
	}
	return s.llm
}

// cardCache fronts the Hub for model cards, so a second run within a few
// minutes does not fetch them again.
func (s *Server) cardCache() autoconfig.Fetcher {
	s.cards.once.Do(func() {
		s.cards.inner = s.hfClient
		s.cards.ttl = 15 * time.Minute
		s.cards.entries = map[string]cardEntry{}
	})
	return &s.cards
}

type cardEntry struct {
	card, base string
	at         time.Time
}

type cachedCards struct {
	once    sync.Once
	mu      sync.Mutex
	inner   autoconfig.Fetcher
	ttl     time.Duration
	entries map[string]cardEntry
}

func (c *cachedCards) get(id string) (cardEntry, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	e, ok := c.entries[id]
	return e, ok && time.Since(e.at) < c.ttl
}

func (c *cachedCards) ModelCard(ctx context.Context, id string) (string, error) {
	if e, ok := c.get("card:" + id); ok {
		return e.card, nil
	}
	card, err := c.inner.ModelCard(ctx, id)
	if err == nil {
		c.mu.Lock()
		c.entries["card:"+id] = cardEntry{card: card, at: time.Now()}
		c.mu.Unlock()
	}
	return card, err
}

func (c *cachedCards) BaseModel(ctx context.Context, id string) string {
	if e, ok := c.get("base:" + id); ok {
		return e.base
	}
	base := c.inner.BaseModel(ctx, id)
	c.mu.Lock()
	c.entries["base:"+id] = cardEntry{base: base, at: time.Now()}
	c.mu.Unlock()
	return base
}

func capitalise(s string) string {
	if s == "" {
		return s
	}
	return strings.ToUpper(s[:1]) + s[1:]
}
