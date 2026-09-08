package api

import (
	"fmt"
	"html/template"
	"io"
	"io/fs"
	"log/slog"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
	"github.com/tmac1973/vllm-toolchest/internal/benchmark"
	"github.com/tmac1973/vllm-toolchest/internal/config"
	"github.com/tmac1973/vllm-toolchest/internal/huggingface"
	"github.com/tmac1973/vllm-toolchest/internal/models"
	"github.com/tmac1973/vllm-toolchest/internal/monitor"
	"github.com/tmac1973/vllm-toolchest/internal/process"
	"github.com/tmac1973/vllm-toolchest/internal/tuning"
	"github.com/tmac1973/vllm-toolchest/internal/vllmenv"
	"github.com/tmac1973/vllm-toolchest/web"
)

type Server struct {
	cfg        *config.Config
	pages      map[string]*template.Template
	partials   *template.Template
	router     chi.Router
	monitor    *monitor.Monitor
	hfClient   *huggingface.Client
	downloader *huggingface.Downloader
	registry   *models.Registry
	process    *process.Manager
	bench      *benchmark.Store
	benchSvc   *benchmark.Service
	probe      *probeManager
	tuner      *tuning.Manager
	vllmEnv    vllmenv.Env
	version    string
}

func NewServer(cfg *config.Config) *Server {
	return NewServerWithEnv(cfg, vllmenv.Detect(), "dev")
}

// NewServerWithEnv builds the server against an already-detected vLLM
// environment, so main can log and reuse the same detection. version is the
// build stamp shown under the sidebar brand.
func NewServerWithEnv(cfg *config.Config, env vllmenv.Env, version string) *Server {
	mon := monitor.New(3 * time.Second)
	mon.Start()

	reg := models.NewRegistry(cfg.DataDir)
	dl := huggingface.NewDownloader(cfg.DataDir, cfg.HFToken)
	dl.SetOnComplete(func(downloadID, modelID, modelDir string) {
		reg.RegisterFromDownload(modelID, modelDir)
	})

	s := &Server{
		cfg:        cfg,
		monitor:    mon,
		hfClient:   huggingface.NewClient(cfg.HFToken),
		downloader: dl,
		registry:   reg,
		process:    process.NewManager(cfg.VLLMHost, cfg.VLLMPort),
		vllmEnv:    env,
		version:    version,
	}
	// Radiance's entrypoint takes the same arguments `vllm serve` does, so
	// launching through it keeps its startup banner and topology sweep in the
	// log stream instead of losing them to the bypassed ENTRYPOINT.
	if len(env.Launcher) > 0 {
		s.process.SetLauncher(process.Launcher{Bin: env.Launcher[0], Args: env.Launcher[1:]})
	}
	s.bench = benchmark.NewStore(cfg.DataDir)
	s.benchSvc = benchmark.NewService(s.bench)
	s.benchSvc.SetJobEnv(newJobEnv(s))
	s.probe = newProbeManager(s)
	s.tuner = tuning.NewManager(cfg.DataDir, cfg.DeviceNameSuffix(), env.TunerScript, s.process)
	s.tuner.SetPython(env.Python)
	s.tuner.SetConfigsDir(env.BlockFP8ConfigsDir)

	reg.Maintenance()
	s.initTemplates()
	s.router = s.buildRouter()
	return s
}

func (s *Server) templateFuncs() template.FuncMap {
	return template.FuncMap{
		"divf": func(a, b interface{}) float64 {
			af, bf := toFloat64(a), toFloat64(b)
			if bf == 0 {
				return 0
			}
			return af / bf
		},
		"pctOf": func(value, max float64) float64 {
			if max == 0 {
				return 0
			}
			return (value / max) * 100
		},
		"formatBytes": huggingface.FormatBytes,

		// cssID makes a model ID usable as an element id and as the tail of a
		// querySelector — model IDs carry slashes and dots, which are selector
		// syntax.
		"cssID": safeID,
		// divGB renders a byte count in GiB.
		"divGB": func(bytes int64) float64 { return float64(bytes) / (1024 * 1024 * 1024) },
		// hfModelURL is the HuggingFace page for a model, or "" when the ID is
		// not a linkable owner/name pair — which is how the templates decide
		// whether to render a link at all.
		"hfModelURL": hfModelURL,
		"add":        func(a, b int) int { return a + b },

		// version is the build stamp under the sidebar brand.
		"version": s.versionLabel,
		// themeDefault is the saved theme a browser with no choice of its own
		// starts from.
		"themeDefault": func() string { return s.cfg.Theme },
	}
}

// versionLabel renders the build stamp for display. Released versions
// (e.g. "1.2.3") get a "v" prefix; anything git describe produced for an
// untagged or dirty tree already carries its own marker and reads fine
// without one.
func (s *Server) versionLabel() string {
	v := s.version
	if v == "" || v == "dev" {
		return "dev"
	}
	if strings.HasPrefix(v, "v") || strings.HasPrefix(v, "dev") {
		return v
	}
	if _, err := strconv.Atoi(strings.SplitN(v, ".", 2)[0]); err == nil {
		return "v" + v
	}
	return v
}

// initTemplates parses the layout and partials once, then clones that base per
// page so each page's {{define "content"}} does not collide with the others.
//
// The partials are kept as their own handle as well: fragment handlers render
// them directly, and looking one up by walking the pages map (which is what
// this used to do) picked whichever page Go's map iteration reached first.
func (s *Server) initTemplates() {
	base := template.Must(template.New("").Funcs(s.templateFuncs()).ParseFS(web.Templates,
		"templates/layout.html",
		"templates/partials/*.html",
	))
	s.partials = template.Must(base.Clone())

	pages := map[string]*template.Template{}
	pageFiles := []string{
		"models.html",
		"models_browse.html",
		"server.html",
		"benchmarks.html",
		"visualize.html",
		"tuning.html",
		"settings.html",
	}
	for _, pf := range pageFiles {
		clone := template.Must(base.Clone())
		pages[pf] = template.Must(clone.ParseFS(web.Templates, "templates/"+pf))
	}
	s.pages = pages
}

// hfModelURL returns the HuggingFace page for an owner/name model ID, or ""
// for anything that is not one.
func hfModelURL(modelID string) string {
	parts := strings.Split(modelID, "/")
	if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
		return ""
	}
	return "https://huggingface.co/" + modelID
}

func (s *Server) Router() http.Handler {
	return s.router
}

func (s *Server) buildRouter() chi.Router {
	r := chi.NewRouter()
	r.Use(middleware.Logger)
	r.Use(middleware.Recoverer)
	r.Use(middleware.Compress(5))

	staticFS, _ := fs.Sub(web.Static, "static")
	r.Handle("/static/*", http.StripPrefix("/static/",
		http.FileServer(http.FS(staticFS))))

	// Page routes
	r.Get("/", s.handleIndex)
	r.Get("/models", s.handleModelsPage)
	r.Get("/models/browse", s.handleModelsBrowsePage)
	r.Get("/server", s.handleServerPage)
	r.Get("/benchmarks", s.handleBenchmarksPage)
	r.Get("/benchmarks/visualize", s.handleVisualizePage)
	r.Get("/tuning", s.handleTuningPage)
	r.Get("/settings", s.handleSettingsPage)

	// Health check
	r.Get("/healthz", s.handleHealthCheck)

	// API routes
	r.Route("/api", func(r chi.Router) {
		r.Get("/dashboard", s.handleDashboard)
		r.Get("/gpu-map", s.handleGPUMap)

		r.Route("/models", func(r chi.Router) {
			r.Get("/", s.handleListModels)
			r.Post("/scan", s.handleScanModels)
			r.Get("/get", s.handleGetModel)
			r.Get("/config-panel", s.handleModelConfigPanel)
			r.Put("/activate", s.handleActivateModel)
			r.Put("/config", s.handleUpdateModelConfig)
			r.Delete("/delete", s.handleDeleteModel)
		})
		r.Route("/hf", func(r chi.Router) {
			r.Get("/search", s.handleHFSearch)
			r.Get("/model", s.handleHFModel)
			r.Post("/download", s.handleHFDownload)
			r.Get("/downloads", s.handleHFDownloads)
			r.Delete("/incomplete", s.handleHFDiscardIncomplete)
			r.Get("/download/{id}/progress", s.handleHFDownloadProgress)
			r.Delete("/download/{id}", s.handleHFDownloadCancel)
		})
		r.Route("/service", func(r chi.Router) {
			r.Get("/status", s.handleServiceStatus)
			r.Post("/start", s.handleServiceStart)
			r.Post("/stop", s.handleServiceStop)
			r.Post("/restart", s.handleServiceRestart)
			r.Get("/logs", s.handleServiceLogs)
			r.Delete("/logs", s.handleClearServiceLogs)
			r.Get("/log-stream", s.handleServiceLogStream)
			r.Get("/health", s.handleServiceHealth)
		})
		r.Route("/benchmarks", func(r chi.Router) {
			r.Get("/", s.handleListBenchmarks)
			r.Post("/", s.handleStartBenchmark)
			r.Get("/form", s.handleBenchmarkForm)
			r.Get("/about", s.handleBenchmarksAbout)
			r.Get("/compare", s.handleCompareBenchmarks)
			r.Get("/visualize", s.handleVisualizeData)
			r.Delete("/batch-delete", s.handleBatchDeleteBenchmarks)
			r.Get("/timings", s.handleTimingsList)
			r.Get("/timings/*", s.handleTimingsForModel)
			r.Get("/probe-context/form", s.handleProbeForm)
			r.Post("/probe-context", s.handleStartContextProbe)
			r.Post("/probe-context/apply", s.handleApplyProbe)
			r.Get("/probe-context/{id}/progress", s.handleContextProbeProgress)
			r.Get("/probe-context/result/*", s.handleGetProbeResult)
			r.Get("/{id}", s.handleGetBenchmark)
			r.Delete("/{id}", s.handleDeleteBenchmark)
			r.Post("/{id}/cancel", s.handleCancelBenchmark)
			r.Get("/{id}/progress", s.handleBenchmarkProgress)
		})
		r.Route("/benchmark-jobs", func(r chi.Router) {
			r.Get("/", s.handleListJobs)
			r.Post("/", s.handleCreateJob)
			r.Get("/form", s.handleJobForm)
			r.Get("/{id}", s.handleGetJob)
			r.Get("/{id}/export", s.handleExportJob)
			r.Delete("/{id}", s.handleDeleteJob)
			r.Post("/{id}/cancel", s.handleCancelJob)
			r.Post("/{id}/retry-failed", s.handleRetryFailedCells)
		})
		r.Route("/settings", func(r chi.Router) {
			r.Get("/", s.handleGetSettings)
			r.Put("/", s.handleUpdateSettings)
			r.Post("/test-connection", s.handleTestConnection)
		})
		r.Route("/monitor", func(r chi.Router) {
			r.Get("/", s.handleMonitorStatus)
			r.Get("/stream", s.handleMonitorStream)
		})
		r.Route("/tuning", func(r chi.Router) {
			r.Get("/status", s.handleTuningStatus)
			r.Post("/start", s.handleStartTuning)
			r.Post("/cancel", s.handleCancelTuning)
			r.Get("/logs", s.handleTuningLogs)
			r.Get("/log-stream", s.handleTuningLogStream)
		})
	})

	// OpenAI-compatible proxy
	r.Route("/v1", func(r chi.Router) {
		r.Use(s.apiKeyAuth)
		r.Get("/models", s.handleV1Models)
		r.Handle("/*", s.newProxyHandler())
	})

	return r
}

type pageData struct {
	Title string
	Nav   string
}

// handleIndex redirects to the server page, which is where the dashboard now
// lives — status, controls and logs on one screen rather than a read-only
// summary linking to a separate Service tab.
func (s *Server) handleIndex(w http.ResponseWriter, r *http.Request) {
	http.Redirect(w, r, "/server", http.StatusFound)
}

func (s *Server) handleModelsPage(w http.ResponseWriter, r *http.Request) {
	s.render(w, "models.html", pageData{Title: "Models", Nav: "models"})
}

func (s *Server) handleModelsBrowsePage(w http.ResponseWriter, r *http.Request) {
	s.render(w, "models_browse.html", struct {
		pageData
		QuantFilters []huggingface.QuantFilterOption
	}{
		pageData:     pageData{Title: "Search HuggingFace", Nav: "browse"},
		QuantFilters: huggingface.QuantFilterOptions(),
	})
}

// modelChoice is one entry of the server page's model picker.
type modelChoice struct {
	ID     string
	Name   string
	Active bool
}

func (s *Server) handleServerPage(w http.ResponseWriter, r *http.Request) {
	var choices []modelChoice
	for _, m := range s.registry.List() {
		if m.Orphaned {
			continue
		}
		choices = append(choices, modelChoice{
			ID:     m.ID,
			Name:   displayNameOf(m),
			Active: m.ID == s.cfg.ActiveModel,
		})
	}

	s.render(w, "server.html", struct {
		pageData
		Models []modelChoice
	}{
		pageData: pageData{Title: "Server", Nav: "server"},
		Models:   choices,
	})
}

func (s *Server) handleBenchmarksPage(w http.ResponseWriter, r *http.Request) {
	s.render(w, "benchmarks.html", pageData{Title: "Benchmarks", Nav: "benchmarks"})
}

func (s *Server) handleSettingsPage(w http.ResponseWriter, r *http.Request) {
	c := s.cfg
	data := struct {
		pageData
		ExternalURL         string
		VLLMPort            int
		HasAPIKey           bool
		HasHFToken          bool
		DefaultDtype        string
		GPUMemoryUtil       float64
		MaxNumSeqs          int
		AttentionBackend    string
		EnforceEager        bool
		EnablePrefixCache   bool
		ToolUseEnabled      bool
		DefaultToolParser   string
		PreferMarlin        bool
		DefaultKVCacheDtype string
		AutoRestart         bool
		Theme               string

		Variant           string
		RadianceVersion   string
		IsRadiance        bool
		VLLMDeviceName    string
		VenvRoot          string
		AttentionBackends []backendOption
		Radiance          config.RadianceConfig
	}{
		pageData:            pageData{Title: "Settings", Nav: "settings"},
		ExternalURL:         c.ExternalURL,
		VLLMPort:            c.VLLMPort,
		HasAPIKey:           c.APIKey != "",
		HasHFToken:          c.HFToken != "",
		DefaultDtype:        c.DefaultDtype,
		GPUMemoryUtil:       c.GPUMemoryUtil,
		MaxNumSeqs:          c.MaxNumSeqs,
		AttentionBackend:    c.AttentionBackend,
		EnforceEager:        c.EnforceEager,
		EnablePrefixCache:   c.EnablePrefixCache,
		ToolUseEnabled:      c.ToolUseEnabled,
		DefaultToolParser:   c.DefaultToolParser,
		PreferMarlin:        c.PreferMarlin,
		DefaultKVCacheDtype: c.DefaultKVCacheDtype,
		AutoRestart:         c.AutoRestart,
		Theme:               c.Theme,

		Variant:           s.vllmEnv.Variant,
		RadianceVersion:   s.vllmEnv.RadianceVersion,
		IsRadiance:        s.vllmEnv.IsRadiance(),
		VLLMDeviceName:    s.deviceName(),
		VenvRoot:          s.vllmEnv.VenvRoot,
		AttentionBackends: attentionBackendOptions(s.vllmEnv.IsRadiance()),
		Radiance:          c.Radiance,
	}
	s.render(w, "settings.html", data)
}

func (s *Server) handleHealthCheck(w http.ResponseWriter, r *http.Request) {
	respondJSON(w, map[string]string{
		"status":  "ok",
		"version": "0.1.0",
	})
}

// dashboardGPU is one GPU's line on the dashboard card.
type dashboardGPU struct {
	Name    string
	UsedGB  float64
	TotalGB float64
	// Versions is the driver and ROCm line, already joined, or "" when
	// neither is known.
	Versions string
}

// dashboardTiming is one model's row in the live-activity table.
type dashboardTiming struct {
	ModelID   string
	AvgGenTPS float64
	Count     int
	LastSeen  string
}

func (s *Server) handleDashboard(w http.ResponseWriter, r *http.Request) {
	metrics := s.monitor.Current()

	gpus := make([]dashboardGPU, 0, len(metrics.GPU))
	for _, g := range metrics.GPU {
		var versions string
		switch {
		case g.DriverVersion != "" && g.ROCmVersion != "":
			versions = fmt.Sprintf("Driver: %s \u00b7 ROCm: %s", g.DriverVersion, g.ROCmVersion)
		case g.DriverVersion != "":
			versions = "Driver: " + g.DriverVersion
		case g.ROCmVersion != "":
			versions = "ROCm: " + g.ROCmVersion
		}
		gpus = append(gpus, dashboardGPU{
			Name:     g.Name,
			UsedGB:   float64(g.VRAMUsedMB) / 1024,
			TotalGB:  float64(g.VRAMTotalMB) / 1024,
			Versions: versions,
		})
	}

	// The name clients pass in the "model" field, which is only meaningful
	// while something is actually being served.
	served := ""
	if st := s.process.GetStatus(); st.State == process.StateRunning {
		served = st.ModelID
	}

	respondHTML(w)
	s.renderPartial(w, "dashboard_cards", struct {
		GPUs           []dashboardGPU
		ModelCount     int
		APIURL         string
		ServedModel    string
		ToolUseEnabled bool
	}{
		GPUs:           gpus,
		ModelCount:     len(s.registry.List()),
		APIURL:         strings.TrimRight(s.cfg.ExternalURL, "/") + "/v1",
		ServedModel:    served,
		ToolUseEnabled: s.cfg.ToolUseEnabled,
	})
}

func (s *Server) render(w http.ResponseWriter, name string, data any) {
	w.Header().Set("Cache-Control", "no-cache, no-store, must-revalidate")
	w.Header().Set("Pragma", "no-cache")
	tmpl, ok := s.pages[name]
	if !ok {
		slog.Error("template not found", "name", name)
		http.Error(w, "page not found", http.StatusNotFound)
		return
	}
	respondHTML(w)
	if err := tmpl.ExecuteTemplate(w, "layout", data); err != nil {
		slog.Error("template render error", "name", name, "error", err)
		http.Error(w, "template error", http.StatusInternalServerError)
	}
}

// renderPartial writes one partial to w. Callers are fragment handlers
// responding to htmx, so there is no page around the output to carry an error:
// a failure is logged and left as an HTML comment, which is visible in the
// swapped-in markup without breaking the surrounding page.
func (s *Server) renderPartial(w io.Writer, name string, data any) {
	t := s.partials.Lookup(name)
	if t == nil {
		slog.Error("partial not found", "name", name)
		io.WriteString(w, "<!-- partial not found: "+name+" -->")
		return
	}
	if err := t.Execute(w, data); err != nil {
		// html/template writes what it rendered before the error, so the
		// response is already partly written by this point; the comment marks
		// where it stopped.
		slog.Error("partial render error", "name", name, "error", err)
		io.WriteString(w, "<!-- render error: "+name+" -->")
	}
}

// SetDeviceName updates the GPU device name tuned kernel configs are keyed by,
// once the boot-time probe has resolved what the running vLLM reports.
func (s *Server) SetDeviceName(name string) {
	if s.tuner != nil {
		s.tuner.SetDeviceName(name)
	}
}

// deviceName is the live device name: the tuner holds the probed value, and
// the config's architecture-derived one is the fallback before the probe lands.
func (s *Server) deviceName() string {
	if s.tuner != nil {
		if n := s.tuner.DeviceName(); n != "" {
			return n
		}
	}
	return s.cfg.DeviceNameSuffix()
}

// VLLMEnv exposes the detected image environment to handlers and templates.
func (s *Server) VLLMEnv() vllmenv.Env { return s.vllmEnv }
