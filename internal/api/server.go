package api

import (
	"fmt"
	"html/template"
	"io"
	"io/fs"
	"log/slog"
	"net/http"
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
	"github.com/tmac1973/vllm-toolchest/web"
)

type Server struct {
	cfg        *config.Config
	pages      map[string]*template.Template
	router     chi.Router
	monitor    *monitor.Monitor
	hfClient   *huggingface.Client
	downloader *huggingface.Downloader
	registry   *models.Registry
	process    *process.Manager
	bench      *benchmark.Store
	benchSvc   *benchmark.Service
}

func NewServer(cfg *config.Config) *Server {
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
	}
	s.bench = benchmark.NewStore(cfg.DataDir)
	s.benchSvc = benchmark.NewService(s.bench)

	reg.Maintenance()
	s.pages = s.parseTemplates()
	s.router = s.buildRouter()
	return s
}

func (s *Server) parseTemplates() map[string]*template.Template {
	funcMap := template.FuncMap{
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
	}

	base := template.Must(template.New("").Funcs(funcMap).ParseFS(web.Templates,
		"templates/layout.html",
		"templates/partials/*.html",
	))

	pages := map[string]*template.Template{}
	pageFiles := []string{
		"index.html",
		"models.html",
		"models_browse.html",
		"service.html",
		"benchmarks.html",
		"settings.html",
	}
	for _, pf := range pageFiles {
		clone := template.Must(base.Clone())
		pages[pf] = template.Must(clone.ParseFS(web.Templates, "templates/"+pf))
	}
	return pages
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
	r.Get("/server", s.handleServicePage)
	r.Get("/benchmarks", s.handleBenchmarksPage)
	r.Get("/settings", s.handleSettingsPage)

	// Health check
	r.Get("/healthz", s.handleHealthCheck)

	// API routes
	r.Route("/api", func(r chi.Router) {
		r.Get("/dashboard", s.handleDashboard)

		r.Route("/models", func(r chi.Router) {
			r.Get("/", s.handleListModels)
			r.Post("/scan", s.handleScanModels)
			r.Get("/get", s.handleGetModel)
			r.Get("/config-panel", s.handleModelConfigPanel)
			r.Put("/config", s.handleUpdateModelConfig)
			r.Delete("/delete", s.handleDeleteModel)
		})
		r.Route("/hf", func(r chi.Router) {
			r.Get("/search", s.handleHFSearch)
			r.Get("/model", s.handleHFModel)
			r.Post("/download", s.handleHFDownload)
			r.Get("/downloads", s.handleHFActiveDownloads)
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
			r.Get("/{id}", s.handleGetBenchmark)
			r.Delete("/{id}", s.handleDeleteBenchmark)
			r.Post("/{id}/cancel", s.handleCancelBenchmark)
			r.Get("/{id}/progress", s.handleBenchmarkProgress)
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

func (s *Server) handleIndex(w http.ResponseWriter, r *http.Request) {
	s.render(w, "index.html", pageData{Nav: "home"})
}

func (s *Server) handleModelsPage(w http.ResponseWriter, r *http.Request) {
	s.render(w, "models.html", pageData{Title: "Models", Nav: "models"})
}

func (s *Server) handleModelsBrowsePage(w http.ResponseWriter, r *http.Request) {
	s.render(w, "models_browse.html", pageData{Title: "Search HuggingFace", Nav: "browse"})
}

func (s *Server) handleServicePage(w http.ResponseWriter, r *http.Request) {
	s.render(w, "service.html", pageData{Title: "Service", Nav: "service"})
}

func (s *Server) handleBenchmarksPage(w http.ResponseWriter, r *http.Request) {
	s.render(w, "benchmarks.html", pageData{Title: "Benchmarks", Nav: "benchmarks"})
}

func (s *Server) handleSettingsPage(w http.ResponseWriter, r *http.Request) {
	c := s.cfg
	data := struct {
		pageData
		ExternalURL        string
		VLLMPort           int
		HasAPIKey          bool
		HasHFToken         bool
		DefaultDtype       string
		GPUMemoryUtil      float64
		MaxNumSeqs         int
		AttentionBackend   string
		EnforceEager       bool
		EnablePrefixCache  bool
		ToolUseEnabled     bool
		DefaultToolParser  string
		PreferMarlin       bool
		DefaultKVCacheDtype string
		AutoRestart        bool
		Theme              string
	}{
		pageData:           pageData{Title: "Settings", Nav: "settings"},
		ExternalURL:        c.ExternalURL,
		VLLMPort:           c.VLLMPort,
		HasAPIKey:          c.APIKey != "",
		HasHFToken:         c.HFToken != "",
		DefaultDtype:       c.DefaultDtype,
		GPUMemoryUtil:      c.GPUMemoryUtil,
		MaxNumSeqs:         c.MaxNumSeqs,
		AttentionBackend:   c.AttentionBackend,
		EnforceEager:       c.EnforceEager,
		EnablePrefixCache:  c.EnablePrefixCache,
		ToolUseEnabled:     c.ToolUseEnabled,
		DefaultToolParser:  c.DefaultToolParser,
		PreferMarlin:       c.PreferMarlin,
		DefaultKVCacheDtype: c.DefaultKVCacheDtype,
		AutoRestart:        c.AutoRestart,
		Theme:              c.Theme,
	}
	s.render(w, "settings.html", data)
}

func (s *Server) handleHealthCheck(w http.ResponseWriter, r *http.Request) {
	respondJSON(w, map[string]string{
		"status":  "ok",
		"version": "0.1.0",
	})
}

func (s *Server) handleDashboard(w http.ResponseWriter, r *http.Request) {
	metrics := s.monitor.Current()
	apiURL := strings.TrimRight(s.cfg.ExternalURL, "/") + "/v1"

	// GPU card
	gpuHTML := "<p>No GPU detected</p>"
	if len(metrics.GPU) > 0 {
		var buf strings.Builder
		for _, g := range metrics.GPU {
			buf.WriteString(fmt.Sprintf(`<p><strong>%s</strong></p>`, g.Name))
			buf.WriteString(fmt.Sprintf(`<p>VRAM: %.1f / %.1f GB</p>`,
				float64(g.VRAMUsedMB)/1024, float64(g.VRAMTotalMB)/1024))
			if g.ROCmVersion != "" || g.DriverVersion != "" {
				buf.WriteString("<p>")
				if g.DriverVersion != "" {
					buf.WriteString(fmt.Sprintf("Driver: %s", g.DriverVersion))
				}
				if g.ROCmVersion != "" {
					if g.DriverVersion != "" {
						buf.WriteString(" &middot; ")
					}
					buf.WriteString(fmt.Sprintf("ROCm: %s", g.ROCmVersion))
				}
				buf.WriteString("</p>")
			}
		}
		gpuHTML = buf.String()
	}

	// Tool use indicator
	toolUseLabel := "disabled"
	if s.cfg.ToolUseEnabled {
		toolUseLabel = "<ins>enabled</ins>"
	}

	// Service status
	svcStatus := s.process.GetStatus()
	var svcBadge, svcModel string
	switch svcStatus.State {
	case "running":
		svcBadge = "<ins>Running</ins>"
	case "starting":
		svcBadge = "<mark>Starting...</mark>"
	case "error":
		svcBadge = "<del>Error</del>"
	default:
		svcBadge = "Stopped"
	}
	if svcStatus.ModelID != "" {
		svcModel = fmt.Sprintf(`<p>Model: <strong>%s</strong></p>`, svcStatus.ModelID)
	}

	respondHTML(w)
	fmt.Fprintf(w, `<div class="grid">
    <article>
        <header>vLLM Service</header>
        <p>%s</p>
        %s
        <p><a href="/service">Manage &rarr;</a></p>
    </article>
    <article>
        <header>GPU</header>
        %s
    </article>
    <article>
        <header>Models</header>
        <p><strong>%d</strong> models registered</p>
        <p><a href="/models">Manage &rarr;</a> &middot; <a href="/models/browse">Get New &rarr;</a></p>
    </article>
    <article>
        <header>API Endpoint</header>
        <pre style="user-select: all; cursor: pointer;">%s</pre>
        <p>Tool use: %s</p>
        <p><a href="/settings">Settings &rarr;</a></p>
    </article>
</div>`, svcBadge, svcModel, gpuHTML, len(s.registry.List()), apiURL, toolUseLabel)
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

func (s *Server) renderPartial(w http.ResponseWriter, name string, data any) {
	// Look for the partial in any of the parsed page templates
	for _, tmpl := range s.pages {
		if t := tmpl.Lookup(name); t != nil {
			if err := t.Execute(w, data); err != nil {
				slog.Error("partial render error", "name", name, "error", err)
			}
			return
		}
	}
	slog.Error("partial not found", "name", name)
	io.WriteString(w, "<!-- partial not found: "+name+" -->")
}
