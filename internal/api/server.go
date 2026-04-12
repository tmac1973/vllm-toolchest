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
	"github.com/tmac1973/vllm-toolchest/internal/config"
	"github.com/tmac1973/vllm-toolchest/internal/monitor"
	"github.com/tmac1973/vllm-toolchest/web"
)

type Server struct {
	cfg     *config.Config
	pages   map[string]*template.Template
	router  chi.Router
	monitor *monitor.Monitor
}

func NewServer(cfg *config.Config) *Server {
	mon := monitor.New(3 * time.Second)
	mon.Start()

	s := &Server{
		cfg:     cfg,
		monitor: mon,
	}
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
	r.Get("/service", s.handleServicePage)
	r.Get("/benchmarks", s.handleBenchmarksPage)
	r.Get("/settings", s.handleSettingsPage)

	// Health check
	r.Get("/healthz", s.handleHealthCheck)

	// API routes
	r.Route("/api", func(r chi.Router) {
		r.Get("/dashboard", s.handleDashboard)

		r.Route("/models", func(r chi.Router) {
			// Phase 4
		})
		r.Route("/hf", func(r chi.Router) {
			// Phase 3
		})
		r.Route("/service", func(r chi.Router) {
			// Phase 5
		})
		r.Route("/benchmarks", func(r chi.Router) {
			// Phase 6
		})
		r.Route("/settings", func(r chi.Router) {
			// Phase 7
		})
		r.Route("/monitor", func(r chi.Router) {
			r.Get("/", s.handleMonitorStatus)
			r.Get("/stream", s.handleMonitorStream)
		})
	})

	// OpenAI-compatible proxy (Phase 5)
	r.Route("/v1", func(r chi.Router) {
		r.Use(s.apiKeyAuth)
		// Will proxy to vLLM's /v1 endpoints
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
	proxyEndpoint := strings.TrimRight(s.cfg.ExternalURL, "/") + "/v1"
	data := struct {
		pageData
		ProxyEndpoint string
		VLLMPort      int
		HasAPIKey     bool
		HasHFToken    bool
		HasExtURL     bool
		ExternalURL   string
		DataDir       string
	}{
		pageData:      pageData{Title: "Settings", Nav: "settings"},
		ProxyEndpoint: proxyEndpoint,
		VLLMPort:      s.cfg.VLLMPort,
		HasAPIKey:     s.cfg.APIKey != "",
		HasHFToken:    s.cfg.HFToken != "",
		HasExtURL:     s.cfg.ExternalURL != "",
		ExternalURL:   s.cfg.ExternalURL,
		DataDir:       s.cfg.DataDir,
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

	gpuInfo := "No GPU detected"
	if len(metrics.GPU) > 0 {
		g := metrics.GPU[0]
		gpuInfo = fmt.Sprintf("%s (%.0fGB)", g.Name, float64(g.VRAMTotalMB)/1024)
		if len(metrics.GPU) > 1 {
			gpuInfo += fmt.Sprintf(" x%d", len(metrics.GPU))
		}
	}

	respondHTML(w)
	fmt.Fprintf(w, `<div class="grid">
    <article>
        <header>Service</header>
        <p>vLLM: Stopped</p>
        <p><a href="/service">Manage &rarr;</a></p>
    </article>
    <article>
        <header>GPU</header>
        <p>%s</p>
    </article>
    <article>
        <header>Models</header>
        <p><strong>0</strong> models registered</p>
        <p><a href="/models/browse">Download Models &rarr;</a></p>
    </article>
    <article>
        <header>API Endpoint</header>
        <pre style="user-select: all; cursor: pointer;">%s</pre>
        <p><a href="/settings">Settings &rarr;</a></p>
    </article>
</div>`, gpuInfo, apiURL)
}

func (s *Server) render(w http.ResponseWriter, name string, data any) {
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
