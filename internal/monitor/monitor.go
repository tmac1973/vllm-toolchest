package monitor

import (
	"sync"
	"time"

	"github.com/tmac1973/vllm-toolchest/internal/broadcast"
)

// Metrics holds a snapshot of system resource usage.
type Metrics struct {
	Timestamp time.Time  `json:"timestamp"`
	GPU       []GPUInfo  `json:"gpu,omitempty"`
	CPU       CPUInfo    `json:"cpu"`
	Memory    MemoryInfo `json:"memory"`
}

type GPUInfo struct {
	Index         int     `json:"index"`
	Name          string  `json:"name"`
	UtilPercent   int     `json:"util_percent"`
	VRAMUsedMB    int     `json:"vram_used_mb"`
	VRAMTotalMB   int     `json:"vram_total_mb"`
	TempC         int     `json:"temp_c"`
	PowerW        float64 `json:"power_w,omitempty"`
	FanPercent    int     `json:"fan_percent,omitempty"`
	ClockMHz      int     `json:"clock_mhz,omitempty"`
	DriverVersion string  `json:"driver_version,omitempty"`
	ROCmVersion   string  `json:"rocm_version,omitempty"`
	// Arch is the gfx target when the driver reports one, and IsIGPU marks
	// an integrated GPU: a carve-out of system memory beside the processor,
	// which no model is meant to run on when a discrete card is present.
	Arch   string `json:"arch,omitempty"`
	IsIGPU bool   `json:"is_igpu,omitempty"`
}

type CPUInfo struct {
	UsagePercent float64 `json:"usage_percent"`
	Cores        int     `json:"cores"`
}

type MemoryInfo struct {
	UsedMB  int `json:"used_mb"`
	TotalMB int `json:"total_mb"`
}

// GPUBackend provides GPU-specific metric collection.
type GPUBackend interface {
	Name() string
	Collect() ([]GPUInfo, error)
}

// Monitor polls system metrics at a regular interval.
type Monitor struct {
	gpu      GPUBackend
	interval time.Duration

	mu      sync.RWMutex
	current Metrics
	hub     *broadcast.Hub[Metrics]

	stop chan struct{}
}

func New(interval time.Duration) *Monitor {
	return &Monitor{
		gpu:      detectGPUBackend(),
		interval: interval,
		hub:      broadcast.NewHub[Metrics](4),
		stop:     make(chan struct{}),
	}
}

func (m *Monitor) Start() {
	m.collect()

	go func() {
		ticker := time.NewTicker(m.interval)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				m.collect()
			case <-m.stop:
				return
			}
		}
	}()
}

func (m *Monitor) Stop() {
	close(m.stop)
}

func (m *Monitor) Current() Metrics {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.current
}

func (m *Monitor) Subscribe() chan Metrics {
	return m.hub.Subscribe()
}

func (m *Monitor) Unsubscribe(ch chan Metrics) {
	m.hub.Unsubscribe(ch)
}

func (m *Monitor) collect() {
	metrics := Metrics{
		Timestamp: time.Now(),
		CPU:       collectCPU(),
		Memory:    collectMemory(),
	}

	if m.gpu != nil {
		if gpus, err := m.gpu.Collect(); err == nil {
			metrics.GPU = gpus
		}
	}

	m.mu.Lock()
	m.current = metrics
	m.mu.Unlock()
	m.hub.Send(metrics)
}

func detectGPUBackend() GPUBackend {
	if b := newNVIDIA(); b != nil {
		return b
	}
	if b := newROCm(); b != nil {
		return b
	}
	return nil
}
