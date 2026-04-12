package monitor

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
)

type rocmBackend struct {
	rocmVersion   string
	driverVersion string
}

func newROCm() GPUBackend {
	if _, err := os.Stat("/dev/kfd"); err != nil {
		return nil
	}
	b := &rocmBackend{
		rocmVersion:   readROCmVersion(),
		driverVersion: readDriverVersion(),
	}
	return b
}

func (r *rocmBackend) Name() string { return "rocm" }

func (r *rocmBackend) Collect() ([]GPUInfo, error) {
	if gpus, err := r.collectROCmSMI(); err == nil && len(gpus) > 0 {
		return gpus, nil
	}
	return r.collectSysfs()
}

func (r *rocmBackend) collectROCmSMI() ([]GPUInfo, error) {
	out, err := exec.Command("rocm-smi",
		"--showuse", "--showmemuse", "--showtemp", "--showpower",
		"--showfan", "--showclocks",
		"--csv").Output()
	if err != nil {
		return nil, fmt.Errorf("rocm-smi: %w", err)
	}

	lines := strings.Split(strings.TrimSpace(string(out)), "\n")
	if len(lines) < 2 {
		return nil, fmt.Errorf("rocm-smi: unexpected output")
	}

	header := strings.Split(lines[0], ",")
	colIdx := make(map[string]int)
	for i, h := range header {
		colIdx[strings.TrimSpace(h)] = i
	}

	var gpus []GPUInfo
	for _, line := range lines[1:] {
		fields := strings.Split(line, ",")
		if len(fields) < 2 {
			continue
		}

		gpu := GPUInfo{}
		if i, ok := colIdx["device"]; ok && i < len(fields) {
			gpu.Index, _ = strconv.Atoi(strings.TrimSpace(fields[i]))
		}
		if i, ok := colIdx["GPU use (%)"]; ok && i < len(fields) {
			gpu.UtilPercent, _ = strconv.Atoi(strings.TrimSpace(fields[i]))
		}
		if i, ok := colIdx["Temperature (Sensor edge) (C)"]; ok && i < len(fields) {
			f, _ := strconv.ParseFloat(strings.TrimSpace(fields[i]), 64)
			gpu.TempC = int(f)
		}
		if i, ok := colIdx["Average Graphics Package Power (W)"]; ok && i < len(fields) {
			gpu.PowerW, _ = strconv.ParseFloat(strings.TrimSpace(fields[i]), 64)
		}
		// Fan speed — column name varies across ROCm versions
		for _, col := range []string{"Fan speed (%)", "Fan Speed (%)", "Fan speed"} {
			if i, ok := colIdx[col]; ok && i < len(fields) {
				f, _ := strconv.ParseFloat(strings.TrimSpace(fields[i]), 64)
				gpu.FanPercent = int(f)
				break
			}
		}
		// GPU clock
		for _, col := range []string{"sclk clock speed (MHz)", "SCLK", "sclk clock speed"} {
			if i, ok := colIdx[col]; ok && i < len(fields) {
				f, _ := strconv.ParseFloat(strings.TrimSpace(fields[i]), 64)
				gpu.ClockMHz = int(f)
				break
			}
		}

		vramUsed, vramTotal := readVRAMSysfs(gpu.Index)
		gpu.VRAMUsedMB = vramUsed
		gpu.VRAMTotalMB = vramTotal
		gpu.Name = readGPUNameSysfs(gpu.Index)
		gpu.ROCmVersion = r.rocmVersion
		gpu.DriverVersion = r.driverVersion

		gpus = append(gpus, gpu)
	}
	return gpus, nil
}

func (r *rocmBackend) collectSysfs() ([]GPUInfo, error) {
	var gpus []GPUInfo

	cards, _ := filepath.Glob("/sys/class/drm/card[0-9]*/device/vendor")
	idx := 0
	for _, vendorFile := range cards {
		vendor, _ := os.ReadFile(vendorFile)
		if strings.TrimSpace(string(vendor)) != "0x1002" {
			continue
		}

		deviceDir := filepath.Dir(vendorFile)
		gpu := GPUInfo{Index: idx}

		if data, err := os.ReadFile(filepath.Join(deviceDir, "gpu_busy_percent")); err == nil {
			gpu.UtilPercent, _ = strconv.Atoi(strings.TrimSpace(string(data)))
		}

		gpu.VRAMUsedMB, gpu.VRAMTotalMB = readVRAMSysfs(idx)

		hwmonDirs, _ := filepath.Glob(filepath.Join(deviceDir, "hwmon", "hwmon*"))
		for _, hwmon := range hwmonDirs {
			if data, err := os.ReadFile(filepath.Join(hwmon, "temp1_input")); err == nil {
				millideg, _ := strconv.Atoi(strings.TrimSpace(string(data)))
				gpu.TempC = millideg / 1000
				break
			}
		}

		for _, hwmon := range hwmonDirs {
			if data, err := os.ReadFile(filepath.Join(hwmon, "power1_average")); err == nil {
				microwatts, _ := strconv.Atoi(strings.TrimSpace(string(data)))
				gpu.PowerW = float64(microwatts) / 1_000_000
				break
			}
		}

		// Fan speed from hwmon (pwm1: 0-255 scale)
		for _, hwmon := range hwmonDirs {
			if data, err := os.ReadFile(filepath.Join(hwmon, "pwm1")); err == nil {
				pwm, _ := strconv.Atoi(strings.TrimSpace(string(data)))
				gpu.FanPercent = pwm * 100 / 255
				break
			}
		}

		// GPU clock from pp_dpm_sclk (active entry marked with *)
		if data, err := os.ReadFile(filepath.Join(deviceDir, "pp_dpm_sclk")); err == nil {
			for _, line := range strings.Split(string(data), "\n") {
				if strings.Contains(line, "*") {
					fields := strings.Fields(line)
					if len(fields) >= 2 {
						clockStr := strings.TrimSuffix(fields[1], "Mhz")
						clockStr = strings.TrimSuffix(clockStr, "MHz")
						gpu.ClockMHz, _ = strconv.Atoi(clockStr)
					}
				}
			}
		}

		gpu.Name = readGPUNameSysfs(idx)
		gpu.ROCmVersion = r.rocmVersion
		gpu.DriverVersion = r.driverVersion
		gpus = append(gpus, gpu)
		idx++
	}

	if len(gpus) == 0 {
		return nil, fmt.Errorf("no AMD GPUs found in sysfs")
	}
	return gpus, nil
}

func readVRAMSysfs(gpuIdx int) (usedMB, totalMB int) {
	cards, _ := filepath.Glob("/sys/class/drm/card[0-9]*/device/vendor")
	idx := 0
	for _, vendorFile := range cards {
		vendor, _ := os.ReadFile(vendorFile)
		if strings.TrimSpace(string(vendor)) != "0x1002" {
			continue
		}
		if idx != gpuIdx {
			idx++
			continue
		}

		deviceDir := filepath.Dir(vendorFile)
		if data, err := os.ReadFile(filepath.Join(deviceDir, "mem_info_vram_used")); err == nil {
			bytes, _ := strconv.ParseInt(strings.TrimSpace(string(data)), 10, 64)
			usedMB = int(bytes / (1024 * 1024))
		}
		if data, err := os.ReadFile(filepath.Join(deviceDir, "mem_info_vram_total")); err == nil {
			bytes, _ := strconv.ParseInt(strings.TrimSpace(string(data)), 10, 64)
			totalMB = int(bytes / (1024 * 1024))
		}
		return
	}
	return 0, 0
}

func readGPUNameSysfs(gpuIdx int) string {
	if out, err := exec.Command("rocminfo").Output(); err == nil {
		var currentAgent int
		for _, line := range strings.Split(string(out), "\n") {
			line = strings.TrimSpace(line)
			if strings.HasPrefix(line, "Marketing Name:") {
				name := strings.TrimSpace(strings.TrimPrefix(line, "Marketing Name:"))
				if name != "" && !strings.HasPrefix(name, "AMD Ryzen") && !strings.HasPrefix(name, "AMD EPYC") {
					if currentAgent == gpuIdx {
						return name
					}
					currentAgent++
				}
			}
		}
	}
	return fmt.Sprintf("AMD GPU %d", gpuIdx)
}

func readROCmVersion() string {
	// Try /opt/rocm/.info/version
	if data, err := os.ReadFile("/opt/rocm/.info/version"); err == nil {
		return strings.TrimSpace(string(data))
	}
	// Try /opt/rocm/include/rocm-core/rocm_version.h or similar
	if data, err := os.ReadFile("/opt/rocm/.info/version-dev"); err == nil {
		return strings.TrimSpace(string(data))
	}
	return ""
}

func readDriverVersion() string {
	// Try /sys/module/amdgpu/version
	if data, err := os.ReadFile("/sys/module/amdgpu/version"); err == nil {
		return strings.TrimSpace(string(data))
	}
	return ""
}
