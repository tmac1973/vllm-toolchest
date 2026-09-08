package monitor

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
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
		"--showbus", "--showuse", "--showmemuse", "--showtemp", "--showpower",
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

	// rocm-smi's "device" column ("card0", "card1", …) is not a usable
	// identity. Depending on version and machine it is either the DRM card
	// number (driver probe order, shifted by any display device) or rocm-smi's
	// own row number, sorted by PCI bus address — which on some boards is the
	// reverse of KFD order.
	//
	// Reading it as the GPU index is what put all four R9700s at index 0, each
	// then reading GPU 0's VRAM out of sysfs: the sidebar showed "GPU0" four
	// times with one card's figures repeated. The PCI bus address is the only
	// unambiguous key both sides share, so rows are matched to KFD positions by
	// it; without that column the whole collection is rejected and the sysfs
	// fallback — consistent by construction — takes over.
	busCol, ok := colIdx["PCI Bus"]
	if !ok {
		return nil, fmt.Errorf("rocm-smi: no PCI Bus column")
	}

	// KFD-ordered device dirs: position N is the GPU vLLM addresses as N.
	dirs := listAMDGPUDirs()
	byBDF := kfdIndexByBDF(dirs)
	seen := make(map[int]bool)

	var gpus []GPUInfo
	for _, line := range lines[1:] {
		fields := strings.Split(line, ",")
		if len(fields) < 2 {
			continue
		}

		if busCol >= len(fields) {
			return nil, fmt.Errorf("rocm-smi: row without PCI Bus field")
		}
		bdf := strings.ToLower(strings.TrimSpace(fields[busCol]))
		idx, ok := byBDF[bdf]
		if !ok || seen[idx] {
			// A device KFD does not know, or two rows claiming one GPU: the
			// mapping is unreliable, so let sysfs take over rather than render
			// numbers against the wrong card.
			return nil, fmt.Errorf("rocm-smi: device %s not uniquely in KFD topology", bdf)
		}
		seen[idx] = true

		gpu := GPUInfo{Index: idx}
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

		// VRAM comes from sysfs rather than the CSV, read from the directory
		// this row was matched to.
		gpu.VRAMUsedMB, gpu.VRAMTotalMB = readVRAMFromDir(dirs[idx])
		gpu.Name = readGPUNameSysfs(idx)
		gpu.ROCmVersion = r.rocmVersion
		gpu.DriverVersion = r.driverVersion

		gpus = append(gpus, gpu)
	}

	// Rows arrive in rocm-smi's PCI-address order; present them by GPU index
	// so the sidebar reads GPU0, GPU1, …
	sort.Slice(gpus, func(i, j int) bool { return gpus[i].Index < gpus[j].Index })
	return gpus, nil
}

func (r *rocmBackend) collectSysfs() ([]GPUInfo, error) {
	var gpus []GPUInfo

	for idx, deviceDir := range listAMDGPUDirs() {
		gpu := GPUInfo{Index: idx}

		if data, err := os.ReadFile(filepath.Join(deviceDir, "gpu_busy_percent")); err == nil {
			gpu.UtilPercent, _ = strconv.Atoi(strings.TrimSpace(string(data)))
		}

		gpu.VRAMUsedMB, gpu.VRAMTotalMB = readVRAMFromDir(deviceDir)

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
	}

	if len(gpus) == 0 {
		return nil, fmt.Errorf("no AMD GPUs found in sysfs")
	}
	return gpus, nil
}

// listAMDGPUDirs returns the sysfs device directories of AMD GPUs in KFD
// topology order — the order rocminfo and vLLM's HIP runtime enumerate in, and
// therefore the order the tensor-parallel ranks map onto.
//
// rocm-smi does not share that order: its rows are sorted by PCI bus address,
// which is why collectROCmSMI matches rows by that address rather than by
// position. DRM card numbers do not share it either — they follow driver probe
// order, so a display device ahead of the accelerators shifts every index.
//
// The card glob remains as a fallback for kernels without KFD topology.
func listAMDGPUDirs() []string {
	if dirs := listAMDGPUDirsKFD(); len(dirs) > 0 {
		return dirs
	}
	cards, _ := filepath.Glob("/sys/class/drm/card[0-9]*/device/vendor")
	var dirs []string
	for _, vendorFile := range cards {
		vendor, _ := os.ReadFile(vendorFile)
		if strings.TrimSpace(string(vendor)) != "0x1002" {
			continue
		}
		dirs = append(dirs, filepath.Dir(vendorFile))
	}
	return dirs
}

// listAMDGPUDirsKFD enumerates GPU nodes from /sys/class/kfd, mapping each
// node's drm_render_minor to its device directory. CPU agents carry
// gfx_target_version 0 and are skipped.
func listAMDGPUDirsKFD() []string {
	nodes, _ := filepath.Glob("/sys/class/kfd/kfd/topology/nodes/*/properties")
	// Glob order is lexical ("10" before "2"); sort by numeric node id.
	sort.Slice(nodes, func(i, j int) bool {
		return kfdNodeID(nodes[i]) < kfdNodeID(nodes[j])
	})
	var dirs []string
	for _, propsPath := range nodes {
		data, err := os.ReadFile(propsPath)
		if err != nil {
			continue
		}
		gfx, minor := 0, -1
		for _, line := range strings.Split(string(data), "\n") {
			fields := strings.Fields(line)
			if len(fields) != 2 {
				continue
			}
			switch fields[0] {
			case "gfx_target_version":
				gfx, _ = strconv.Atoi(fields[1])
			case "drm_render_minor":
				minor, _ = strconv.Atoi(fields[1])
			}
		}
		if gfx == 0 || minor <= 0 {
			continue
		}
		dir := fmt.Sprintf("/sys/class/drm/renderD%d/device", minor)
		if vendor, err := os.ReadFile(filepath.Join(dir, "vendor")); err == nil &&
			strings.TrimSpace(string(vendor)) == "0x1002" {
			dirs = append(dirs, dir)
		}
	}
	return dirs
}

// kfdNodeID extracts the numeric node id from a topology properties path.
func kfdNodeID(propsPath string) int {
	id, _ := strconv.Atoi(filepath.Base(filepath.Dir(propsPath)))
	return id
}

// kfdIndexByBDF maps each device's PCI bus address to its position in dirs,
// which is the GPU index. The addresses are lowercased because rocm-smi
// reports them in a different case than sysfs names them.
func kfdIndexByBDF(dirs []string) map[string]int {
	m := make(map[string]int, len(dirs))
	for i, d := range dirs {
		resolved, err := filepath.EvalSymlinks(d)
		if err != nil {
			continue
		}
		m[strings.ToLower(filepath.Base(resolved))] = i
	}
	return m
}

// readVRAMFromDir reads mem_info_vram_* from an already-resolved device
// directory.
//
// It replaced a readVRAMSysfs(index) that re-globbed /sys/class/drm and counted
// AMD cards to find the index'th one — which meant every caller had to agree
// with that glob's ordering, and none of them did.
func readVRAMFromDir(deviceDir string) (usedMB, totalMB int) {
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

func readGPUNameSysfs(gpuIdx int) string {
	if out, err := exec.Command("rocminfo").Output(); err == nil {
		names := parseROCmGPUNames(string(out))
		if gpuIdx >= 0 && gpuIdx < len(names) {
			return names[gpuIdx]
		}
	}
	return fmt.Sprintf("AMD GPU %d", gpuIdx)
}

// parseROCmGPUNames extracts the marketing names of GPU agents from rocminfo
// output, in agent order. rocminfo lists every HSA agent — including the host
// CPU — under "Agent N" blocks, each carrying a "Device Type:" (CPU or GPU)
// and a "Marketing Name:". Only GPU agents are returned.
//
// This replaced a blacklist of marketing names starting with "AMD Ryzen" or
// "AMD EPYC", which was how the CPU agent got skipped. Any other CPU leaked
// through and was labelled GPU 0 — an Intel host would show its Xeon as the
// first card. Keying off "Device Type: GPU" does not care who made the CPU.
//
// When a GPU agent reports no marketing name, its "Name:" (e.g. "gfx1201") is
// used instead so the entry is never blank.
func parseROCmGPUNames(out string) []string {
	type agent struct {
		name      string
		marketing string
		isGPU     bool
	}
	var agents []agent
	cur := -1
	for _, line := range strings.Split(out, "\n") {
		line = strings.TrimSpace(line)
		switch {
		case strings.HasPrefix(line, "Agent "):
			agents = append(agents, agent{})
			cur = len(agents) - 1
		case cur < 0:
			// Header lines before the first agent block.
			continue
		case strings.HasPrefix(line, "Marketing Name:"):
			agents[cur].marketing = strings.TrimSpace(strings.TrimPrefix(line, "Marketing Name:"))
		case strings.HasPrefix(line, "Name:"):
			// Only the first "Name:" of a block is the agent's; the later ones
			// belong to nested pool and cache entries.
			if agents[cur].name == "" {
				agents[cur].name = strings.TrimSpace(strings.TrimPrefix(line, "Name:"))
			}
		case strings.HasPrefix(line, "Device Type:"):
			if strings.Contains(line, "GPU") {
				agents[cur].isGPU = true
			}
		}
	}

	var names []string
	for _, a := range agents {
		if !a.isGPU {
			continue
		}
		if a.marketing != "" {
			names = append(names, a.marketing)
		} else {
			names = append(names, a.name)
		}
	}
	return names
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
