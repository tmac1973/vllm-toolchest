package monitor

import (
	"fmt"
	"os/exec"
	"strconv"
	"strings"
)

type nvidiaBackend struct{}

func newNVIDIA() GPUBackend {
	if _, err := exec.LookPath("nvidia-smi"); err != nil {
		return nil
	}
	if err := exec.Command("nvidia-smi", "--query-gpu=name", "--format=csv,noheader").Run(); err != nil {
		return nil
	}
	return &nvidiaBackend{}
}

func (n *nvidiaBackend) Name() string { return "nvidia" }

func (n *nvidiaBackend) Collect() ([]GPUInfo, error) {
	out, err := exec.Command("nvidia-smi",
		"--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
		"--format=csv,noheader,nounits").Output()
	if err != nil {
		return nil, fmt.Errorf("nvidia-smi: %w", err)
	}

	var gpus []GPUInfo
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		// Seven columns were asked for. The name is the only free text, and
		// can itself hold ", ", so the numbers are read from the ends and the
		// name is whatever lies between them. A short row is skipped rather
		// than indexed past its end, which used to panic the server.
		fields := strings.Split(line, ", ")
		if len(fields) < 7 {
			continue
		}
		n := len(fields)
		name := strings.Join(fields[1:n-5], ", ")

		idx, _ := strconv.Atoi(strings.TrimSpace(fields[0]))
		util, _ := strconv.Atoi(strings.TrimSpace(fields[n-5]))
		vramUsed, _ := strconv.Atoi(strings.TrimSpace(fields[n-4]))
		vramTotal, _ := strconv.Atoi(strings.TrimSpace(fields[n-3]))
		temp, _ := strconv.Atoi(strings.TrimSpace(fields[n-2]))
		power, _ := strconv.ParseFloat(strings.TrimSpace(fields[n-1]), 64)

		gpus = append(gpus, GPUInfo{
			Index:       idx,
			Name:        strings.TrimSpace(name),
			UtilPercent: util,
			VRAMUsedMB:  vramUsed,
			VRAMTotalMB: vramTotal,
			TempC:       temp,
			PowerW:      power,
		})
	}
	return gpus, nil
}
