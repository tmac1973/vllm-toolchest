package variants_test

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// There are two ways to start this container and they are described in two
// unrelated places: the compose files, and the Quadlet unit setup.sh writes
// for auto-start. Compose runs on install; the unit runs after a reboot. When
// they disagree, the machine works all day and comes up wrong in the morning,
// which is the hardest version of this bug to see.
//
// It has already bitten three times: ulimits missing from the unit (f7029ac),
// ShmSize set alongside --ipc=host so podman refused to start at all, and the
// models bind mount reaching only the compose path. So check the two against
// each other.

// quadletVendor ties a manifest vendor -- the name in the compose filename --
// to the compute stack GPU_VENDOR names inside setup.sh.
var quadletVendor = map[string]string{
	"amd":    "rocm",
	"nvidia": "cuda",
	"intel":  "xpu",
}

// A variant of each vendor, to give generate_quadlet a manifest to read
// capabilities from.
var quadletVariant = map[string]string{
	"amd":    "radiance",
	"nvidia": "cuda",
	"intel":  "xpu",
}

// composeService is the subset of a compose file this test compares against.
type composeService struct {
	Ports    []string `yaml:"ports"`
	Volumes  []string `yaml:"volumes"`
	Devices  []string `yaml:"devices"`
	GroupAdd []string `yaml:"group_add"`
	IPC      string   `yaml:"ipc"`
	ShmSize  string   `yaml:"shm_size"`
	CapAdd   []string `yaml:"cap_add"`
	Ulimits  map[string]struct {
		Soft int `yaml:"soft"`
		Hard int `yaml:"hard"`
	} `yaml:"ulimits"`
	EnvFile []struct {
		Path     string `yaml:"path"`
		Required bool   `yaml:"required"`
	} `yaml:"env_file"`
}

func readComposeService(t *testing.T, path string) composeService {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("reading compose file: %v", err)
	}
	var doc struct {
		Services map[string]composeService `yaml:"services"`
	}
	if err := yaml.Unmarshal(data, &doc); err != nil {
		t.Fatalf("parsing %s: %v", filepath.Base(path), err)
	}
	svc, ok := doc.Services["vllm-toolchest"]
	if !ok {
		t.Fatalf("%s has no vllm-toolchest service", filepath.Base(path))
	}
	return svc
}

// generateQuadlet runs setup.sh's own unit writer for one vendor. CONTAINER_CMD
// is set to a command that fails so get_volume_name falls back to its default
// rather than inspecting whatever happens to be on the machine running the
// test.
func generateQuadlet(t *testing.T, vendor, modelsDir string) string {
	t.Helper()
	return runSetupSh(t, strings.Join([]string{
		"CONTAINER_CMD=false",
		"GPU_VENDOR=" + quadletVendor[vendor],
		"BUILD_VARIANT=" + quadletVariant[vendor],
		"HOST_VIDEO_GID=44",
		"HOST_RENDER_GID=105",
		"VLLMCTL_MODELS_DIR=" + modelsDir,
		"generate_quadlet",
	}, "\n")+"\n")
}

// quadletKeys collects the unit's [Container] settings as key -> values. Unit
// comments carry the reasoning and would otherwise read as settings.
func quadletKeys(unit string) map[string][]string {
	keys := map[string][]string{}
	for _, line := range strings.Split(unit, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") || strings.HasPrefix(line, "[") {
			continue
		}
		k, v, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		keys[k] = append(keys[k], v)
	}
	return keys
}

func has(values []string, want string) bool {
	for _, v := range values {
		if v == want {
			return true
		}
	}
	return false
}

// composeFields splits a compose short-syntax mapping ("source:target:opts")
// into its fields. A ${VAR:-default} substitution is one field: the ":-" and
// the "./" inside it are not separators.
func composeFields(s string) []string {
	s = strings.Trim(s, `"`)
	var out []string
	depth, start := 0, 0
	for i := 0; i < len(s); i++ {
		switch {
		case strings.HasPrefix(s[i:], "${"):
			depth++
			i++
		case s[i] == '}' && depth > 0:
			depth--
		case s[i] == ':' && depth == 0:
			out = append(out, s[start:i])
			start = i + 1
		}
	}
	return append(out, s[start:])
}

// podman refuses --ipc=host and --shm-size together:
//
//	Error: invalid config provided: cannot set shmsize when running in the
//	{host } IPC Namespace
//
// and exits 125 before the container ever runs, which systemd turns into a
// restart loop that ends in about a second. The compose files legitimately
// carry both -- podman-compose drops ipc:host, so shm_size is the only thing
// that lifts /dev/shm off the 64 MB default there -- but Quadlet applies
// ipc:host for real, so the unit must not name a shm size at all.
func TestQuadletNeverSizesShmUnderHostIPC(t *testing.T) {
	for vendor := range quadletVendor {
		t.Run(vendor, func(t *testing.T) {
			keys := quadletKeys(generateQuadlet(t, vendor, ""))

			hostIPC := false
			for _, args := range keys["PodmanArgs"] {
				if strings.Contains(args, "--ipc=host") {
					hostIPC = true
				}
				if strings.Contains(args, "--shm-size") {
					t.Error("PodmanArgs sets --shm-size; podman rejects it under host IPC")
				}
			}
			if !hostIPC {
				t.Error("no --ipc=host; the compose files give the container the host's " +
					"/dev/shm and the unit has to as well")
			}
			if got, ok := keys["ShmSize"]; ok {
				t.Errorf("ShmSize=%v alongside --ipc=host: podman exits 125 and systemd "+
					"restart-loops the unit on every boot", got)
			}
		})
	}
}

// docker-compose.models.yml bind-mounts the host models directory on the
// compose path. A unit without the same mount means a container started by
// systemd after a reboot sees no models while the same install started through
// compose sees all of them.
func TestQuadletMountsTheModelsDir(t *testing.T) {
	// The mount point is compose's to define; read it rather than repeat it.
	models := readComposeService(t, filepath.Join(repoRoot(t), "docker-compose.models.yml"))
	var target string
	for _, v := range models.Volumes {
		if f := composeFields(v); strings.Contains(f[0], "VLLMCTL_MODELS_DIR") {
			target = strings.Join(f[1:], ":")
			break
		}
	}
	if target == "" {
		t.Fatal("docker-compose.models.yml no longer mounts VLLMCTL_MODELS_DIR")
	}

	const dir = "/srv/llm-models"
	for vendor := range quadletVendor {
		t.Run(vendor, func(t *testing.T) {
			with := quadletKeys(generateQuadlet(t, vendor, dir))
			if want := dir + ":" + target; !has(with["Volume"], want) {
				t.Errorf("models dir configured but the unit has no Volume=%s\n got: %v",
					want, with["Volume"])
			}

			// And no phantom mount when none is configured -- podman would
			// create the source directory and the operator would never be
			// told where their models went.
			without := quadletKeys(generateQuadlet(t, vendor, ""))
			for _, v := range without["Volume"] {
				if strings.Contains(v, "/data/models") {
					t.Errorf("no models dir configured, but the unit mounts %q", v)
				}
			}
		})
	}
}

// Everything else compose grants, the unit has to grant too.
func TestQuadletMatchesCompose(t *testing.T) {
	for vendor := range quadletVendor {
		t.Run(vendor, func(t *testing.T) {
			svc := readComposeService(t, filepath.Join(repoRoot(t), "docker-compose."+vendor+".yml"))
			keys := quadletKeys(generateQuadlet(t, vendor, ""))

			// Ports. Compose writes them with ${VAR:-default} substitutions;
			// the unit carries the resolved values, so compare the container
			// side, which is the half that cannot move.
			for _, p := range svc.Ports {
				containerPort := composeFields(p)[1]
				found := false
				for _, pub := range keys["PublishPort"] {
					if strings.HasSuffix(pub, ":"+containerPort) {
						found = true
					}
				}
				if !found {
					t.Errorf("compose publishes %s; the unit publishes %v", p, keys["PublishPort"])
				}
			}

			// Named volumes. Bind mounts driven by a variable are the models
			// directory's business, checked above.
			for _, v := range svc.Volumes {
				if strings.Contains(v, "${") {
					continue
				}
				if !has(keys["Volume"], v) {
					t.Errorf("compose mounts %s; the unit mounts %v", v, keys["Volume"])
				}
			}

			// Devices. NVIDIA is the exception: compose reserves the GPU
			// through deploy.resources, and the unit names the CDI device.
			for _, d := range svc.Devices {
				host := composeFields(d)[0]
				if !has(keys["AddDevice"], host) {
					t.Errorf("compose grants device %s; the unit grants %v", host, keys["AddDevice"])
				}
			}
			if vendor == "nvidia" && len(keys["AddDevice"]) == 0 {
				t.Error("the unit grants no GPU device; compose reserves one through deploy.resources")
			}

			// Supplementary groups. Compose substitutes the detected GIDs and
			// so does the unit; both were given the same ones here.
			for _, g := range svc.GroupAdd {
				if strings.Contains(g, "${") {
					continue
				}
				if !has(keys["GroupAdd"], g) {
					t.Errorf("compose adds group %s; the unit adds %v", g, keys["GroupAdd"])
				}
			}

			// Capabilities. Compose carries the union across a vendor's
			// variants because it cannot be conditional; the unit is written
			// per install and takes the selected variant's set. So the unit's
			// caps must be a subset, never something compose never grants.
			composeCaps := map[string]bool{}
			for _, c := range svc.CapAdd {
				composeCaps[c] = true
			}
			for _, c := range keys["AddCapability"] {
				if !composeCaps[c] {
					t.Errorf("the unit adds capability %s, which docker-compose.%s.yml does not grant",
						c, vendor)
				}
			}

			// Ulimits.
			for name, lim := range svc.Ulimits {
				want := name + "=" + strconv.Itoa(lim.Soft) + ":" + strconv.Itoa(lim.Hard)
				if !has(keys["Ulimit"], want) {
					t.Errorf("compose sets ulimit %s; the unit sets %v", want, keys["Ulimit"])
				}
			}

			// Host IPC.
			if svc.IPC == "host" {
				hostIPC := false
				for _, args := range keys["PodmanArgs"] {
					if strings.Contains(args, "--ipc=host") {
						hostIPC = true
					}
				}
				if !hostIPC {
					t.Error("compose sets ipc:host; the unit does not")
				}
			}

			// .env. Compose loads it wholesale, which is how the feature
			// knobs, the GPU selection and HF_TOKEN reach the container. The
			// unit emits EnvironmentFile only when the file exists, mirroring
			// compose's required:false -- podman fails a unit outright on a
			// missing --env-file.
			if len(svc.EnvFile) > 0 {
				withEnvFile(t)
				keys := quadletKeys(generateQuadlet(t, vendor, ""))
				if len(keys["EnvironmentFile"]) == 0 {
					t.Error("compose loads .env through env_file; the unit loads nothing, so " +
						"every knob, HIP_VISIBLE_DEVICES and HF_TOKEN are missing after a reboot")
				}
			}
		})
	}
}

// withEnvFile makes sure there is a .env to find. A checkout has none until
// the first install writes one, and the unit deliberately omits
// EnvironmentFile when the file is absent -- podman fails a unit outright on a
// missing --env-file -- so without this the assertion would pass vacuously on
// a clean tree. An existing .env is left exactly as it is.
func withEnvFile(t *testing.T) {
	t.Helper()
	path := filepath.Join(repoRoot(t), ".env")
	if _, err := os.Stat(path); err == nil {
		return
	}
	if err := os.WriteFile(path, []byte("# written by TestQuadletMatchesCompose\n"), 0o600); err != nil {
		t.Skipf("cannot create %s: %v", path, err)
	}
	t.Cleanup(func() { os.Remove(path) })
}
