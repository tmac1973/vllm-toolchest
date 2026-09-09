package backup

import (
	"errors"
	"strings"
	"testing"

	"github.com/tmac1973/vllm-toolchest/internal/config"
	"github.com/tmac1973/vllm-toolchest/internal/models"
)

// recorder captures what Apply wrote, so a test asserts on effects rather
// than on the report's prose.
type recorder struct {
	settings     *Settings
	env          *RuntimeEnv
	radiance     *config.RadianceConfig
	configs      map[string]models.VLLMConfig
	pending      []MissingModel
	installed    map[string]bool
	failSettings error
	failPending  error
}

func newRecorder(installed ...string) *recorder {
	r := &recorder{configs: map[string]models.VLLMConfig{}, installed: map[string]bool{}}
	for _, id := range installed {
		r.installed[id] = true
	}
	return r
}

func (r *recorder) deps(numGPUs int) Deps {
	return Deps{
		ApplySettings: func(in Settings) ([]string, error) {
			if r.failSettings != nil {
				return nil, r.failSettings
			}
			r.settings = &in
			return []string{"theme"}, nil
		},
		CurrentEnv: func() RuntimeEnv {
			return RuntimeEnv{Curated: map[string]string{"VLLM_LOGGING_LEVEL": "INFO"}, Extra: "EXISTING=1"}
		},
		ApplyEnv:       func(e RuntimeEnv) error { r.env = &e; return nil },
		ApplyRadiance:  func(c config.RadianceConfig) error { r.radiance = &c; return nil },
		InstalledModel: func(id string) bool { return r.installed[id] },
		ApplyModelConfig: func(id string, cfg models.VLLMConfig) error {
			r.configs[id] = cfg
			return nil
		},
		SavePending: func(m MissingModel) error {
			if r.failPending != nil {
				return r.failPending
			}
			r.pending = append(r.pending, m)
			return nil
		},
		NumGPUs: numGPUs,
	}
}

func allSections() Selections {
	return Selections{Settings: true, RuntimeEnv: true, Radiance: true, ModelConfigs: true}
}

// A restore merges; it never removes what the target already had. A variable
// the backup doesn't mention has to survive.
func TestRuntimeEnvMergeNeverDeletes(t *testing.T) {
	r := newRecorder()
	f := &File{Version: Version, RuntimeEnv: &RuntimeEnv{
		Curated: map[string]string{"VLLM_ROCM_USE_AITER": "1"},
	}}
	Apply(f, Selections{RuntimeEnv: true}, r.deps(1))

	if r.env == nil {
		t.Fatal("no environment applied")
	}
	if r.env.Curated["VLLM_LOGGING_LEVEL"] != "INFO" {
		t.Error("a variable the backup did not mention was dropped")
	}
	if r.env.Curated["VLLM_ROCM_USE_AITER"] != "1" {
		t.Error("the backup's variable was not applied")
	}
	// An empty extra block in the backup means "nothing to say", not "clear
	// what's there".
	if r.env.Extra != "EXISTING=1" {
		t.Errorf("an empty extra block cleared the target's; got %q", r.env.Extra)
	}
}

func TestRuntimeEnvExtraIsReplacedWhenPresent(t *testing.T) {
	r := newRecorder()
	f := &File{Version: Version, RuntimeEnv: &RuntimeEnv{Extra: "FROM_BACKUP=1"}}
	Apply(f, Selections{RuntimeEnv: true}, r.deps(1))
	if r.env.Extra != "FROM_BACKUP=1" {
		t.Errorf("got %q", r.env.Extra)
	}
}

// An environment that fails validation is skipped rather than written, or the
// target ends up holding a value its own UI would refuse.
func TestInvalidRuntimeEnvIsSkipped(t *testing.T) {
	r := newRecorder()
	f := &File{Version: Version, RuntimeEnv: &RuntimeEnv{
		Curated: map[string]string{"NOT_A_REAL_VARIABLE": "1"},
	}}
	rep := Apply(f, Selections{RuntimeEnv: true}, r.deps(1))
	if r.env != nil {
		t.Error("an invalid environment was written")
	}
	if len(rep.Skipped) != 1 || rep.Skipped[0].Item != "runtime env" {
		t.Errorf("expected the section to be reported as skipped; got %+v", rep.Skipped)
	}
}

// Tensor parallelism is the one setting that cannot cross machines unchecked:
// too high, and the failure arrives minutes into a model load rather than at
// restore time.
func TestTensorParallelIsClampedToTheGPUCount(t *testing.T) {
	r := newRecorder("a/b")
	f := &File{Version: Version, ModelConfigs: []ModelConfigExport{
		{ModelID: "a/b", Config: models.VLLMConfig{TensorParallelSize: 4}},
	}}
	rep := Apply(f, Selections{ModelConfigs: true}, r.deps(2))

	if got := r.configs["a/b"].TensorParallelSize; got != 2 {
		t.Errorf("tensor parallel size = %d, want 2", got)
	}
	if len(rep.Warnings) != 1 || !strings.Contains(rep.Warnings[0], "reduced to 2") {
		t.Errorf("the change should be reported; got %v", rep.Warnings)
	}
}

// With no GPU reading available, guessing would be worse than leaving it: the
// config is imported as written and the operator can see it.
func TestTensorParallelUntouchedWhenGPUCountUnknown(t *testing.T) {
	r := newRecorder("a/b")
	f := &File{Version: Version, ModelConfigs: []ModelConfigExport{
		{ModelID: "a/b", Config: models.VLLMConfig{TensorParallelSize: 4}},
	}}
	Apply(f, Selections{ModelConfigs: true}, r.deps(0))
	if got := r.configs["a/b"].TensorParallelSize; got != 4 {
		t.Errorf("tensor parallel size = %d, want it left at 4", got)
	}
}

func TestMissingModelsAreHeldPending(t *testing.T) {
	r := newRecorder("here/installed")
	f := &File{Version: Version, ModelConfigs: []ModelConfigExport{
		{ModelID: "here/installed", Config: models.VLLMConfig{MaxModelLen: 1}},
		{ModelID: "not/here", Config: models.VLLMConfig{MaxModelLen: 2}},
	}}
	rep := Apply(f, Selections{ModelConfigs: true}, r.deps(1))

	if rep.AppliedModelConfigs != 1 {
		t.Errorf("applied %d configs, want 1", rep.AppliedModelConfigs)
	}
	if len(rep.Missing) != 1 || rep.Missing[0].ModelID != "not/here" {
		t.Fatalf("expected one missing model; got %+v", rep.Missing)
	}
	if !rep.Missing[0].Pending {
		t.Error("the missing model should have been held as pending")
	}
	if len(r.pending) != 1 || r.pending[0].Config.MaxModelLen != 2 {
		t.Errorf("pending entry wrong: %+v", r.pending)
	}
}

// A pending entry must carry the config already normalized for this machine,
// so a later claim is a plain attach rather than a second round of fixing up
// — by then there is no report to warn into.
func TestPendingConfigIsNormalizedBeforeItIsHeld(t *testing.T) {
	r := newRecorder()
	f := &File{Version: Version, ModelConfigs: []ModelConfigExport{
		{ModelID: "not/here", Config: models.VLLMConfig{TensorParallelSize: 8}},
	}}
	Apply(f, Selections{ModelConfigs: true}, r.deps(2))
	if len(r.pending) != 1 {
		t.Fatalf("expected one pending entry, got %d", len(r.pending))
	}
	if got := r.pending[0].Config.TensorParallelSize; got != 2 {
		t.Errorf("held config kept tensor parallel size %d; want it clamped to 2", got)
	}
}

// Without somewhere to hold it, a missing model's config is a skip that says
// so — never a silent drop.
func TestMissingModelWithoutPendingIsSkipped(t *testing.T) {
	r := newRecorder()
	d := r.deps(1)
	d.SavePending = nil
	f := &File{Version: Version, ModelConfigs: []ModelConfigExport{{ModelID: "not/here"}}}
	rep := Apply(f, Selections{ModelConfigs: true}, d)

	if len(rep.Skipped) != 1 || !strings.Contains(rep.Skipped[0].Reason, "not installed") {
		t.Errorf("expected a skip explaining why; got %+v", rep.Skipped)
	}
	if rep.Missing[0].Pending {
		t.Error("nothing was held, so Pending must be false")
	}
}

func TestPendingSaveFailureIsReported(t *testing.T) {
	r := newRecorder()
	r.failPending = errors.New("disk full")
	f := &File{Version: Version, ModelConfigs: []ModelConfigExport{{ModelID: "not/here"}}}
	rep := Apply(f, Selections{ModelConfigs: true}, r.deps(1))
	if len(rep.Skipped) != 1 || !strings.Contains(rep.Skipped[0].Reason, "disk full") {
		t.Errorf("got %+v", rep.Skipped)
	}
	if rep.Missing[0].Pending {
		t.Error("Pending must stay false when the save failed")
	}
}

// A hand-edited file could carry an empty secret. Applying it would blank a
// working credential, which is the one restore outcome that cannot be undone
// from the file.
func TestEmptySecretsAreIgnoredOnRestore(t *testing.T) {
	r := newRecorder()
	f := &File{Version: Version, Settings: &Settings{HFToken: ptr(""), APIKey: ptr("")}}
	rep := Apply(f, Selections{Settings: true}, r.deps(1))

	if r.settings.HFToken != nil || r.settings.APIKey != nil {
		t.Error("an empty secret reached the target")
	}
	if len(rep.Warnings) != 2 {
		t.Errorf("both should be reported; got %v", rep.Warnings)
	}
}

// Unticking a section is a choice, and the report says so rather than staying
// silent about a section the file did contain.
func TestUnselectedSectionsAreReported(t *testing.T) {
	r := newRecorder()
	f := &File{
		Version:      Version,
		Settings:     &Settings{},
		RuntimeEnv:   &RuntimeEnv{},
		Radiance:     &config.RadianceConfig{},
		ModelConfigs: []ModelConfigExport{{ModelID: "a/b"}},
	}
	rep := Apply(f, Selections{}, r.deps(1))

	joined := strings.Join(rep.NotSelected, ",")
	for _, want := range []string{"settings", "runtime env", "radiance", "model configs"} {
		if !strings.Contains(joined, want) {
			t.Errorf("%q missing from %q", want, joined)
		}
	}
	if r.settings != nil || r.env != nil || r.radiance != nil || len(r.configs) != 0 {
		t.Error("an unselected section was applied anyway")
	}
}

// A section the file doesn't carry isn't "not selected" — there was nothing
// to select. Saying otherwise reads as though something was withheld.
func TestAbsentSectionsAreNotReportedAsUnselected(t *testing.T) {
	r := newRecorder()
	rep := Apply(&File{Version: Version}, Selections{}, r.deps(1))
	if len(rep.NotSelected) != 0 {
		t.Errorf("nothing was in the file, so nothing was skipped by choice; got %v", rep.NotSelected)
	}
}

func TestSettingsFailureIsSkippedNotFatal(t *testing.T) {
	r := newRecorder("a/b")
	r.failSettings = errors.New("config is read-only")
	f := &File{
		Version:      Version,
		Settings:     &Settings{Theme: ptr("dark")},
		ModelConfigs: []ModelConfigExport{{ModelID: "a/b"}},
	}
	rep := Apply(f, allSections(), r.deps(1))

	if len(rep.Skipped) != 1 || !strings.Contains(rep.Skipped[0].Reason, "read-only") {
		t.Errorf("expected the settings failure reported; got %+v", rep.Skipped)
	}
	// The rest of the restore still runs: one bad section is not a reason to
	// abandon the others.
	if rep.AppliedModelConfigs != 1 {
		t.Error("a settings failure stopped the model configs from applying")
	}
}

func TestSelectionsNone(t *testing.T) {
	if !(Selections{}).None() {
		t.Error("an empty selection should report None")
	}
	if (Selections{Radiance: true}).None() {
		t.Error("one ticked section is not None")
	}
}
