package variants

import (
	"regexp"
	"strings"
	"testing"
)

// A variant that offers its base image's releases lists them from the
// registry, newest first, and builds on the one chosen. The registry is
// stood in for by redefining registry_get, the one function that touches the
// network.

// dockerHubPage is the shape of one page of Docker Hub's tag listing, cut
// down to the fields list_base_tags reads. Each tag also carries an "images"
// array in the real thing; it holds no "name" key, which is what lets a plain
// grep pick the tag names out.
func dockerHubPage(next bool, names ...string) string {
	var b strings.Builder
	b.WriteString(`{"count":99,"next":`)
	if next {
		b.WriteString(`"https://hub.docker.com/v2/repositories/x/y/tags?page=2"`)
	} else {
		b.WriteString(`null`)
	}
	b.WriteString(`,"results":[`)
	for i, n := range names {
		if i > 0 {
			b.WriteString(",")
		}
		b.WriteString(`{"images":[{"architecture":"amd64","digest":"sha256:00"}],"last_updater_username":"someone","name":"` + n + `","tag_status":"active"}`)
	}
	b.WriteString("]}")
	return b.String()
}

// fakeRegistry defines registry_get to answer from the pages given, keyed by
// a substring of the URL, and to fail for anything else.
func fakeRegistry(pages map[string]string) string {
	var b strings.Builder
	b.WriteString("registry_get() {\n  case \"$*\" in\n")
	for key, body := range pages {
		b.WriteString("    *'" + key + "'*) cat <<'JSON'\n" + body + "\nJSON\n ;;\n")
	}
	b.WriteString("    *) return 1 ;;\n  esac\n}\n")
	return b.String()
}

// Docker Hub's order is push order, which is the order that matters:
// tcclaviger/vllm pushed 29.08.10 two weeks before 29.08.9, and :latest
// points at 29.08.9. A version sort would put the wrong one first.
func TestDockerHubTagsKeepPushOrder(t *testing.T) {
	reg := fakeRegistry(map[string]string{
		"page=1&": dockerHubPage(true, "29.08.9", "latest", "dev", "29.08.10", "exp"),
		"page=2&": dockerHubPage(false, "29.08.1", "29.07.16", "29.08.9"),
	})
	out := sourceSetupSh(t, reg+`list_base_tags docker.io/tcclaviger/vllm '^[0-9]+\.[0-9]+\.[0-9]+$'`+"\n")

	got := strings.Fields(out)
	want := []string{"29.08.9", "29.08.10", "29.08.1", "29.07.16"}
	if strings.Join(got, " ") != strings.Join(want, " ") {
		t.Errorf("tags = %v, want %v (push order, filtered, each once)", got, want)
	}
}

// ghcr lists names only, so they are sorted as versions.
func TestGHCRTagsSortAsVersions(t *testing.T) {
	reg := fakeRegistry(map[string]string{
		"ghcr.io/token": `{"token":"abc"}`,
		"tags/list":     `{"name":"timothystewart6/vllm-gb10","tags":["v0.28.0-gb10.2","latest","v0.32.0-gb10.0","v0.31.0-gb10.1","v0.31.0-gb10.0","sha-1234"]}`,
	})
	out := sourceSetupSh(t, reg+`list_base_tags ghcr.io/timothystewart6/vllm-gb10 '^v[0-9]+\.[0-9]+\.[0-9]+-gb10\.[0-9]+$'`+"\n")

	got := strings.Fields(out)
	want := []string{"v0.32.0-gb10.0", "v0.31.0-gb10.1", "v0.31.0-gb10.0", "v0.28.0-gb10.2"}
	if strings.Join(got, " ") != strings.Join(want, " ") {
		t.Errorf("tags = %v, want %v", got, want)
	}
}

// pull, and an install with no terminal, take the newest without asking.
func TestNewestIsTakenWithoutAsking(t *testing.T) {
	reg := fakeRegistry(map[string]string{
		"page=1&": dockerHubPage(false, "nightly", "v0.32.0", "v0.32.0-cu129", "v0.31.0"),
	})
	out := sourceSetupSh(t, reg+"BUILD_VARIANT=cuda\nprompt_base_tag </dev/null\necho \"chosen=$CHOSEN_BASE_REF\"\n")
	if !strings.Contains(out, "chosen=docker.io/vllm/vllm-openai:v0.32.0\n") {
		t.Errorf("did not choose the newest release:\n%s", out)
	}
}

// Offline, the manifest's own base image is what gets built.
func TestAnUnreachableRegistryFallsBackToTheManifest(t *testing.T) {
	d, ok := Get("cuda")
	if !ok {
		t.Skip("cuda manifest missing")
	}
	out := sourceSetupSh(t, fakeRegistry(nil)+"BUILD_VARIANT=cuda\nprompt_base_tag newest\necho \"chosen=$CHOSEN_BASE_REF\"\nvariant_base_ref\n")
	if !strings.Contains(out, "chosen=\n") {
		t.Errorf("chose something with no registry to ask:\n%s", out)
	}
	if !strings.Contains(out, "Could not list the releases") {
		t.Errorf("did not say the registry could not be asked:\n%s", out)
	}
	if !strings.HasSuffix(strings.TrimSpace(out), d.BaseImage) {
		t.Errorf("base = %q, want the manifest's %q", out, d.BaseImage)
	}
}

// The operator's own image outranks the list, and nothing is asked.
func TestAnOverrideSkipsTheList(t *testing.T) {
	out := sourceSetupSh(t, "registry_get() { echo CALLED >&2; return 1; }\n"+
		"BUILD_VARIANT=cuda\nVLLMCTL_BASE_IMAGE=docker.io/me/mine:1 prompt_base_tag newest\necho \"chosen=$CHOSEN_BASE_REF\"\n")
	if strings.Contains(out, "CALLED") || !strings.Contains(out, "chosen=\n") {
		t.Errorf("asked the registry despite an override:\n%s", out)
	}
}

// The release chosen is built on, recorded, kept by quick, and stepped back
// from by choosing an older one -- with the newer still on disk.
func TestAChosenReleaseIsRecordedAndKept(t *testing.T) {
	b := newTrackBed(t)
	const newer = "docker.io/tcclaviger/vllm:29.06.1"
	const older = "docker.io/tcclaviger/vllm:28.04.9"
	b.point(newer, newBase)
	b.point(older, oldBase)

	b.run(nil, "CHOSEN_BASE_REF="+newer+"\nensure_base_image refresh\nwrite_env_file\n")
	wantEnv(t, b.env(), map[string]string{
		"VLLMCTL_BASE_IMAGE": newAlias,
		"VLLMCTL_BASE_TAG":   newer,
		"VLLMCTL_VLLM_PIN":   "v0.29.0",
	})

	b.build("keep")
	wantEnv(t, b.env(), map[string]string{"VLLMCTL_BASE_IMAGE": newAlias, "VLLMCTL_BASE_TAG": newer})

	b.run(nil, "CHOSEN_BASE_REF="+older+"\nensure_base_image refresh\nwrite_env_file\n")
	wantEnv(t, b.env(), map[string]string{
		"VLLMCTL_BASE_IMAGE":    oldAlias,
		"VLLMCTL_BASE_TAG":      older,
		"VLLMCTL_BASE_PREVIOUS": newAlias,
		"VLLMCTL_VLLM_PIN":      "v0.27.0",
	})
	if strings.Contains(b.calls(), "pull docker.io/tcclaviger/vllm:latest") {
		t.Error("pulled the fallback tag although a release was chosen")
	}
}

// Each manifest's fallback must be one of the releases it would list, or an
// offline install builds on something the menu would never have offered.
func TestTheFallbackMatchesTheTagPattern(t *testing.T) {
	for _, d := range All() {
		if d.BaseTags == "" {
			continue
		}
		t.Run(d.ID, func(t *testing.T) {
			name := d.BaseImage[strings.LastIndex(d.BaseImage, "/")+1:]
			_, tag, _ := strings.Cut(name, ":")
			if tag == "latest" {
				return
			}
			if !regexp.MustCompile(d.BaseTags).MatchString(tag) {
				t.Errorf("fallback tag %q does not match VARIANT_BASE_TAGS %q", tag, d.BaseTags)
			}
		})
	}
}
