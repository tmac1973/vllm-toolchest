package autoconfig

import (
	"strings"
	"testing"

	"github.com/tmac1973/vllm-toolchest/internal/models"
)

// tcclaviger's Gemma 4 card says it needs tcclaviger/vllm. On that image it
// gets a note; on another, a warning; and an example docker command naming
// some image is not a requirement.
func TestTheCardsImageRequirement(t *testing.T) {
	card := "# Gemma 4\n\n> - Requires [`tcclaviger/vllm:latest`](https://hub.docker.com/r/tcclaviger/vllm) — an RDNA 4 (gfx12xx) vLLM image and the only build with the `mxfp4_16` kernels; no other vLLM build loads these weights.\n"
	clav := Image{ID: "rdna4-clav", Label: "RDNA4 native-HIP (tcclaviger)", BaseImage: "docker.io/tcclaviger/vllm:latest"}
	radiance := Image{ID: "radiance", Label: "RDNA4 Radiance", BaseImage: "docker.io/someone/radiance-vllm:0.27"}
	in := Inputs{Model: &models.Model{ID: "tcclaviger/gemma-4-31B-it-MXFP416-MTP"}, Card: Card{Raw: card, Text: card}}

	in.Image = clav
	c := Validate(in)
	if c.ImageWarning != "" || !noteFor(c.Notes, "", "the one this machine serves with") {
		t.Errorf("on its own image: warning %q, notes %+v", c.ImageWarning, c.Notes)
	}

	in.Image = radiance
	c = Validate(in)
	if !strings.Contains(c.ImageWarning, "“Requires tcclaviger/vllm:latest — an RDNA 4") || !strings.Contains(c.ImageWarning, "RDNA4 Radiance") {
		t.Errorf("on another image: %q", c.ImageWarning)
	}

	in.Image = Image{}
	if c := Validate(in); c.ImageWarning != "" || !noteFor(c.Notes, "", "not known") {
		t.Errorf("image unknown: warning %q", c.ImageWarning)
	}

	example := "# Model\n\nServe it with Docker:\n\n```\ndocker run --gpus all vllm/vllm-openai:latest --model org/m\n```\n"
	in = Inputs{Model: &models.Model{ID: "org/m"}, Card: Card{Raw: example, Text: example}, Image: radiance}
	if c := Validate(in); c.ImageWarning != "" {
		t.Errorf("an example command was taken as a requirement: %q", c.ImageWarning)
	}
}
