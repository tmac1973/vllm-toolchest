package variants_test

import (
	"strings"
	"testing"
)

// An install on a machine that already runs vllm-toolchest used to warn that
// its own ports were taken, because prompt_ports runs before
// container_install stops the container it is replacing. It then told the
// operator to "choose alternative ports", which would move a working UI off
// 3000 for nothing. Seen on a real install, 2026-09-21.
//
// These drive the bash directly rather than the prompt, because the prompt
// blocks on read: the decision being pinned is which listener counts.
func TestOurOwnContainerIsNotAPortConflict(t *testing.T) {
	// Stand in for podman, rendering ports exactly as
	// `podman ps --format '{{.Ports}}'` does.
	const stub = "fakecontainer() { echo '0.0.0.0:3000->3000/tcp, 0.0.0.0:8000->8000/tcp'; }\n" +
		"CONTAINER_CMD=fakecontainer\n"

	for _, tc := range []struct {
		name string
		port string
		want bool
	}{
		{"the management port it publishes", "3000", true},
		{"the inference port it publishes", "8000", true},
		{"a port it does not publish", "3001", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			out := runSetupSh(t, stub+"port_published_by_us "+tc.port+" && echo yes || echo no\n")
			if got := strings.Contains(out, "yes"); got != tc.want {
				t.Errorf("port_published_by_us %s = %v, want %v; output %q",
					tc.port, got, tc.want, out)
			}
		})
	}
}

// The host side of a mapping only. A container forwarding 3000 from host port
// 3001 does not hold 3000, and reading the wrong half would suppress a real
// conflict -- the failure mode worth guarding, since it hands someone a build
// that cannot bind.
func TestOnlyTheHostSideOfAMappingCounts(t *testing.T) {
	const stub = "fakecontainer() { echo '0.0.0.0:3001->3000/tcp'; }\n" +
		"CONTAINER_CMD=fakecontainer\n"

	out := runSetupSh(t, stub+"port_published_by_us 3000 && echo yes || echo no\n")
	if strings.Contains(out, "yes") {
		t.Errorf("3000 appears only as a container port, never as a host port; got %q", out)
	}
}

// Nothing running, or no runtime detected yet: claim nothing, and in
// particular do not shell out to an empty command name.
func TestNoContainerMeansNoClaim(t *testing.T) {
	for _, tc := range []struct{ name, stub string }{
		{"no runtime detected", "CONTAINER_CMD=\n"},
		{"runtime present, nothing running",
			"fakecontainer() { :; }\nCONTAINER_CMD=fakecontainer\n"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			out := runSetupSh(t, tc.stub+"port_published_by_us 3000 && echo yes || echo no\n")
			if strings.Contains(out, "yes") {
				t.Errorf("claimed a port with nothing to claim it; got %q", out)
			}
		})
	}
}
