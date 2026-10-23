package models

import "testing"

func TestMaxContext(t *testing.T) {
	const yarn2 = `--override-generation-config '{"max_tokens": 65536}' --hf-overrides '{"text_config": {"rope_parameters": {"rope_type": "yarn", "factor": 2.0, "original_max_position_embeddings": 262144, "rope_theta": 10000000}}}'`
	cases := []struct {
		name    string
		trained int
		flags   string
		want    int
	}{
		{"no override", 262144, "", 262144},
		{"yarn under text_config", 262144, yarn2, 524288},
		{"yarn factor 1", 262144, `--hf-overrides '{"rope_parameters": {"rope_type": "yarn", "factor": 1.0}}'`, 262144},
		{"legacy rope_scaling with =", 32768, `--hf-overrides='{"rope_scaling":{"type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}'`, 131072},
		{"not yarn", 32768, `--hf-overrides '{"rope_scaling": {"rope_type": "linear", "factor": 4.0}}'`, 32768},
		{"bad json", 32768, `--hf-overrides '{'`, 32768},
		{"unknown trained", 0, yarn2, 0},
	}
	for _, c := range cases {
		m := &Model{HFConfig: HFConfig{MaxPositionEmbeddings: c.trained}, VLLMConfig: VLLMConfig{ExtraFlags: c.flags}}
		if got := MaxContext(m); got != c.want {
			t.Errorf("%s: MaxContext = %d, want %d", c.name, got, c.want)
		}
	}
}
