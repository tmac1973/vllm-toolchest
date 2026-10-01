package recommend

import (
	"context"
	"errors"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// cacheDir is where a finalist's config.json and weight size are kept: a
// directory per repository revision, so a hit never expires and a new
// revision is simply a new key. models.ParseHFConfig reads the directory as
// it reads a downloaded model's, so a remote config takes the same code.
func cacheDir(dataDir string, c Candidate) string {
	key := url.PathEscape(c.ID) + "@" + url.PathEscape(c.LastModified.UTC().Format(time.RFC3339))
	return filepath.Join(dataDir, "recommend", "configs", key)
}

var (
	errNoConfig  = errors.New("config unavailable")
	errNoListing = errors.New("file listing unavailable")
)

// fetchFinalist makes sure a finalist's config.json and weight size are in
// its cache directory, fetching what is missing. The weight size is the
// weight files a download takes, from the file tree -- exact for every
// format, where the Hub's counts are not (see leastWeightGB).
func fetchFinalist(ctx context.Context, hub Hub, dir string, c Candidate) (weights int64, known bool, err error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return 0, false, err
	}
	rev := c.SHA
	if rev == "" {
		rev = "main"
	}
	cfgPath := filepath.Join(dir, "config.json")
	if _, statErr := os.Stat(cfgPath); statErr != nil {
		body, err := hub.FetchConfigJSON(ctx, c.ID, rev)
		if err != nil {
			return 0, false, errNoConfig
		}
		if err := os.WriteFile(cfgPath, body, 0o644); err != nil {
			return 0, false, err
		}
	}

	wPath := filepath.Join(dir, "weights.txt")
	if b, err := os.ReadFile(wPath); err == nil {
		n, perr := strconv.ParseInt(strings.TrimSpace(string(b)), 10, 64)
		if perr == nil {
			return n, n > 0, nil
		}
	}
	_, files, err := hub.GetFiles(ctx, c.ID, rev)
	if err != nil {
		return 0, false, errNoListing
	}
	for _, f := range files {
		if f.Category == "weight" {
			weights += f.Size
		}
	}
	os.WriteFile(wPath, []byte(strconv.FormatInt(weights, 10)), 0o644)
	return weights, weights > 0, nil
}
