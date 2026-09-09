package huggingface

import (
	"os"
	"path/filepath"
	"syscall"
)

// DiskSafetyMarginBytes is free space downloads are never allowed to consume,
// so a finished download cannot leave the filesystem at 100% and take the OS
// down with it.
const DiskSafetyMarginBytes int64 = 2 * 1024 * 1024 * 1024 // 2 GiB

// freeBytesAt returns free bytes on the filesystem holding path, or -1 when it
// cannot be determined.
//
// -1 means "unknown", never "zero": callers must be able to tell the two apart,
// because treating a failed statfs as a full disk would grey out every download
// button on the page with no way to find out why.
//
// The path need not exist yet — before the first download the models directory
// usually does not — so this walks up to the nearest ancestor that does. Any
// ancestor is on the same filesystem unless a mount sits between them, and a
// mount point exists by definition.
func freeBytesAt(path string) int64 {
	for path != "" {
		var st syscall.Statfs_t
		if err := syscall.Statfs(path, &st); err == nil {
			return int64(st.Bavail) * int64(st.Bsize)
		}
		if !os.IsNotExist(errStat(path)) {
			return -1
		}
		parent := filepath.Dir(path)
		if parent == path {
			return -1
		}
		path = parent
	}
	return -1
}

// errStat reports why path could not be used, so freeBytesAt can tell a
// missing directory (walk up) from anything else (give up).
func errStat(path string) error {
	_, err := os.Stat(path)
	return err
}

// FreeBytes is the free space where downloads land, or -1 if unknown.
func (d *Downloader) FreeBytes() int64 {
	return freeBytesAt(d.modelsDir)
}

// PendingBytes is what in-flight downloads have yet to write. Space they will
// need is not space a new download can have.
func (d *Downloader) PendingBytes() int64 {
	d.mu.Lock()
	defer d.mu.Unlock()

	var pending int64
	for _, dl := range d.active {
		dl.mu.Lock()
		if dl.status == "downloading" {
			var written int64
			for _, f := range dl.files {
				written += f.downloaded
			}
			if remaining := dl.totalBytes - written; remaining > 0 {
				pending += remaining
			}
		}
		dl.mu.Unlock()
	}
	return pending
}

// AvailableForDownload is how many bytes a new download may consume: free space
// less the safety margin, less what in-flight downloads have reserved.
//
// Returns -1 when free space is unknown, and clamps to 0 when there genuinely
// is no room.
func (d *Downloader) AvailableForDownload() int64 {
	free := d.FreeBytes()
	if free < 0 {
		return -1
	}
	avail := free - DiskSafetyMarginBytes - d.PendingBytes()
	if avail < 0 {
		return 0
	}
	return avail
}
