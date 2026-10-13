package index

import (
	"context"
	"errors"
	"fmt"
)

// errNoSnapshotPath is Flush's error for an engine built without a snapshot
// path: there is nowhere to write one.
var errNoSnapshotPath = errors.New("no snapshot path configured")

// flush is Engine.Flush for both engines.
func flush(ctx context.Context, e FileEngine, snapshotPath string) error {
	if snapshotPath == "" {
		return errNoSnapshotPath
	}

	return e.SaveToFile(ctx, snapshotPath)
}

// recoverFrom is Engine.Recover for both engines: the snapshot at
// snapshotPath, if any, then the WAL replayed over it.
//
// A snapshot that exists but can't be loaded is an error, not a warning: the
// WAL only holds writes since the last save, so replaying it alone would serve
// a fraction of the data as if it were all of it. A missing snapshot is a
// fresh start -- LoadFromFile returns nil for it.
//
// A WAL that can't be read to the end is an error for the same reason: every
// write after the failure point would be silently dropped. A missing or empty
// WAL is a fresh start; see replayWALFile for what counts as a torn write.
func recoverFrom(ctx context.Context, e FileEngine, snapshotPath string, wal *WAL) error {
	if snapshotPath != "" {
		if err := e.LoadFromFile(ctx, snapshotPath); err != nil {
			return fmt.Errorf("load snapshot %s (restore a backup, or delete it to start empty): %w", snapshotPath, err)
		}
	}

	if wal != nil {
		if err := e.ReplayWAL(wal.Path()); err != nil {
			return fmt.Errorf("replay WAL %s (fix or move it aside to start from the snapshot alone): %w", wal.Path(), err)
		}
	}

	return nil
}
