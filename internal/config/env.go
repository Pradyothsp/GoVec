package config

import (
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/rs/zerolog/log"
)

// Typed readers for GOVEC_* environment overrides.
//
// Every override follows the same three rules, and keeping them in one place
// is what makes adding a config field a one-line change rather than another
// copy of the read/parse/warn block:
//
//   - An unset or empty variable leaves the existing value alone, so an
//     override never resets a field to its zero value by accident.
//   - An unparseable value warns, naming the variable, and keeps the previous
//     value. Refusing to start over one bad env var is a worse failure mode
//     for a container than running with the configured default.
//   - Validation is not done here. Load runs Validate on the merged config, so
//     a value that parses but makes no sense is rejected the same way whether
//     it came from YAML or the environment.

// envString applies a string-like override. The type parameter covers the
// named string types (IndexType, Quantization, DistanceMetric) as well as
// plain strings.
func envString[T ~string](key string, dst *T) {
	if val := os.Getenv(key); val != "" {
		*dst = T(val)
	}
}

func envInt(key string, dst *int) {
	val := os.Getenv(key)
	if val == "" {
		return
	}

	n, err := strconv.Atoi(val)
	if err != nil {
		log.Warn().Str("var", key).Str("value", val).Err(err).Msg("failed to parse env override, using previous value")
		return
	}

	*dst = n
}

func envBool(key string, dst *bool) {
	val := os.Getenv(key)
	if val == "" {
		return
	}

	switch strings.ToLower(val) {
	case "true":
		*dst = true
	case "false":
		*dst = false
	default:
		log.Warn().Str("var", key).Str("value", val).Msg("failed to parse env override, expected 'true' or 'false', using previous value")
	}
}

func envDuration(key string, dst *time.Duration) {
	val := os.Getenv(key)
	if val == "" {
		return
	}

	d, err := time.ParseDuration(val)
	if err != nil {
		log.Warn().Str("var", key).Str("value", val).Err(err).Msg("failed to parse env override, using previous value")
		return
	}

	*dst = d
}
