package core_logger

import (
	"io"
	"log/slog"

	core_config "github.com/r0mbeg/TramCast/backend/internal/core/config"
)

// New creates a logger from validated configuration. The caller owns output.
func New(cfg core_config.LoggerConfig, output io.Writer) *slog.Logger {
	options := &slog.HandlerOptions{Level: cfg.Level}
	var handler slog.Handler
	if cfg.Format == "json" {
		handler = slog.NewJSONHandler(output, options)
	} else {
		handler = slog.NewTextHandler(output, options)
	}
	return slog.New(handler)
}
