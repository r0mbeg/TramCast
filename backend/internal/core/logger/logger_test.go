package core_logger

import (
	"bytes"
	"encoding/json"
	"log/slog"
	"strings"
	"testing"

	core_config "github.com/r0mbeg/TramCast/backend/internal/core/config"
)

func TestNewJSONFiltersLevelsAndPreservesFields(t *testing.T) {
	var output bytes.Buffer
	log := New(core_config.LoggerConfig{Level: slog.LevelWarn, Format: "json"}, &output)
	log.Info("filtered")
	log.With("component", "postgres").Warn("connection unavailable", "attempt", 2)

	var record map[string]any
	if err := json.Unmarshal(output.Bytes(), &record); err != nil {
		t.Fatalf("expected one JSON log record: %v", err)
	}
	if record["level"] != "WARN" || record["msg"] != "connection unavailable" ||
		record["component"] != "postgres" || record["attempt"] != float64(2) {
		t.Errorf("unexpected log record: %+v", record)
	}
}

func TestNewTextWritesToProvidedOutput(t *testing.T) {
	var output bytes.Buffer
	log := New(core_config.LoggerConfig{Level: slog.LevelDebug, Format: "text"}, &output)
	log.Debug("server starting", "addr", ":8080")
	for _, fragment := range []string{"level=DEBUG", `msg="server starting"`, "addr=:8080"} {
		if !strings.Contains(output.String(), fragment) {
			t.Errorf("text output does not contain %q", fragment)
		}
	}
}
