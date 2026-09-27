package core_config

import (
	"errors"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// These tests change process environment and therefore must not run in parallel.
func cleanConfigEnv(t *testing.T) {
	t.Helper()
	for _, key := range []string{
		"HTTP_ADDR", "HTTP_READ_HEADER_TIMEOUT", "HTTP_READ_TIMEOUT",
		"HTTP_WRITE_TIMEOUT", "HTTP_IDLE_TIMEOUT", "HTTP_SHUTDOWN_TIMEOUT", "HTTP_PROBE_TIMEOUT",
		"LOGGER_LEVEL", "LOGGER_FORMAT", "POSTGRES_HOST", "POSTGRES_PORT",
		"POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB", "POSTGRES_SSLMODE",
		"POSTGRES_MAX_CONNS", "POSTGRES_MIN_CONNS", "POSTGRES_CONNECT_TIMEOUT", "POSTGRES_STARTUP_TIMEOUT",
		"WEB_DIR", "CATALOG_FILE", "CATALOG_OSM_FILE",
	} {
		previous, existed := os.LookupEnv(key)
		if err := os.Unsetenv(key); err != nil {
			t.Fatalf("unset %s: %v", key, err)
		}
		t.Cleanup(func() {
			var err error
			if existed {
				err = os.Setenv(key, previous)
			} else {
				err = os.Unsetenv(key)
			}
			if err != nil {
				t.Errorf("restore %s: %v", key, err)
			}
		})
	}
}

func setRequiredEnv(t *testing.T) {
	t.Helper()
	t.Setenv("POSTGRES_USER", "test_user")
	t.Setenv("POSTGRES_PASSWORD", "test_password")
	t.Setenv("POSTGRES_DB", "test_db")
}

func writeEnvFile(t *testing.T, contents string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), ".env")
	if err := os.WriteFile(path, []byte(contents), 0600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestLoadDefaults(t *testing.T) {
	cleanConfigEnv(t)
	setRequiredEnv(t)
	cfg, err := Load("")
	if err != nil {
		t.Fatal(err)
	}
	wantHTTP := HTTPConfig{
		Addr: ":8080", ReadHeaderTimeout: 5 * time.Second, ReadTimeout: 15 * time.Second,
		WriteTimeout: 30 * time.Second, IdleTimeout: time.Minute,
		ShutdownTimeout: 10 * time.Second, ProbeTimeout: 2 * time.Second,
	}
	if cfg.HTTP != wantHTTP {
		t.Errorf("HTTP defaults: got %+v, want %+v", cfg.HTTP, wantHTTP)
	}
	if cfg.Logger.Level != slog.LevelInfo || cfg.Logger.Format != "text" {
		t.Error("unexpected logger defaults")
	}
	if cfg.Postgres.Host != "127.0.0.1" || cfg.Postgres.Port != 5433 || cfg.Postgres.SSLMode != "disable" ||
		cfg.Postgres.MaxConns != 10 || cfg.Postgres.MinConns != 0 ||
		cfg.Postgres.ConnectTimeout != 5*time.Second || cfg.Postgres.StartupTimeout != 10*time.Second {
		t.Error("unexpected PostgreSQL defaults")
	}
}

func TestLoadResolvesPaths(t *testing.T) {
	root := t.TempDir()
	backend := filepath.Join(root, "backend")
	if err := os.Mkdir(backend, 0700); err != nil {
		t.Fatal(err)
	}
	t.Chdir(backend)

	defaults := [3]string{"frontend/dist", "data/catalog/catalog.xlsx", "data/osm/tram_routes.json"}
	absolute := [3]string{
		filepath.Join(root, "absolute", "web"),
		filepath.Join(root, "absolute", "catalog.xlsx"),
		filepath.Join(root, "absolute", "osm.json"),
	}
	for _, tt := range []struct {
		name         string
		envFile      string
		filePaths    [3]string
		processPaths [3]string
		wantBase     string
		wantPaths    [3]string
	}{
		{
			name:     "defaults without env file use working directory",
			wantBase: backend, wantPaths: defaults,
		},
		{
			name:         "relative process paths without env file use working directory",
			processPaths: [3]string{"../static", "../catalog.xlsx", "../osm.json"},
			wantBase:     root, wantPaths: [3]string{"static", "catalog.xlsx", "osm.json"},
		},
		{
			name: "defaults with root env file use repository root", envFile: "../.env",
			wantBase: root, wantPaths: defaults,
		},
		{
			name: "relative file paths use env directory", envFile: "../.env",
			filePaths: [3]string{"./static", "./catalog.xlsx", "./osm.json"},
			wantBase:  root, wantPaths: [3]string{"static", "catalog.xlsx", "osm.json"},
		},
		{
			name: "process overrides use env directory", envFile: "../.env",
			filePaths:    [3]string{"./file/web", "./file/catalog.xlsx", "./file/osm.json"},
			processPaths: [3]string{"./process/web", "./process/catalog.xlsx", "./process/osm.json"},
			wantBase:     root, wantPaths: [3]string{"process/web", "process/catalog.xlsx", "process/osm.json"},
		},
		{
			name: "absolute env file path", envFile: filepath.Join(root, ".env"),
			wantBase: root, wantPaths: defaults,
		},
		{
			name: "absolute file paths remain absolute", envFile: "../.env",
			filePaths: absolute, wantPaths: absolute,
		},
		{
			name:         "absolute process paths without env file remain absolute",
			processPaths: absolute, wantPaths: absolute,
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			cleanConfigEnv(t)
			setRequiredEnv(t)
			keys := [3]string{"WEB_DIR", "CATALOG_FILE", "CATALOG_OSM_FILE"}
			var contents strings.Builder
			for i, key := range keys {
				if tt.filePaths[i] != "" {
					contents.WriteString(key + "='" + tt.filePaths[i] + "'\n")
				}
				if tt.processPaths[i] != "" {
					t.Setenv(key, tt.processPaths[i])
				}
			}
			if tt.envFile != "" {
				if err := os.WriteFile(filepath.Join(root, ".env"), []byte(contents.String()), 0600); err != nil {
					t.Fatal(err)
				}
			}
			cfg, err := Load(tt.envFile)
			if err != nil {
				t.Fatal(err)
			}
			for i, got := range [3]string{cfg.Web.Dir, cfg.Catalog.File, cfg.Catalog.OSMFile} {
				want := filepath.Join(tt.wantBase, tt.wantPaths[i])
				if got != want {
					t.Errorf("%s = %q, want %q", keys[i], got, want)
				}
			}
		})
	}
}

func TestLoadExplicitEnvFileAndProcessPriority(t *testing.T) {
	cleanConfigEnv(t)
	path := writeEnvFile(t, "POSTGRES_USER=file_user\nPOSTGRES_PASSWORD='file $ # password'\nPOSTGRES_DB=file_db\nHTTP_ADDR=:9090\nLOGGER_LEVEL=DEBUG\nLOGGER_FORMAT=json\n")
	t.Setenv("POSTGRES_USER", "process_user")
	t.Setenv("HTTP_ADDR", "127.0.0.1:8081")
	t.Setenv("LOGGER_LEVEL", "WARN")

	cfg, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Postgres.User != "process_user" || cfg.HTTP.Addr != "127.0.0.1:8081" || cfg.Logger.Level != slog.LevelWarn {
		t.Error("process environment must override the dotenv file")
	}
	if cfg.Postgres.Database != "file_db" || cfg.Postgres.Password != "file $ # password" || cfg.Logger.Format != "json" {
		t.Error("unset variables must be loaded from the explicit dotenv file without changing quoted values")
	}
}

func TestLoadEmptyProcessValueOverridesFile(t *testing.T) {
	cleanConfigEnv(t)
	path := writeEnvFile(t, "POSTGRES_USER=file_user\nPOSTGRES_PASSWORD=file_password\nPOSTGRES_DB=file_db\n")
	t.Setenv("POSTGRES_PASSWORD", "")
	_, err := Load(path)
	if err == nil || !strings.Contains(err.Error(), "POSTGRES_PASSWORD") {
		t.Fatal("an explicitly empty password must override the file and fail validation")
	}
}

func TestLoadDoesNotSearchForDotEnv(t *testing.T) {
	cleanConfigEnv(t)
	path := writeEnvFile(t, "POSTGRES_USER=file_user\nPOSTGRES_PASSWORD=file_password\nPOSTGRES_DB=file_db\n")
	t.Chdir(filepath.Dir(path))
	_, err := Load("")
	if err == nil || !strings.Contains(err.Error(), "POSTGRES_USER") {
		t.Fatal("Load without an env-file path must not implicitly load .env")
	}
}

func TestLoadValidation(t *testing.T) {
	for _, tt := range []struct{ name, key, value string }{
		{"blank host", "POSTGRES_HOST", " "},
		{"blank user", "POSTGRES_USER", " "},
		{"blank password", "POSTGRES_PASSWORD", " "},
		{"blank database", "POSTGRES_DB", " "},
		{"zero port", "POSTGRES_PORT", "0"},
		{"overflow port", "POSTGRES_PORT", "65536"},
		{"zero pool", "POSTGRES_MAX_CONNS", "0"},
		{"negative pool minimum", "POSTGRES_MIN_CONNS", "-1"},
		{"minimum exceeds maximum", "POSTGRES_MIN_CONNS", "11"},
		{"invalid sslmode", "POSTGRES_SSLMODE", "off"},
		{"zero connect timeout", "POSTGRES_CONNECT_TIMEOUT", "0s"},
		{"negative startup timeout", "POSTGRES_STARTUP_TIMEOUT", "-1s"},
		{"zero header timeout", "HTTP_READ_HEADER_TIMEOUT", "0s"},
		{"zero read timeout", "HTTP_READ_TIMEOUT", "0s"},
		{"zero write timeout", "HTTP_WRITE_TIMEOUT", "0s"},
		{"zero idle timeout", "HTTP_IDLE_TIMEOUT", "0s"},
		{"zero shutdown timeout", "HTTP_SHUTDOWN_TIMEOUT", "0s"},
		{"zero probe timeout", "HTTP_PROBE_TIMEOUT", "0s"},
		{"missing HTTP port", "HTTP_ADDR", "localhost"},
		{"named HTTP port", "HTTP_ADDR", "localhost:http"},
		{"overflow HTTP port", "HTTP_ADDR", ":65536"},
		{"invalid HTTP host", "HTTP_ADDR", "bad host:8080"},
		{"invalid log level", "LOGGER_LEVEL", "quiet"},
		{"invalid log format", "LOGGER_FORMAT", "xml"},
		{"blank web directory", "WEB_DIR", " "},
		{"blank catalog file", "CATALOG_FILE", " "},
		{"blank OSM snapshot file", "CATALOG_OSM_FILE", " "},
	} {
		t.Run(tt.name, func(t *testing.T) {
			cleanConfigEnv(t)
			setRequiredEnv(t)
			t.Setenv(tt.key, tt.value)
			_, err := Load("")
			if err == nil || !strings.Contains(err.Error(), tt.key) {
				t.Errorf("expected a validation error naming %s", tt.key)
			}
		})
	}
}

func TestLoadRequiredEnvironment(t *testing.T) {
	for _, key := range []string{"POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB"} {
		t.Run(key, func(t *testing.T) {
			cleanConfigEnv(t)
			setRequiredEnv(t)
			if err := os.Unsetenv(key); err != nil {
				t.Fatal(err)
			}
			_, err := Load("")
			if err == nil || !strings.Contains(err.Error(), key) {
				t.Errorf("expected a missing-variable error naming %s", key)
			}
		})
	}
}

func TestLoadErrorsDoNotExposeValues(t *testing.T) {
	const sensitiveValue = "sensitive-test-value"
	t.Run("dotenv syntax", func(t *testing.T) {
		cleanConfigEnv(t)
		path := writeEnvFile(t, "POSTGRES_PASSWORD='"+sensitiveValue)
		_, err := Load(path)
		if err == nil || strings.Contains(err.Error(), sensitiveValue) {
			t.Fatal("invalid dotenv must return an error without file contents")
		}
	})
	t.Run("typed environment value", func(t *testing.T) {
		cleanConfigEnv(t)
		setRequiredEnv(t)
		t.Setenv("POSTGRES_PORT", sensitiveValue)
		_, err := Load("")
		if err == nil || strings.Contains(err.Error(), sensitiveValue) || !strings.Contains(err.Error(), "POSTGRES_PORT") {
			t.Fatal("parse errors must identify the setting without its value")
		}
	})
}

func TestLoadMissingExplicitFile(t *testing.T) {
	cleanConfigEnv(t)
	setRequiredEnv(t)
	_, err := Load(filepath.Join(t.TempDir(), "missing.env"))
	if !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("expected a missing-file error, got %v", err)
	}
}
