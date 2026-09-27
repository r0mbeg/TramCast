package core_config

import (
	"errors"
	"fmt"
	"log/slog"
	"net"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/joho/godotenv"
	"github.com/kelseyhightower/envconfig"
)

type Config struct {
	HTTP     HTTPConfig
	Logger   LoggerConfig
	Postgres PostgresConfig
	Web      WebConfig
	Catalog  CatalogConfig
}

type HTTPConfig struct {
	Addr              string        `envconfig:"HTTP_ADDR" default:":8080"`
	ReadHeaderTimeout time.Duration `envconfig:"HTTP_READ_HEADER_TIMEOUT" default:"5s"`
	ReadTimeout       time.Duration `envconfig:"HTTP_READ_TIMEOUT" default:"15s"`
	WriteTimeout      time.Duration `envconfig:"HTTP_WRITE_TIMEOUT" default:"30s"`
	IdleTimeout       time.Duration `envconfig:"HTTP_IDLE_TIMEOUT" default:"60s"`
	ShutdownTimeout   time.Duration `envconfig:"HTTP_SHUTDOWN_TIMEOUT" default:"10s"`
	ProbeTimeout      time.Duration `envconfig:"HTTP_PROBE_TIMEOUT" default:"2s"`
}

type LoggerConfig struct {
	Level  slog.Level `envconfig:"LOGGER_LEVEL" default:"INFO"`
	Format string     `envconfig:"LOGGER_FORMAT" default:"text"`
}

type PostgresConfig struct {
	Host           string        `envconfig:"POSTGRES_HOST" default:"127.0.0.1"`
	Port           uint16        `envconfig:"POSTGRES_PORT" default:"5433"`
	User           string        `envconfig:"POSTGRES_USER" required:"true"`
	Password       string        `envconfig:"POSTGRES_PASSWORD" required:"true"`
	Database       string        `envconfig:"POSTGRES_DB" required:"true"`
	SSLMode        string        `envconfig:"POSTGRES_SSLMODE" default:"disable"`
	MaxConns       int32         `envconfig:"POSTGRES_MAX_CONNS" default:"10"`
	MinConns       int32         `envconfig:"POSTGRES_MIN_CONNS" default:"0"`
	ConnectTimeout time.Duration `envconfig:"POSTGRES_CONNECT_TIMEOUT" default:"5s"`
	StartupTimeout time.Duration `envconfig:"POSTGRES_STARTUP_TIMEOUT" default:"10s"`
}

type WebConfig struct {
	// Dir is the frontend build directory, resolved by Load like catalog paths.
	Dir string `envconfig:"WEB_DIR" default:"./frontend/dist"`
}

type CatalogConfig struct {
	// File is the reference workbook read by the catalog import command. Load
	// makes it absolute; the default is the sanitized workbook in data/catalog.
	File string `envconfig:"CATALOG_FILE" default:"./data/catalog/catalog.xlsx"`
	// OSMFile is the OpenStreetMap snapshot of the target routes the workbook
	// lacks, also read by the import command. It is resolved like File; the
	// default is the snapshot committed in data/osm.
	OSMFile string `envconfig:"CATALOG_OSM_FILE" default:"./data/osm/tram_routes.json"`
}

// Load reads process environment and optionally loads the explicitly named
// dotenv file first. Existing environment variables, including empty values,
// take precedence. Call once during startup, before launching goroutines.
// An empty envFile never triggers an implicit search for .env.
// Relative filesystem paths, including defaults and process overrides, are
// resolved from the env file's directory, or the working directory without one.
func Load(envFile string) (Config, error) {
	pathBase := "."
	if envFile != "" {
		absoluteEnvFile, err := filepath.Abs(envFile)
		if err != nil {
			return Config{}, fmt.Errorf("resolve env file %q: %w", envFile, err)
		}
		pathBase = filepath.Dir(absoluteEnvFile)
		if err := godotenv.Load(envFile); err != nil {
			var pathErr *os.PathError
			if errors.As(err, &pathErr) {
				return Config{}, fmt.Errorf("read env file %q: %w", envFile, pathErr.Err)
			}
			// Parser errors may contain dotenv values, including credentials.
			return Config{}, fmt.Errorf("load env file %q: invalid dotenv contents", envFile)
		}
	}

	var cfg Config
	for _, section := range []any{&cfg.HTTP, &cfg.Logger, &cfg.Postgres, &cfg.Web, &cfg.Catalog} {
		if err := envconfig.Process("", section); err != nil {
			var parseErr *envconfig.ParseError
			if errors.As(err, &parseErr) {
				// envconfig.ParseError includes the original value in its message.
				return Config{}, fmt.Errorf("invalid value for %s", parseErr.KeyName)
			}
			return Config{}, fmt.Errorf("load configuration: %w", err)
		}
	}

	if err := cfg.validate(); err != nil {
		return Config{}, err
	}
	for _, setting := range []struct {
		name  string
		value *string
	}{
		{"WEB_DIR", &cfg.Web.Dir},
		{"CATALOG_FILE", &cfg.Catalog.File},
		{"CATALOG_OSM_FILE", &cfg.Catalog.OSMFile},
	} {
		path := *setting.value
		if !filepath.IsAbs(path) {
			path = filepath.Join(pathBase, path)
		}
		absolute, err := filepath.Abs(path)
		if err != nil {
			return Config{}, fmt.Errorf("resolve %s: %w", setting.name, err)
		}
		*setting.value = absolute
	}
	return cfg, nil
}

func (cfg Config) validate() error {
	host, port, err := net.SplitHostPort(cfg.HTTP.Addr)
	if err != nil || strings.ContainsAny(host, " /\\\t\r\n?#") {
		return errors.New("HTTP_ADDR must have the form host:port or :port")
	}
	if _, err := strconv.ParseUint(port, 10, 16); err != nil {
		return errors.New("HTTP_ADDR must contain a numeric port from 0 to 65535")
	}

	for _, setting := range []struct {
		name  string
		value time.Duration
	}{
		{"HTTP_READ_HEADER_TIMEOUT", cfg.HTTP.ReadHeaderTimeout},
		{"HTTP_READ_TIMEOUT", cfg.HTTP.ReadTimeout},
		{"HTTP_WRITE_TIMEOUT", cfg.HTTP.WriteTimeout},
		{"HTTP_IDLE_TIMEOUT", cfg.HTTP.IdleTimeout},
		{"HTTP_SHUTDOWN_TIMEOUT", cfg.HTTP.ShutdownTimeout},
		{"HTTP_PROBE_TIMEOUT", cfg.HTTP.ProbeTimeout},
		{"POSTGRES_CONNECT_TIMEOUT", cfg.Postgres.ConnectTimeout},
		{"POSTGRES_STARTUP_TIMEOUT", cfg.Postgres.StartupTimeout},
	} {
		if setting.value <= 0 {
			return fmt.Errorf("%s must be positive", setting.name)
		}
	}

	if cfg.Logger.Format != "text" && cfg.Logger.Format != "json" {
		return errors.New("LOGGER_FORMAT must be text or json")
	}
	for _, setting := range []struct {
		name  string
		value string
	}{
		{"POSTGRES_HOST", cfg.Postgres.Host},
		{"POSTGRES_USER", cfg.Postgres.User},
		{"POSTGRES_PASSWORD", cfg.Postgres.Password},
		{"POSTGRES_DB", cfg.Postgres.Database},
		{"WEB_DIR", cfg.Web.Dir},
		{"CATALOG_FILE", cfg.Catalog.File},
		{"CATALOG_OSM_FILE", cfg.Catalog.OSMFile},
	} {
		if strings.TrimSpace(setting.value) == "" {
			return fmt.Errorf("%s must not be blank", setting.name)
		}
	}
	if cfg.Postgres.Port == 0 {
		return errors.New("POSTGRES_PORT must be between 1 and 65535")
	}
	if cfg.Postgres.MaxConns <= 0 {
		return errors.New("POSTGRES_MAX_CONNS must be positive")
	}
	if cfg.Postgres.MinConns < 0 || cfg.Postgres.MinConns > cfg.Postgres.MaxConns {
		return errors.New("POSTGRES_MIN_CONNS must be between 0 and POSTGRES_MAX_CONNS")
	}
	switch cfg.Postgres.SSLMode {
	case "disable", "allow", "prefer", "require", "verify-ca", "verify-full":
	default:
		return errors.New("POSTGRES_SSLMODE must be disable, allow, prefer, require, verify-ca or verify-full")
	}
	return nil
}
