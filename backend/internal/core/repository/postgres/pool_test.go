package core_postgres

import (
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	core_config "github.com/r0mbeg/TramCast/backend/internal/core/config"
	forecasts_sqlc "github.com/r0mbeg/TramCast/backend/internal/features/forecasts/repository/postgres/sqlc"
	routes_sqlc "github.com/r0mbeg/TramCast/backend/internal/features/routes/repository/postgres/sqlc"
	stops_sqlc "github.com/r0mbeg/TramCast/backend/internal/features/stops/repository/postgres/sqlc"
)

// Keep the pool compatible with all generated query packages, including COPY.
var (
	_ forecasts_sqlc.DBTX = (*pgxpool.Pool)(nil)
	_ routes_sqlc.DBTX    = (*pgxpool.Pool)(nil)
	_ stops_sqlc.DBTX     = (*pgxpool.Pool)(nil)
)

func TestPoolConfigPreservesCredentials(t *testing.T) {
	cfg := core_config.PostgresConfig{
		Host: "::1", Port: 5433, User: "user@project", Password: "p@ss:/?#% 'word",
		Database: "tram cast/тест", SSLMode: "disable", MaxConns: 10, MinConns: 0,
		ConnectTimeout: 3 * time.Second,
	}
	poolConfig, err := newPoolConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	conn := poolConfig.ConnConfig
	if conn.User != cfg.User || conn.Password != cfg.Password || conn.Database != cfg.Database {
		t.Fatal("connection URL did not preserve credential/database values")
	}
	if conn.Host != cfg.Host || conn.Port != cfg.Port {
		t.Fatal("connection URL did not preserve IPv6 host and port")
	}
	if conn.ConnectTimeout != cfg.ConnectTimeout || poolConfig.MaxConns != cfg.MaxConns {
		t.Fatal("pool configuration did not apply limits")
	}
}

func TestPoolConfigErrorDoesNotExposePassword(t *testing.T) {
	const secret = "do-not-log-this-password"
	_, err := newPoolConfig(core_config.PostgresConfig{
		Host: "localhost", Port: 5433, User: "tramcast", Password: secret,
		Database: "tramcast", SSLMode: "invalid",
	})
	if err == nil {
		t.Fatal("expected invalid sslmode to fail")
	}
	if strings.Contains(err.Error(), secret) {
		t.Fatal("configuration error exposes the password")
	}
}
