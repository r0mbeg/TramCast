package core_postgres

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/url"
	"strconv"

	"github.com/jackc/pgx/v5/pgxpool"

	core_config "github.com/r0mbeg/TramCast/backend/internal/core/config"
)

// NewPool returns a pool that can be passed directly to sqlc's New functions.
// The application owns the pool and closes it after HTTP handlers have stopped.
func NewPool(ctx context.Context, cfg core_config.PostgresConfig) (*pgxpool.Pool, error) {
	poolConfig, err := newPoolConfig(cfg)
	if err != nil {
		return nil, err
	}

	startupCtx, cancel := context.WithTimeout(ctx, cfg.StartupTimeout)
	defer cancel()

	pool, err := pgxpool.NewWithConfig(startupCtx, poolConfig)
	if err != nil {
		return nil, fmt.Errorf("create postgres pool: %w", err)
	}
	if err := pool.Ping(startupCtx); err != nil {
		pool.Close()
		return nil, fmt.Errorf("connect to postgres: %w", err)
	}
	return pool, nil
}

func newPoolConfig(cfg core_config.PostgresConfig) (*pgxpool.Config, error) {
	connectionURL := &url.URL{
		Scheme: "postgres",
		User:   url.UserPassword(cfg.User, cfg.Password),
		Host:   net.JoinHostPort(cfg.Host, strconv.Itoa(int(cfg.Port))),
		Path:   "/" + cfg.Database,
	}
	query := url.Values{}
	query.Set("sslmode", cfg.SSLMode)
	connectionURL.RawQuery = query.Encode()

	poolConfig, err := pgxpool.ParseConfig(connectionURL.String())
	if err != nil {
		// ParseConfig errors may include the complete connection string.
		return nil, errors.New("invalid postgres connection configuration")
	}
	poolConfig.MaxConns = cfg.MaxConns
	poolConfig.MinConns = cfg.MinConns
	poolConfig.ConnConfig.ConnectTimeout = cfg.ConnectTimeout
	poolConfig.ConnConfig.RuntimeParams["application_name"] = "tramcast"
	return poolConfig, nil
}
