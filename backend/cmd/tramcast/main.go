package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"syscall"

	core_config "github.com/r0mbeg/TramCast/backend/internal/core/config"
	core_logger "github.com/r0mbeg/TramCast/backend/internal/core/logger"
	core_postgres "github.com/r0mbeg/TramCast/backend/internal/core/repository/postgres"
	core_http_server "github.com/r0mbeg/TramCast/backend/internal/core/transport/http"
)

func main() {
	envFile := flag.String("env-file", "", "optional dotenv file; process environment takes priority")
	flag.Parse()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	if err := run(ctx, *envFile); err != nil {
		slog.Error("application stopped with an error", "error", err)
		os.Exit(1)
	}
}

func run(ctx context.Context, envFile string) error {
	cfg, err := core_config.Load(envFile)
	if err != nil {
		return fmt.Errorf("load configuration: %w", err)
	}
	log := core_logger.New(cfg.Logger, os.Stdout)
	slog.SetDefault(log)

	pool, err := core_postgres.NewPool(ctx, cfg.Postgres)
	if err != nil {
		return err
	}
	defer pool.Close()
	log.Info("postgres connection established")

	server := core_http_server.New(cfg.HTTP, log, pool.Ping)
	if err := server.Run(ctx); err != nil {
		return err
	}
	log.Info("application stopped")
	return nil
}
