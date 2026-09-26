package main

import (
	"context"
	"flag"
	"fmt"
	"io/fs"
	"log/slog"
	"os"
	"os/signal"
	"syscall"

	core_config "github.com/r0mbeg/TramCast/backend/internal/core/config"
	core_logger "github.com/r0mbeg/TramCast/backend/internal/core/logger"
	core_postgres "github.com/r0mbeg/TramCast/backend/internal/core/repository/postgres"
	core_http_server "github.com/r0mbeg/TramCast/backend/internal/core/transport/http"
	routes_sqlc "github.com/r0mbeg/TramCast/backend/internal/features/routes/repository/postgres/sqlc"
	routes_service "github.com/r0mbeg/TramCast/backend/internal/features/routes/service"
	routes_transport_http "github.com/r0mbeg/TramCast/backend/internal/features/routes/transport/http"
	stops_sqlc "github.com/r0mbeg/TramCast/backend/internal/features/stops/repository/postgres/sqlc"
	stops_service "github.com/r0mbeg/TramCast/backend/internal/features/stops/service"
	stops_transport_http "github.com/r0mbeg/TramCast/backend/internal/features/stops/transport/http"
	web_service "github.com/r0mbeg/TramCast/backend/internal/features/web/service"
	web_transport_http "github.com/r0mbeg/TramCast/backend/internal/features/web/transport/http"
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
	api := server.Router().Group("/api")
	routes_transport_http.NewHandler(routes_service.NewService(routes_sqlc.New(pool))).Register(api)
	stops_transport_http.NewHandler(stops_service.NewService(stops_sqlc.New(pool))).Register(api)

	webFiles := os.DirFS(cfg.Web.Dir)
	server.Router().NoRoute(web_transport_http.NewHandler(web_service.NewService(webFiles)).Serve)
	if _, err := fs.Stat(webFiles, web_service.IndexFile); err != nil {
		log.Warn("frontend build not found; pages will return 404", "dir", cfg.Web.Dir)
	} else {
		log.Info("serving frontend", "dir", cfg.Web.Dir)
	}

	if err := server.Run(ctx); err != nil {
		return err
	}
	log.Info("application stopped")
	return nil
}
