package main

import (
	"context"
	"flag"
	"fmt"
	"io/fs"
	"log/slog"
	"os"
	"os/signal"
	"sync"
	"syscall"

	core_config "github.com/r0mbeg/TramCast/backend/internal/core/config"
	core_logger "github.com/r0mbeg/TramCast/backend/internal/core/logger"
	core_postgres "github.com/r0mbeg/TramCast/backend/internal/core/repository/postgres"
	core_http_server "github.com/r0mbeg/TramCast/backend/internal/core/transport/http"
	forecasts_predictor_grpc "github.com/r0mbeg/TramCast/backend/internal/features/forecasts/predictor/grpc"
	forecasts_service "github.com/r0mbeg/TramCast/backend/internal/features/forecasts/service"
	forecasts_worker "github.com/r0mbeg/TramCast/backend/internal/features/forecasts/worker"
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

	// runCtx also stops the worker when the HTTP server fails to start.
	runCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	var wg sync.WaitGroup
	workerStarted := false
	if jobs := cfg.PredictionJobs; jobs.WorkerEnabled {
		client, err := forecasts_predictor_grpc.New(cfg.ML.Addr, cfg.ML.Timeout)
		if err != nil {
			return err
		}
		// Deferred after pool.Close, so it runs first; both run after wg.Wait.
		defer client.Close()
		queue := forecasts_service.NewQueue(pool, forecasts_service.JobPolicy{
			QueueCapacity: jobs.QueueCapacity,
			MaxAttempts:   jobs.MaxAttempts,
			LeaseDuration: jobs.LeaseDuration,
			RetryDelay:    jobs.RetryDelay,
		})
		worker := forecasts_worker.New(queue, client, forecasts_worker.Config{
			Workers:         jobs.Workers,
			PollInterval:    jobs.PollInterval,
			LeaseDuration:   jobs.LeaseDuration,
			ShutdownTimeout: jobs.ShutdownTimeout,
		}, log)
		log.Info("prediction job worker started", "workers", jobs.Workers, "ml_grpc_addr", cfg.ML.Addr)
		wg.Go(func() { worker.Run(runCtx) })
		workerStarted = true
	} else {
		log.Info("prediction job worker disabled")
	}

	// The HTTP drain and the worker's shutdown grace run in parallel and must
	// fit the compose stop grace period.
	err = server.Run(runCtx)
	cancel()
	wg.Wait()
	if workerStarted {
		log.Info("prediction job worker stopped")
	}
	if err != nil {
		return err
	}
	log.Info("application stopped")
	return nil
}
