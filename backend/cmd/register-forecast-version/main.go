// Command register-forecast-version registers the forecast version described
// by the ML service (`service.py --describe`, or a replay bundle manifest),
// checks that the ML server at ML_GRPC_ADDR serves it and optionally makes it
// the active version.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	core_config "github.com/r0mbeg/TramCast/backend/internal/core/config"
	core_domain "github.com/r0mbeg/TramCast/backend/internal/core/domain"
	core_logger "github.com/r0mbeg/TramCast/backend/internal/core/logger"
	core_postgres "github.com/r0mbeg/TramCast/backend/internal/core/repository/postgres"
	forecasts_predictor_grpc "github.com/r0mbeg/TramCast/backend/internal/features/forecasts/predictor/grpc"
	forecasts_service "github.com/r0mbeg/TramCast/backend/internal/features/forecasts/service"
)

type options struct {
	envFile     string
	metadata    string
	activate    bool
	dryRun      bool
	verify      bool
	allowReplay bool
}

func main() {
	var o options
	flag.StringVar(&o.envFile, "env-file", "", "optional dotenv file; process environment takes priority")
	flag.StringVar(&o.metadata, "metadata", "", "required: JSON of service.py --describe or of a replay forecast_bundle.json")
	flag.BoolVar(&o.activate, "activate", false, "make the version active for new requests")
	flag.BoolVar(&o.dryRun, "dry-run", false, "validate, verify and look the version up without writing")
	flag.BoolVar(&o.verify, "verify", true, "call Predict for route 5 at ML_GRPC_ADDR and require the same versions and period")
	flag.BoolVar(&o.allowReplay, "allow-replay", false, "accept replay bundle metadata; for development only")
	flag.Parse()
	// Flag parsing stops at the first positional argument, so a stray path would
	// silently drop the flags after it, including -dry-run.
	if flag.NArg() > 0 {
		slog.Error("unexpected arguments; pass the file with -metadata", "args", flag.Args())
		os.Exit(2)
	}
	if o.metadata == "" {
		slog.Error("-metadata is required")
		os.Exit(2)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	if err := run(ctx, o); err != nil {
		slog.Error("forecast version registration failed", "error", err)
		os.Exit(1)
	}
}

func run(ctx context.Context, o options) error {
	cfg, err := core_config.Load(o.envFile)
	if err != nil {
		return fmt.Errorf("load configuration: %w", err)
	}
	log := core_logger.New(cfg.Logger, os.Stdout)
	slog.SetDefault(log)

	// Like the -file flag of import-catalog, the path is taken from the
	// working directory, not from the env file.
	path, err := filepath.Abs(o.metadata)
	if err != nil {
		return fmt.Errorf("resolve -metadata: %w", err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("read metadata: %w", err)
	}
	spec, err := forecasts_service.ParseVersionMetadata(data)
	if err != nil {
		return fmt.Errorf("parse metadata %s: %w", path, err)
	}
	log.Info("registering forecast version", "metadata", path,
		"model_version", spec.ModelVersion, "dataset_version", spec.DatasetVersion,
		"history_end", moscow(spec.HistoryEnd), "forecast_from", moscow(spec.ForecastFrom), "forecast_to", moscow(spec.ForecastTo),
		"replay", spec.Replay, "dry_run", o.dryRun, "verify", o.verify, "activate", o.activate)

	// Metadata errors must not depend on a reachable database.
	err = forecasts_service.CheckSpec(spec, o.allowReplay)
	if errors.Is(err, forecasts_service.ErrReplayNotAllowed) {
		return fmt.Errorf("%w; pass -allow-replay to use it in development, never as the live GPU version", err)
	}
	if err != nil {
		return fmt.Errorf("invalid metadata %s: %w", path, err)
	}

	pool, err := core_postgres.NewPool(ctx, cfg.Postgres)
	if err != nil {
		return err
	}
	defer pool.Close()

	// A nil *Client must not become a non-nil Predictor.
	var predictor forecasts_service.Predictor
	if o.verify {
		client, err := forecasts_predictor_grpc.New(cfg.ML.Addr, cfg.ML.Timeout)
		if err != nil {
			return err
		}
		defer client.Close()
		predictor = client
	} else {
		log.Warn("ML server check skipped; activating a version it does not serve fails every job of it")
	}

	outcome, err := forecasts_service.NewService(pool, predictor).Apply(ctx, spec, forecasts_service.ApplyOptions{
		Verify: o.verify, DryRun: o.dryRun, Activate: o.activate,
	})
	if err != nil {
		return err
	}
	if o.verify {
		log.Info("ML server serves the version", "ml_grpc_addr", cfg.ML.Addr)
	}
	if o.dryRun && o.activate {
		log.Info("dry run: activation skipped")
	}

	version, created, activated := outcome.Version, outcome.Created, outcome.Activated
	message := "forecast version registered"
	switch {
	case o.dryRun && !version.ID.Valid:
		log.Info("dry run: metadata is valid; the version is not registered yet")
		return nil
	case o.dryRun:
		message = "dry run: the version is already registered"
	case !created:
		message = "forecast version already registered"
	}
	log.Info(message,
		"id", version.ID.String(),
		"created", created,
		"activated", activated,
		"is_active", version.IsActive,
		"model_version", version.ModelVersion,
		"dataset_version", version.DatasetVersion,
		"history_end", moscow(version.HistoryEnd.Time),
		"forecast_from", moscow(version.ForecastFrom.Time),
		"forecast_to", moscow(version.ForecastTo.Time),
	)
	return nil
}

// moscow formats t in the forecast zone, whatever location pgx returns.
func moscow(t time.Time) string {
	return t.In(core_domain.Moscow).Format(time.RFC3339)
}
