// Command enqueue-prediction-jobs admits one prediction job for every route
// with forecasts enabled, so the worker of a running backend computes and
// publishes the full horizon of a forecast version before the first request.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/jackc/pgx/v5/pgtype"

	core_config "github.com/r0mbeg/TramCast/backend/internal/core/config"
	core_domain "github.com/r0mbeg/TramCast/backend/internal/core/domain"
	core_logger "github.com/r0mbeg/TramCast/backend/internal/core/logger"
	core_postgres "github.com/r0mbeg/TramCast/backend/internal/core/repository/postgres"
	forecasts_sqlc "github.com/r0mbeg/TramCast/backend/internal/features/forecasts/repository/postgres/sqlc"
	forecasts_service "github.com/r0mbeg/TramCast/backend/internal/features/forecasts/service"
	routes_sqlc "github.com/r0mbeg/TramCast/backend/internal/features/routes/repository/postgres/sqlc"
)

// waitInterval is the pause between two reads of the jobs with -wait.
const waitInterval = 2 * time.Second

type options struct {
	envFile string
	// version is not Valid without -version: the active version is used.
	version pgtype.UUID
	dryRun  bool
	wait    bool
}

// jobQueue is the part of *forecasts_service.Queue the command uses.
type jobQueue interface {
	ActiveVersion(ctx context.Context) (forecasts_sqlc.ForecastVersion, error)
	Version(ctx context.Context, id pgtype.UUID) (forecasts_sqlc.ForecastVersion, error)
	ForecastRoutes(ctx context.Context) ([]routes_sqlc.Route, error)
	Admit(ctx context.Context, versionID pgtype.UUID, routeID int64) (forecasts_sqlc.PredictionJob, bool, error)
	Lookup(ctx context.Context, versionID pgtype.UUID, routeID int64) (forecasts_sqlc.PredictionJob, bool, error)
}

// routeJob is the last read state of the job of one route.
type routeJob struct {
	routeID     int64
	routeNumber int16
	job         forecasts_sqlc.PredictionJob
}

func main() {
	var o options
	flag.StringVar(&o.envFile, "env-file", "", "optional dotenv file; process environment takes priority")
	flag.Func("version", "forecast version ID, a UUID; default: the active version", func(value string) error {
		id, err := forecasts_service.ParseUUID(value)
		o.version = id
		return err
	})
	flag.BoolVar(&o.dryRun, "dry-run", false, "report the job of every forecast route without writing")
	flag.BoolVar(&o.wait, "wait", false, "read the jobs every 2 s until each has succeeded or failed; Ctrl+C stops waiting")
	flag.Parse()
	// Flag parsing stops at the first positional argument, so a stray value
	// would silently drop the flags after it, including -dry-run.
	if flag.NArg() > 0 {
		slog.Error("unexpected arguments; pass the version with -version", "args", flag.Args())
		os.Exit(2)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	if err := run(ctx, o); err != nil {
		slog.Error("prediction job enqueue failed", "error", err)
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

	pool, err := core_postgres.NewPool(ctx, cfg.Postgres)
	if err != nil {
		return err
	}
	defer pool.Close()

	limits := cfg.PredictionJobs
	queue := forecasts_service.NewQueue(pool, forecasts_service.JobPolicy{
		QueueCapacity: limits.QueueCapacity,
		MaxAttempts:   limits.MaxAttempts,
		LeaseDuration: limits.LeaseDuration,
		RetryDelay:    limits.RetryDelay,
	})
	return enqueue(ctx, log, queue, o)
}

// enqueue admits, or with -dry-run only reads, the job of every forecast
// route of the version and with -wait follows the jobs until they finish.
func enqueue(ctx context.Context, log *slog.Logger, queue jobQueue, o options) error {
	var (
		version forecasts_sqlc.ForecastVersion
		err     error
	)
	if o.version.Valid {
		version, err = queue.Version(ctx, o.version)
	} else {
		version, err = queue.ActiveVersion(ctx)
		if errors.Is(err, forecasts_service.ErrNoActiveVersion) {
			err = fmt.Errorf("%w; activate one with make forecast-version-register ACTIVATE=1 or pass -version", err)
		}
	}
	if err != nil {
		return err
	}
	log.Info("enqueueing prediction jobs",
		"forecast_version_id", version.ID.String(),
		"is_active", version.IsActive,
		"model_version", version.ModelVersion,
		"dataset_version", version.DatasetVersion,
		"forecast_from", moscow(version.ForecastFrom.Time),
		"forecast_to", moscow(version.ForecastTo.Time),
		"dry_run", o.dryRun, "wait", o.wait)
	if !version.IsActive {
		log.Warn("the version is not active: the worker takes jobs only while the ML server serves the active version, and a job of another version then fails for good")
	}

	routes, err := queue.ForecastRoutes(ctx)
	if err != nil {
		return err
	}
	if len(routes) == 0 {
		return errors.New("no route has forecasts enabled; import the catalog first")
	}

	var (
		jobs                            []routeJob
		created, notAdmitted, notQueued int
	)
	for _, route := range routes {
		if o.dryRun {
			job, found, err := queue.Lookup(ctx, version.ID, route.ID)
			if err != nil {
				return fmt.Errorf("route %d: %w", route.RouteNumber, err)
			}
			if !found {
				notQueued++
				log.Info("dry run: not queued", "route_number", route.RouteNumber)
				continue
			}
			jobs = append(jobs, routeJob{routeID: route.ID, routeNumber: route.RouteNumber, job: job})
			log.Info("dry run: job exists", jobAttrs(route.RouteNumber, job)...)
			continue
		}

		job, isNew, err := queue.Admit(ctx, version.ID, route.ID)
		if errors.Is(err, forecasts_service.ErrQueueFull) {
			notAdmitted++
			log.Warn("route not admitted", "route_number", route.RouteNumber, "error", err)
			continue
		}
		if err != nil {
			return fmt.Errorf("admit route %d: %w", route.RouteNumber, err)
		}
		if isNew {
			created++
		}
		jobs = append(jobs, routeJob{routeID: route.ID, routeNumber: route.RouteNumber, job: job})
		log.Info("prediction job admitted", append(jobAttrs(route.RouteNumber, job), "created", isNew)...)
	}

	var waitErr error
	if o.wait {
		waitErr = waitForJobs(ctx, log, queue, version.ID, jobs)
	}

	statuses := map[string]int{}
	for _, j := range jobs {
		statuses[j.job.Status]++
	}
	message := "prediction jobs enqueued"
	summary := []any{"forecast_version_id", version.ID.String(), "routes", len(routes)}
	if o.dryRun {
		message = "dry run: nothing written"
		summary = append(summary, "found", len(jobs), "not_queued", notQueued)
	} else {
		summary = append(summary, "created", created, "existing", len(jobs)-created, "not_admitted", notAdmitted)
	}
	log.Info(message, append(summary,
		"queued", statuses["queued"], "running", statuses["running"],
		"succeeded", statuses["succeeded"], "failed", statuses["failed"])...)

	switch {
	case waitErr != nil:
		return waitErr
	case notAdmitted > 0:
		return fmt.Errorf("%d route(s) not admitted: the queue is full; run again when jobs finish or raise PREDICTION_JOB_QUEUE_CAPACITY", notAdmitted)
	case o.wait && statuses["failed"] > 0:
		return fmt.Errorf("%d prediction job(s) failed", statuses["failed"])
	}
	return nil
}

// waitForJobs reads the unfinished jobs every waitInterval and logs each
// change until every job has succeeded or failed. A read error is logged and
// retried on the next round; ending ctx stops waiting with an error, while the
// backend worker keeps processing the jobs.
func waitForJobs(ctx context.Context, log *slog.Logger, queue jobQueue, versionID pgtype.UUID, jobs []routeJob) error {
	ticker := time.NewTicker(waitInterval)
	defer ticker.Stop()
	for {
		unfinished := 0
		for _, j := range jobs {
			if !finished(j.job) {
				unfinished++
			}
		}
		if unfinished == 0 {
			return nil
		}
		select {
		case <-ctx.Done():
			return fmt.Errorf("stopped waiting with %d prediction job(s) unfinished; the backend worker keeps processing them", unfinished)
		case <-ticker.C:
		}

		for i := range jobs {
			j := &jobs[i]
			if finished(j.job) {
				continue
			}
			job, found, err := queue.Lookup(ctx, versionID, j.routeID)
			if err != nil {
				if ctx.Err() == nil {
					log.Warn("cannot read the prediction jobs; retrying", "error", err)
				}
				break
			}
			if !found {
				return fmt.Errorf("the prediction job of route %d disappeared", j.routeNumber)
			}
			if job.Status != j.job.Status || job.AttemptCount != j.job.AttemptCount {
				log.Info("prediction job changed", jobAttrs(j.routeNumber, job)...)
			}
			j.job = job
		}
	}
}

func finished(job forecasts_sqlc.PredictionJob) bool {
	return job.Status == "succeeded" || job.Status == "failed"
}

func jobAttrs(routeNumber int16, job forecasts_sqlc.PredictionJob) []any {
	return []any{
		"route_number", routeNumber,
		"job_id", job.ID.String(),
		"status", job.Status,
		"attempt_count", job.AttemptCount,
		"last_error_code", job.LastErrorCode.String,
	}
}

// moscow formats t in the forecast zone, whatever location pgx returns.
func moscow(t time.Time) string {
	return t.In(core_domain.Moscow).Format(time.RFC3339)
}
