// Command import-catalog loads the reference workbook (CATALOG_FILE), together
// with the OpenStreetMap snapshot of the target routes the workbook lacks
// (CATALOG_OSM_FILE), into routes, stops and their positions.
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

	core_config "github.com/r0mbeg/TramCast/backend/internal/core/config"
	core_logger "github.com/r0mbeg/TramCast/backend/internal/core/logger"
	core_postgres "github.com/r0mbeg/TramCast/backend/internal/core/repository/postgres"
	catalog_osm_repository "github.com/r0mbeg/TramCast/backend/internal/features/catalog/repository/osm"
	catalog_xlsx_repository "github.com/r0mbeg/TramCast/backend/internal/features/catalog/repository/xlsx"
	catalog_service "github.com/r0mbeg/TramCast/backend/internal/features/catalog/service"
)

// maxLoggedProblems keeps the output readable for a badly broken source.
const maxLoggedProblems = 50

func main() {
	envFile := flag.String("env-file", "", "optional dotenv file; process environment takes priority")
	file := flag.String("file", "", "workbook path; overrides CATALOG_FILE for this run")
	osmFile := flag.String("osm-file", "", "OSM snapshot path; overrides CATALOG_OSM_FILE for this run")
	dryRun := flag.Bool("dry-run", false, "validate and run the import, then roll it back")
	ifEmpty := flag.Bool("if-empty", false, "import only when the catalog tables are empty; otherwise skip without reading sources")
	flag.Parse()
	// Flag parsing stops at the first positional argument, so a stray path would
	// silently drop the flags after it, including -dry-run.
	if flag.NArg() > 0 {
		slog.Error("unexpected arguments; pass the paths with -file and -osm-file", "args", flag.Args())
		os.Exit(2)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	if err := run(ctx, *envFile, *file, *osmFile, *dryRun, *ifEmpty); err != nil {
		slog.Error("catalog import failed", "error", err)
		os.Exit(1)
	}
}

func run(ctx context.Context, envFile, file, osmFile string, dryRun, ifEmpty bool) error {
	cfg, err := core_config.Load(envFile)
	if err != nil {
		return fmt.Errorf("load configuration: %w", err)
	}
	log := core_logger.New(cfg.Logger, os.Stdout)
	slog.SetDefault(log)

	var sources catalog_service.Sources
	if sources.Workbook, err = pathOverride(cfg.Catalog.File, file, "-file"); err != nil {
		return err
	}
	if sources.OSM, err = pathOverride(cfg.Catalog.OSMFile, osmFile, "-osm-file"); err != nil {
		return err
	}
	log.Info("importing catalog", "file", sources.Workbook, "osm_file", sources.OSM, "dry_run", dryRun, "if_empty", ifEmpty)

	pool, err := core_postgres.NewPool(ctx, cfg.Postgres)
	if err != nil {
		return err
	}
	defer pool.Close()

	service := catalog_service.NewService(catalog_xlsx_repository.NewRepository(), catalog_osm_repository.New(), pool)
	importCatalog := service.Import
	if ifEmpty {
		importCatalog = service.ImportIfEmpty
	}
	report, err := importCatalog(ctx, sources, dryRun)
	var validation *catalog_service.ValidationError
	if errors.As(err, &validation) {
		for i, issue := range validation.Issues {
			if i == maxLoggedProblems {
				log.Error("more catalog problems not shown", "count", len(validation.Issues)-maxLoggedProblems)
				break
			}
			log.Error("catalog problem", "location", issue.Location(), "problem", issue.Message)
		}
		return fmt.Errorf("catalog has %d problem(s); nothing was imported", len(validation.Issues))
	}
	if err != nil {
		return err
	}
	if report.Skipped {
		log.Info("catalog already contains data; import skipped")
		return nil
	}

	for _, warning := range report.Warnings {
		log.Warn("catalog warning", "warning", warning)
	}
	message := "catalog imported"
	if dryRun {
		message = "catalog checked; dry run rolled back"
	}
	log.Info(message,
		"workbook_routes", report.WorkbookRoutes,
		"workbook_stops", report.WorkbookStops,
		"workbook_positions", report.WorkbookPositions,
		"osm_snapshot", report.OSMSnapshot,
		"osm_routes", report.OSMRoutes,
		"osm_stops", report.OSMStops,
		"osm_positions", report.OSMPositions,
		"routes_created_without_geography", report.RoutesCreatedWithoutGeography,
		"routes_total", report.RoutesTotal,
		"forecast_enabled_routes", report.ForecastEnabledRoutes,
		"forecast_flags_changed", report.ForecastFlagsChanged,
		"positions_cleared", report.PositionsCleared,
		"stops_deleted", report.StopsDeleted,
	)
	return nil
}

// pathOverride returns the configured path, already absolute, or the flag
// value made absolute when the flag is set.
func pathOverride(configured, flagValue, flagName string) (string, error) {
	if flagValue == "" {
		return configured, nil
	}
	path, err := filepath.Abs(flagValue)
	if err != nil {
		return "", fmt.Errorf("resolve %s: %w", flagName, err)
	}
	return path, nil
}
