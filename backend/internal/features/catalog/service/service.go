// Package catalog_service imports the reference workbook, together with the
// OpenStreetMap snapshot of the routes it lacks, into the route, stop and
// position tables.
package catalog_service

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgtype"

	catalog_osm_repository "github.com/r0mbeg/TramCast/backend/internal/features/catalog/repository/osm"
	catalog_sqlc "github.com/r0mbeg/TramCast/backend/internal/features/catalog/repository/postgres/sqlc"
	catalog_xlsx_repository "github.com/r0mbeg/TramCast/backend/internal/features/catalog/repository/xlsx"
	routes_sqlc "github.com/r0mbeg/TramCast/backend/internal/features/routes/repository/postgres/sqlc"
	stops_sqlc "github.com/r0mbeg/TramCast/backend/internal/features/stops/repository/postgres/sqlc"
)

// TargetRouteNumbers are the contest routes. Every import enables forecasts
// exactly for them; targets absent from both catalog sources are created
// without geography.
var TargetRouteNumbers = []int16{1, 5, 7, 11, 12, 17, 25, 26, 28, 50}

// OSMRoutes are the target routes taken from the OSM snapshot: route number →
// its two PTv2 route relations, indexed by direction_id. Direction 0 follows
// the terminus order of the route_master name. The snapshot query
// (data/osm/tram_routes.overpassql at the repository root) lists the same
// relations.
var OSMRoutes = map[int16][2]int64{
	17: {540033, 540139},
	25: {3186264, 3186265},
	26: {1689026, 1689064},
	28: {3184023, 3184022},
	50: {1538169, 1538170},
}

type WorkbookReader interface {
	ReadSheets(ctx context.Context, path string, specs []catalog_xlsx_repository.SheetSpec) (map[string]catalog_xlsx_repository.Sheet, error)
}

// OSMReader is satisfied by *catalog_osm_repository.Repository.
type OSMReader interface {
	ReadSnapshot(ctx context.Context, path string) (catalog_osm_repository.Snapshot, error)
}

// TxBeginner is satisfied by *pgxpool.Pool.
type TxBeginner interface {
	Begin(ctx context.Context) (pgx.Tx, error)
}

// routeQueries, stopQueries and catalogQueries are the parts of the generated
// sqlc packages used by the import; their *Queries types satisfy them directly.
type routeQueries interface {
	GetRouteBySourceID(ctx context.Context, sourceRouteID string) (routes_sqlc.Route, error)
	GetRouteByNumber(ctx context.Context, routeNumber int16) (routes_sqlc.Route, error)
	UpsertRoute(ctx context.Context, arg routes_sqlc.UpsertRouteParams) (routes_sqlc.Route, error)
	EnsureRoute(ctx context.Context, routeNumber int16) (int64, error)
	ListRoutes(ctx context.Context) ([]routes_sqlc.Route, error)
	SetRouteForecastEnabled(ctx context.Context, arg routes_sqlc.SetRouteForecastEnabledParams) (int64, error)
	GetRouteForUpdate(ctx context.Context, routeID int64) (routes_sqlc.Route, error)
	DeleteRouteStops(ctx context.Context, routeID int64) (int64, error)
	InsertRouteStop(ctx context.Context, arg routes_sqlc.InsertRouteStopParams) error
}

type stopQueries interface {
	UpsertStop(ctx context.Context, arg stops_sqlc.UpsertStopParams) (stops_sqlc.Stop, error)
}

type catalogQueries interface {
	LockCatalogImport(ctx context.Context) error
	DeleteUnusedStopsNotInCatalog(ctx context.Context, sourceStopIDs []string) (int64, error)
}

// queries groups the storage of one import transaction.
type queries struct {
	routes  routeQueries
	stops   stopQueries
	catalog catalogQueries
}

func newQueries(tx pgx.Tx) queries {
	return queries{routes: routes_sqlc.New(tx), stops: stops_sqlc.New(tx), catalog: catalog_sqlc.New(tx)}
}

type Service struct {
	workbook WorkbookReader
	osm      OSMReader
	db       TxBeginner
}

func NewService(workbook WorkbookReader, osm OSMReader, db TxBeginner) *Service {
	return &Service{workbook: workbook, osm: osm, db: db}
}

// Sources are the paths of the two catalog sources.
type Sources struct {
	// Workbook is the reference workbook (CATALOG_FILE).
	Workbook string
	// OSM is the OpenStreetMap snapshot as Overpass JSON (CATALOG_OSM_FILE).
	OSM string
}

// Report summarizes an import. Counts other than the source ones refer to the
// database after the import.
type Report struct {
	DryRun bool
	// Workbook* count the validated workbook.
	WorkbookRoutes    int
	WorkbookStops     int
	WorkbookPositions int
	// OSMSnapshot is the state of the OSM database in the snapshot
	// (timestamp_osm_base). OSM* count the snapshot data written, without the
	// routes the workbook takes priority for.
	OSMSnapshot  string
	OSMRoutes    int
	OSMStops     int
	OSMPositions int
	// RoutesCreatedWithoutGeography counts target routes added in this run.
	RoutesCreatedWithoutGeography int
	RoutesTotal                   int
	ForecastEnabledRoutes         int
	ForecastFlagsChanged          int
	// PositionsCleared counts old positions deleted before the new set.
	PositionsCleared int64
	StopsDeleted     int64
	Warnings         []string
}

// Import validates the whole workbook and the whole OSM snapshot, merges them
// and treats the result as a full snapshot of the geography in one
// transaction. With dryRun every write runs and is then rolled back, so the
// report shows the effect without changing data.
func (s *Service) Import(ctx context.Context, sources Sources, dryRun bool) (Report, error) {
	sheets, err := s.workbook.ReadSheets(ctx, sources.Workbook, sheetSpecs)
	if err != nil {
		return Report{}, fmt.Errorf("read catalog workbook: %w", err)
	}
	workbook, issues := parseWorkbook(sheets)
	snapshot, err := s.osm.ReadSnapshot(ctx, sources.OSM)
	if err != nil {
		return Report{}, fmt.Errorf("read OSM snapshot: %w", err)
	}
	osm, osmIssues := parseOSM(snapshot, OSMRoutes)
	// Problems of both sources are reported together, before any write.
	if issues = append(issues, osmIssues...); len(issues) > 0 {
		return Report{}, &ValidationError{Issues: issues}
	}
	catalog, warnings := mergeCatalogs(workbook, osm)

	tx, err := s.db.Begin(ctx)
	if err != nil {
		return Report{}, fmt.Errorf("begin catalog import: %w", err)
	}
	// After Commit this is a no-op; otherwise it undoes every write.
	defer func() { _ = tx.Rollback(context.WithoutCancel(ctx)) }()

	report := newReport(dryRun, workbook, catalog, snapshot.BaseTimestamp, warnings)
	if err := write(ctx, newQueries(tx), catalog, &report); err != nil {
		return Report{}, err
	}
	if dryRun {
		return report, nil
	}
	if err := tx.Commit(ctx); err != nil {
		return Report{}, fmt.Errorf("commit catalog import: %w", err)
	}
	return report, nil
}

// newReport fills the source counts of a report. The sources use disjoint
// identifiers and mergeCatalogs appends the OSM data kept to the workbook, so
// the OSM counts are the merged sizes minus the workbook ones. The merge
// warnings come first; write appends its own.
func newReport(dryRun bool, workbook, merged Catalog, osmSnapshot string, mergeWarnings []string) Report {
	return Report{
		DryRun:            dryRun,
		WorkbookRoutes:    len(workbook.Routes),
		WorkbookStops:     len(workbook.Stops),
		WorkbookPositions: len(workbook.Positions),
		OSMSnapshot:       osmSnapshot,
		OSMRoutes:         len(merged.Routes) - len(workbook.Routes),
		OSMStops:          len(merged.Stops) - len(workbook.Stops),
		OSMPositions:      len(merged.Positions) - len(workbook.Positions),
		Warnings:          slices.Clone(mergeWarnings),
	}
}

func write(ctx context.Context, q queries, catalog Catalog, report *Report) error {
	if err := q.catalog.LockCatalogImport(ctx); err != nil {
		return fmt.Errorf("lock catalog import: %w", err)
	}
	if err := upsertRoutes(ctx, q.routes, catalog.Routes, report); err != nil {
		return err
	}

	inCatalog := make(map[int16]bool, len(catalog.Routes))
	for _, route := range catalog.Routes {
		inCatalog[route.Number] = true
	}
	for _, number := range TargetRouteNumbers {
		if inCatalog[number] {
			continue
		}
		created, err := q.routes.EnsureRoute(ctx, number)
		if err != nil {
			return fmt.Errorf("ensure target route %d: %w", number, err)
		}
		report.RoutesCreatedWithoutGeography += int(created)
	}

	// All routes, including ones missing from both sources, get the target
	// flag and a replaced (possibly empty) set of positions.
	routes, err := q.routes.ListRoutes(ctx)
	if err != nil {
		return fmt.Errorf("list routes: %w", err)
	}
	report.RoutesTotal = len(routes)
	if err := setForecastFlags(ctx, q.routes, routes, report); err != nil {
		return err
	}

	stopIDs, err := upsertStops(ctx, q.stops, catalog.Stops)
	if err != nil {
		return err
	}
	if err := replacePositions(ctx, q.routes, routes, stopIDs, catalog.Positions, report); err != nil {
		return err
	}

	sourceStopIDs := make([]string, len(catalog.Stops))
	for i, stop := range catalog.Stops {
		sourceStopIDs[i] = stop.SourceID
	}
	deleted, err := q.catalog.DeleteUnusedStopsNotInCatalog(ctx, sourceStopIDs)
	if err != nil {
		return fmt.Errorf("delete stops missing from the catalog sources: %w", err)
	}
	report.StopsDeleted = deleted
	return nil
}

// upsertRoutes keeps internal IDs. A route_id that already belongs to another
// route number signals renumbering and stops the import; a changed route_id of
// the same number is updated with a warning. Every route is checked before the
// first write: otherwise an earlier upsert could release a route_id and hide a
// later route's conflict.
func upsertRoutes(ctx context.Context, queries routeQueries, routes []Route, report *Report) error {
	sorted := slices.Clone(routes)
	slices.SortFunc(sorted, func(a, b Route) int { return int(a.Number) - int(b.Number) })
	for _, route := range sorted {
		owner, err := queries.GetRouteBySourceID(ctx, route.SourceID)
		switch {
		case err == nil && owner.RouteNumber != route.Number:
			return fmt.Errorf("route %d: route_id %s already belongs to route %d in the database; renumbered routes need a manual fix",
				route.Number, route.SourceID, owner.RouteNumber)
		case err != nil && !errors.Is(err, pgx.ErrNoRows):
			return fmt.Errorf("find route by route_id %s: %w", route.SourceID, err)
		}

		current, err := queries.GetRouteByNumber(ctx, route.Number)
		switch {
		case err == nil && current.SourceRouteID.Valid && current.SourceRouteID.String != route.SourceID:
			report.Warnings = append(report.Warnings, fmt.Sprintf("route %d: route_id changes from %s to %s",
				route.Number, current.SourceRouteID.String, route.SourceID))
		case err != nil && !errors.Is(err, pgx.ErrNoRows):
			return fmt.Errorf("find route %d: %w", route.Number, err)
		}
	}

	for _, route := range sorted {
		if _, err := queries.UpsertRoute(ctx, routes_sqlc.UpsertRouteParams{
			RouteNumber:   route.Number,
			Name:          pgtype.Text{String: route.Name, Valid: route.Name != ""},
			SourceRouteID: pgtype.Text{String: route.SourceID, Valid: true},
		}); err != nil {
			return fmt.Errorf("upsert route %d: %w", route.Number, err)
		}
	}
	return nil
}

func setForecastFlags(ctx context.Context, queries routeQueries, routes []routes_sqlc.Route, report *Report) error {
	for _, route := range routes {
		enabled := slices.Contains(TargetRouteNumbers, route.RouteNumber)
		if enabled {
			report.ForecastEnabledRoutes++
		}
		if route.ForecastEnabled == enabled {
			continue
		}
		changed, err := queries.SetRouteForecastEnabled(ctx, routes_sqlc.SetRouteForecastEnabledParams{
			ForecastEnabled: enabled,
			RouteID:         route.ID,
		})
		if err != nil {
			return fmt.Errorf("set forecast flag of route %d: %w", route.RouteNumber, err)
		}
		if changed != 1 {
			return fmt.Errorf("set forecast flag of route %d: %d rows changed, want 1", route.RouteNumber, changed)
		}
		report.ForecastFlagsChanged++
	}
	return nil
}

func upsertStops(ctx context.Context, queries stopQueries, stops []Stop) (map[string]int64, error) {
	sorted := slices.Clone(stops)
	slices.SortFunc(sorted, func(a, b Stop) int { return strings.Compare(a.SourceID, b.SourceID) })
	ids := make(map[string]int64, len(sorted))
	for _, stop := range sorted {
		saved, err := queries.UpsertStop(ctx, stops_sqlc.UpsertStopParams{
			SourceStopID: stop.SourceID,
			Name:         stop.Name,
			Latitude:     stop.Latitude,
			Longitude:    stop.Longitude,
		})
		if err != nil {
			return nil, fmt.Errorf("upsert stop %s: %w", stop.SourceID, err)
		}
		ids[stop.SourceID] = saved.ID
	}
	return ids, nil
}

// replacePositions locks each route, deletes its positions and inserts the
// catalog's set. Routes missing from both sources end up without positions.
func replacePositions(ctx context.Context, queries routeQueries, routes []routes_sqlc.Route,
	stopIDs map[string]int64, positions []Position, report *Report) error {
	byRoute := make(map[string][]Position)
	for _, position := range positions {
		byRoute[position.RouteSourceID] = append(byRoute[position.RouteSourceID], position)
	}
	inserted := 0
	for _, route := range routes {
		if _, err := queries.GetRouteForUpdate(ctx, route.ID); err != nil {
			return fmt.Errorf("lock route %d: %w", route.RouteNumber, err)
		}
		cleared, err := queries.DeleteRouteStops(ctx, route.ID)
		if err != nil {
			return fmt.Errorf("delete positions of route %d: %w", route.RouteNumber, err)
		}
		report.PositionsCleared += cleared
		if !route.SourceRouteID.Valid {
			continue
		}
		for _, position := range byRoute[route.SourceRouteID.String] {
			if err := queries.InsertRouteStop(ctx, routes_sqlc.InsertRouteStopParams{
				RouteID:      route.ID,
				PatternKey:   position.PatternKey,
				DirectionID:  position.DirectionID,
				StopSequence: position.StopSequence,
				StopID:       stopIDs[position.StopSourceID],
			}); err != nil {
				return fmt.Errorf("insert position %d of pattern %s on route %d: %w",
					position.StopSequence, position.PatternKey, route.RouteNumber, err)
			}
			inserted++
		}
	}
	if inserted != len(positions) {
		return fmt.Errorf("inserted %d positions, want %d", inserted, len(positions))
	}
	return nil
}
