package catalog_service

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgtype"

	routes_sqlc "github.com/r0mbeg/TramCast/backend/internal/features/routes/repository/postgres/sqlc"
	stops_sqlc "github.com/r0mbeg/TramCast/backend/internal/features/stops/repository/postgres/sqlc"
)

// fakeStore mirrors the SQL semantics used by the import: upserts keyed by
// route number and stop source ID, UNIQUE source_route_id, the position
// primary key and deletion of unreferenced stops.
type fakeStore struct {
	routes    []routes_sqlc.Route
	stops     []stops_sqlc.Stop
	positions []routes_sqlc.InsertRouteStopParams
	nextID    int64
	locked    bool
}

func (s *fakeStore) queries() queries {
	return queries{routes: s, stops: s, catalog: s}
}

func (s *fakeStore) id() int64 {
	s.nextID++
	return s.nextID
}

func (s *fakeStore) route(match func(routes_sqlc.Route) bool) (*routes_sqlc.Route, error) {
	for i := range s.routes {
		if match(s.routes[i]) {
			return &s.routes[i], nil
		}
	}
	return nil, pgx.ErrNoRows
}

func (s *fakeStore) GetRouteBySourceID(_ context.Context, sourceRouteID string) (routes_sqlc.Route, error) {
	route, err := s.route(func(r routes_sqlc.Route) bool {
		return r.SourceRouteID.Valid && r.SourceRouteID.String == sourceRouteID
	})
	if err != nil {
		return routes_sqlc.Route{}, err
	}
	return *route, nil
}

func (s *fakeStore) GetRouteByNumber(_ context.Context, routeNumber int16) (routes_sqlc.Route, error) {
	route, err := s.route(func(r routes_sqlc.Route) bool { return r.RouteNumber == routeNumber })
	if err != nil {
		return routes_sqlc.Route{}, err
	}
	return *route, nil
}

func (s *fakeStore) UpsertRoute(_ context.Context, arg routes_sqlc.UpsertRouteParams) (routes_sqlc.Route, error) {
	if arg.SourceRouteID.Valid {
		if owner, err := s.route(func(r routes_sqlc.Route) bool {
			return r.SourceRouteID == arg.SourceRouteID && r.RouteNumber != arg.RouteNumber
		}); err == nil {
			return routes_sqlc.Route{}, fmt.Errorf("unique violation: source_route_id %s of route %d", arg.SourceRouteID.String, owner.RouteNumber)
		}
	}
	route, err := s.route(func(r routes_sqlc.Route) bool { return r.RouteNumber == arg.RouteNumber })
	if err != nil {
		s.routes = append(s.routes, routes_sqlc.Route{ID: s.id(), RouteNumber: arg.RouteNumber, Name: arg.Name, SourceRouteID: arg.SourceRouteID})
		return s.routes[len(s.routes)-1], nil
	}
	if arg.Name.Valid {
		route.Name = arg.Name
	}
	if arg.SourceRouteID.Valid {
		route.SourceRouteID = arg.SourceRouteID
	}
	return *route, nil
}

func (s *fakeStore) EnsureRoute(_ context.Context, routeNumber int16) (int64, error) {
	if _, err := s.route(func(r routes_sqlc.Route) bool { return r.RouteNumber == routeNumber }); err == nil {
		return 0, nil
	}
	s.routes = append(s.routes, routes_sqlc.Route{ID: s.id(), RouteNumber: routeNumber})
	return 1, nil
}

func (s *fakeStore) ListRoutes(context.Context) ([]routes_sqlc.Route, error) {
	routes := slices.Clone(s.routes)
	slices.SortFunc(routes, func(a, b routes_sqlc.Route) int { return int(a.RouteNumber) - int(b.RouteNumber) })
	return routes, nil
}

func (s *fakeStore) SetRouteForecastEnabled(_ context.Context, arg routes_sqlc.SetRouteForecastEnabledParams) (int64, error) {
	route, err := s.route(func(r routes_sqlc.Route) bool { return r.ID == arg.RouteID })
	if err != nil {
		return 0, nil
	}
	route.ForecastEnabled = arg.ForecastEnabled
	return 1, nil
}

func (s *fakeStore) GetRouteForUpdate(_ context.Context, routeID int64) (routes_sqlc.Route, error) {
	route, err := s.route(func(r routes_sqlc.Route) bool { return r.ID == routeID })
	if err != nil {
		return routes_sqlc.Route{}, err
	}
	return *route, nil
}

func (s *fakeStore) DeleteRouteStops(_ context.Context, routeID int64) (int64, error) {
	before := len(s.positions)
	s.positions = slices.DeleteFunc(s.positions, func(p routes_sqlc.InsertRouteStopParams) bool { return p.RouteID == routeID })
	return int64(before - len(s.positions)), nil
}

func (s *fakeStore) InsertRouteStop(_ context.Context, arg routes_sqlc.InsertRouteStopParams) error {
	for _, p := range s.positions {
		if p.RouteID == arg.RouteID && p.PatternKey == arg.PatternKey && p.StopSequence == arg.StopSequence {
			return fmt.Errorf("primary key violation: %+v", arg)
		}
	}
	s.positions = append(s.positions, arg)
	return nil
}

func (s *fakeStore) UpsertStop(_ context.Context, arg stops_sqlc.UpsertStopParams) (stops_sqlc.Stop, error) {
	for i := range s.stops {
		if s.stops[i].SourceStopID == arg.SourceStopID {
			s.stops[i].Name, s.stops[i].Latitude, s.stops[i].Longitude = arg.Name, arg.Latitude, arg.Longitude
			return s.stops[i], nil
		}
	}
	s.stops = append(s.stops, stops_sqlc.Stop{ID: s.id(), SourceStopID: arg.SourceStopID, Name: arg.Name, Latitude: arg.Latitude, Longitude: arg.Longitude})
	return s.stops[len(s.stops)-1], nil
}

func (s *fakeStore) LockCatalogImport(context.Context) error {
	s.locked = true
	return nil
}

func (s *fakeStore) CatalogHasData(context.Context) (bool, error) {
	return len(s.routes) > 0 || len(s.stops) > 0 || len(s.positions) > 0, nil
}

func (s *fakeStore) DeleteUnusedStopsNotInCatalog(_ context.Context, sourceStopIDs []string) (int64, error) {
	before := len(s.stops)
	s.stops = slices.DeleteFunc(s.stops, func(stop stops_sqlc.Stop) bool {
		referenced := slices.ContainsFunc(s.positions, func(p routes_sqlc.InsertRouteStopParams) bool { return p.StopID == stop.ID })
		return !slices.Contains(sourceStopIDs, stop.SourceStopID) && !referenced
	})
	return int64(before - len(s.stops)), nil
}

func (s *fakeStore) snapshot() string {
	return fmt.Sprintf("%+v|%+v|%+v", s.routes, s.stops, s.positions)
}

func text(value string) pgtype.Text {
	return pgtype.Text{String: value, Valid: true}
}

func validCatalog(t *testing.T) Catalog {
	t.Helper()
	catalog, issues := parseWorkbook(toSheets(validSheets()))
	if len(issues) > 0 {
		t.Fatalf("issues = %+v", issues)
	}
	return catalog
}

func runWrite(t *testing.T, store *fakeStore, catalog Catalog) Report {
	t.Helper()
	var report Report
	if err := write(context.Background(), store.queries(), catalog, &report); err != nil {
		t.Fatal(err)
	}
	return report
}

func TestWriteFreshCatalog(t *testing.T) {
	store := &fakeStore{}
	report := runWrite(t, store, validCatalog(t))

	if !store.locked {
		t.Error("the import must take the catalog lock")
	}
	// Routes 1 and 5 come from the workbook; the other eight targets are created.
	if report.RoutesCreatedWithoutGeography != 8 || report.RoutesTotal != 10 ||
		report.ForecastEnabledRoutes != 10 || report.ForecastFlagsChanged != 10 ||
		report.PositionsCleared != 0 || report.StopsDeleted != 0 || len(report.Warnings) != 0 {
		t.Fatalf("report = %+v", report)
	}
	route5, _ := store.GetRouteByNumber(context.Background(), 5)
	if route5.Name.Valid || route5.SourceRouteID != text("3736") {
		t.Errorf("route 5 = %+v, want no name and route_id 3736", route5)
	}
	route17, _ := store.GetRouteByNumber(context.Background(), 17)
	if route17.Name.Valid || route17.SourceRouteID.Valid || !route17.ForecastEnabled {
		t.Errorf("route 17 = %+v, want an enabled route without geography", route17)
	}
	if len(store.stops) != 3 || len(store.positions) != 4 {
		t.Fatalf("stops = %d, positions = %d", len(store.stops), len(store.positions))
	}
	stop7879, _ := store.stopBySource("7879")
	route1, _ := store.GetRouteByNumber(context.Background(), 1)
	want := routes_sqlc.InsertRouteStopParams{RouteID: route1.ID, PatternKey: "2040921", DirectionID: 1, StopSequence: 1, StopID: stop7879.ID}
	if !slices.Contains(store.positions, want) {
		t.Errorf("positions = %+v, want %+v", store.positions, want)
	}
}

func (s *fakeStore) stopBySource(sourceID string) (stops_sqlc.Stop, bool) {
	for _, stop := range s.stops {
		if stop.SourceStopID == sourceID {
			return stop, true
		}
	}
	return stops_sqlc.Stop{}, false
}

func TestWriteTwiceKeepsIDsAndData(t *testing.T) {
	store := &fakeStore{}
	runWrite(t, store, validCatalog(t))
	before := store.snapshot()
	report := runWrite(t, store, validCatalog(t))
	if report.RoutesCreatedWithoutGeography != 0 || report.ForecastFlagsChanged != 0 || report.PositionsCleared != 4 || report.StopsDeleted != 0 {
		t.Fatalf("second report = %+v", report)
	}
	if after := store.snapshot(); after != before {
		t.Fatalf("second import changed data:\nbefore %s\nafter  %s", before, after)
	}
}

func TestWriteRejectsRenumberingInBothOrders(t *testing.T) {
	for _, tt := range []struct {
		name     string
		existing []routes_sqlc.Route
		workbook []Route
		want     string
	}{
		{
			// Processing route 1 first used to release 4450 before route 2 was checked.
			name:     "route_id moves to a higher number",
			existing: []routes_sqlc.Route{{ID: 1, RouteNumber: 1, SourceRouteID: text("4450")}},
			workbook: []Route{{Number: 1, SourceID: "9999"}, {Number: 2, SourceID: "4450"}},
			want:     "route 2: route_id 4450 already belongs to route 1",
		},
		{
			name:     "route_id moves to a lower number",
			existing: []routes_sqlc.Route{{ID: 1, RouteNumber: 2, SourceRouteID: text("4450")}},
			workbook: []Route{{Number: 1, SourceID: "4450"}, {Number: 2, SourceID: "9999"}},
			want:     "route 1: route_id 4450 already belongs to route 2",
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			store := &fakeStore{routes: slices.Clone(tt.existing), nextID: 10}
			before := store.snapshot()
			var report Report
			err := write(context.Background(), store.queries(), Catalog{Routes: tt.workbook}, &report)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error = %v, want %q", err, tt.want)
			}
			// Checks run before the first write, so nothing changed even without a rollback.
			if after := store.snapshot(); after != before {
				t.Fatalf("store changed before the conflict was detected:\nbefore %s\nafter  %s", before, after)
			}
		})
	}
}

func TestWriteWarnsAboutChangedRouteID(t *testing.T) {
	store := &fakeStore{routes: []routes_sqlc.Route{{ID: 1, RouteNumber: 1, SourceRouteID: text("1111")}}, nextID: 10}
	report := runWrite(t, store, validCatalog(t))
	if len(report.Warnings) != 1 || report.Warnings[0] != "route 1: route_id changes from 1111 to 4450" {
		t.Fatalf("warnings = %q", report.Warnings)
	}
	if route, _ := store.GetRouteByNumber(context.Background(), 1); route.ID != 1 || route.SourceRouteID != text("4450") {
		t.Fatalf("route 1 = %+v, want the same row with route_id 4450", route)
	}
}

func TestWriteTreatsWorkbookAsSnapshot(t *testing.T) {
	store := &fakeStore{
		routes: []routes_sqlc.Route{
			{ID: 1, RouteNumber: 99, SourceRouteID: text("5555"), ForecastEnabled: true},
			{ID: 2, RouteNumber: 3, SourceRouteID: text("4415"), ForecastEnabled: true},
		},
		stops: []stops_sqlc.Stop{
			{ID: 3, SourceStopID: "999998", Name: "Used only by route 99"},
			{ID: 4, SourceStopID: "999999", Name: "Unused"},
		},
		positions: []routes_sqlc.InsertRouteStopParams{{RouteID: 1, PatternKey: "x", StopSequence: 1, StopID: 3}},
		nextID:    10,
	}
	report := runWrite(t, store, validCatalog(t))

	route99, err := store.GetRouteByNumber(context.Background(), 99)
	if err != nil || route99.ForecastEnabled {
		t.Errorf("route 99 = %+v, %v; want the row kept with forecasts disabled", route99, err)
	}
	route3, _ := store.GetRouteByNumber(context.Background(), 3)
	if route3.ForecastEnabled {
		t.Error("a non-target route must lose its forecast flag")
	}
	if slices.ContainsFunc(store.positions, func(p routes_sqlc.InsertRouteStopParams) bool { return p.RouteID == route99.ID }) {
		t.Error("positions of a route missing from the workbook must be cleared")
	}
	for _, sourceID := range []string{"999998", "999999"} {
		if _, found := store.stopBySource(sourceID); found {
			t.Errorf("stop %s is missing from the workbook and unreferenced; it must be deleted", sourceID)
		}
	}
	if report.PositionsCleared != 1 || report.StopsDeleted != 2 || report.ForecastFlagsChanged != 12 {
		t.Fatalf("report = %+v", report)
	}
}

// mergedCatalog adds routes 17 and 25 of validSnapshot to a workbook catalog.
func mergedCatalog(t *testing.T, workbook Catalog) Catalog {
	t.Helper()
	catalog, warnings := mergeCatalogs(workbook, parseValidOSM(t))
	if len(warnings) != 0 {
		t.Fatalf("warnings = %q", warnings)
	}
	return catalog
}

func TestWriteMergedCatalog(t *testing.T) {
	store := &fakeStore{}
	report := runWrite(t, store, mergedCatalog(t, validCatalog(t)))

	// Routes 1 and 5 come from the workbook, 17 and 25 from OSM; six targets are created.
	if report.RoutesCreatedWithoutGeography != 6 || report.RoutesTotal != 10 || report.StopsDeleted != 0 || len(report.Warnings) != 0 {
		t.Fatalf("report = %+v", report)
	}
	route17, _ := store.GetRouteByNumber(context.Background(), 17)
	if route17.Name != text(`Усадьба Останкино - Метро "ВДНХ"`) || route17.SourceRouteID != text("osm:relation/1001") || !route17.ForecastEnabled {
		t.Errorf("route 17 = %+v, want the OSM name and route_master", route17)
	}
	if len(store.stops) != 3+6 || len(store.positions) != 4+10 {
		t.Fatalf("stops = %d, positions = %d", len(store.stops), len(store.positions))
	}
	// The terminus shared by both directions is one stop referenced twice.
	terminus, _ := store.stopBySource("osm:node/13725188937")
	for _, want := range []routes_sqlc.InsertRouteStopParams{
		{RouteID: route17.ID, PatternKey: "osm:relation/101", DirectionID: 0, StopSequence: 3, StopID: terminus.ID},
		{RouteID: route17.ID, PatternKey: "osm:relation/102", DirectionID: 1, StopSequence: 1, StopID: terminus.ID},
	} {
		if !slices.Contains(store.positions, want) {
			t.Errorf("positions = %+v, want %+v", store.positions, want)
		}
	}

	// A second run keeps IDs and data, including the OSM stops.
	before := store.snapshot()
	report = runWrite(t, store, mergedCatalog(t, validCatalog(t)))
	if report.RoutesCreatedWithoutGeography != 0 || report.PositionsCleared != 14 || report.StopsDeleted != 0 || len(report.Warnings) != 0 {
		t.Fatalf("second report = %+v", report)
	}
	if after := store.snapshot(); after != before {
		t.Fatalf("second import changed data:\nbefore %s\nafter  %s", before, after)
	}
}

// workbookWithRoute17 is validCatalog with route 17 and two of its positions,
// as if the organizers added the route to the workbook.
func workbookWithRoute17(t *testing.T) Catalog {
	t.Helper()
	values := validSheets()
	values[RoutesSheet] = append(values[RoutesSheet], routeRow("4417", "17", "Усадьба Останкино - Медведково"))
	values[PositionsSheet] = append(values[PositionsSheet],
		positionRow("4417", "17", "3000001", "0", "1", "2594"),
		positionRow("4417", "17", "3000001", "0", "2", "7879"))
	workbook, issues := parseWorkbook(toSheets(values))
	if len(issues) > 0 {
		t.Fatalf("issues = %+v", issues)
	}
	return workbook
}

func TestWriteMovesRouteFromOSMToWorkbook(t *testing.T) {
	store := &fakeStore{}
	runWrite(t, store, mergedCatalog(t, validCatalog(t)))
	before, _ := store.GetRouteByNumber(context.Background(), 17)

	// The organizers add route 17 to the workbook: it takes priority over OSM.
	catalog, warnings := mergeCatalogs(workbookWithRoute17(t), parseValidOSM(t))
	if len(warnings) != 1 || !strings.Contains(warnings[0], "OSM relations 101 and 102 are skipped") {
		t.Fatalf("merge warnings = %q", warnings)
	}
	report := runWrite(t, store, catalog)

	if len(report.Warnings) != 1 || report.Warnings[0] != "route 17: route_id changes from osm:relation/1001 to 4417" {
		t.Fatalf("warnings = %q", report.Warnings)
	}
	route17, _ := store.GetRouteByNumber(context.Background(), 17)
	if route17.ID != before.ID || route17.SourceRouteID != text("4417") || route17.Name != text("Усадьба Останкино - Медведково") {
		t.Fatalf("route 17 = %+v, want the same row with the workbook route_id and name", route17)
	}
	// Stops used only by the OSM route 17 are deleted; route 25 keeps the shared ones.
	if report.StopsDeleted != 3 {
		t.Fatalf("report = %+v, want 3 deleted stops", report)
	}
	for sourceID, want := range map[string]bool{
		"osm:node/2": false, "osm:node/13725188937": false, "osm:node/4": false,
		"osm:node/1": true, "osm:node/5": true, "osm:node/6": true,
	} {
		if _, found := store.stopBySource(sourceID); found != want {
			t.Errorf("stop %s present = %v, want %v", sourceID, found, want)
		}
	}
	if slices.ContainsFunc(store.positions, func(p routes_sqlc.InsertRouteStopParams) bool {
		return p.RouteID == route17.ID && strings.HasPrefix(p.PatternKey, "osm:")
	}) {
		t.Error("route 17 must lose its OSM positions")
	}
}

// TestNewReportCountsCommittedSnapshot checks the source counts that
// make import-catalog-dry-run shows for the committed snapshot.
func TestNewReportCountsCommittedSnapshot(t *testing.T) {
	snapshot, osm := parseCommittedOSM(t)
	for _, tt := range []struct {
		name                     string
		workbook                 Catalog
		routes, stops, positions int
		warnings                 []string
	}{
		{"all OSM routes kept", validCatalog(t), 5, 242, 257, nil},
		{"route 17 in the workbook", workbookWithRoute17(t), 4, 202, 205, []string{
			"route 17: the workbook has this route; OSM relations 540033 and 540139 are skipped",
		}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			merged, warnings := mergeCatalogs(tt.workbook, osm)
			report := newReport(true, tt.workbook, merged, snapshot.BaseTimestamp, warnings)
			if !slices.Equal(report.Warnings, tt.warnings) {
				t.Errorf("warnings = %q, want %q", report.Warnings, tt.warnings)
			}
			report.Warnings = nil
			want := Report{
				DryRun:            true,
				WorkbookRoutes:    len(tt.workbook.Routes),
				WorkbookStops:     len(tt.workbook.Stops),
				WorkbookPositions: len(tt.workbook.Positions),
				OSMSnapshot:       committedSnapshotBase,
				OSMRoutes:         tt.routes,
				OSMStops:          tt.stops,
				OSMPositions:      tt.positions,
			}
			if !reflect.DeepEqual(report, want) {
				t.Errorf("report = %+v\nwant %+v", report, want)
			}
		})
	}
}

func TestReportPutsMergeWarningsBeforeWriteWarnings(t *testing.T) {
	_, osm := parseCommittedOSM(t)
	workbook := workbookWithRoute17(t)
	merged, warnings := mergeCatalogs(workbook, osm)
	report := newReport(false, workbook, merged, committedSnapshotBase, warnings)

	// Route 1 had another route_id, so write adds a warning of its own.
	store := &fakeStore{routes: []routes_sqlc.Route{{ID: 1, RouteNumber: 1, SourceRouteID: text("1111")}}, nextID: 10}
	if err := write(context.Background(), store.queries(), merged, &report); err != nil {
		t.Fatal(err)
	}
	want := []string{
		"route 17: the workbook has this route; OSM relations 540033 and 540139 are skipped",
		"route 1: route_id changes from 1111 to 4450",
	}
	if !slices.Equal(report.Warnings, want) {
		t.Fatalf("warnings = %q\nwant %q", report.Warnings, want)
	}
	// The report has its own copy of the merge warnings.
	if &report.Warnings[0] == &warnings[0] {
		t.Error("report shares the merge warnings slice")
	}
}

func TestWriteStopsOnStorageErrors(t *testing.T) {
	failure := errors.New("connection reset")
	store := &failingStore{fakeStore: &fakeStore{}, failOn: "UpsertStop", err: failure}
	var report Report
	err := write(context.Background(), queries{routes: store.fakeStore, stops: store, catalog: store.fakeStore}, validCatalog(t), &report)
	if !errors.Is(err, failure) || !strings.Contains(err.Error(), "upsert stop") {
		t.Fatalf("error = %v, want the wrapped storage error", err)
	}
}

type failingStore struct {
	*fakeStore
	failOn string
	err    error
}

func (s *failingStore) UpsertStop(ctx context.Context, arg stops_sqlc.UpsertStopParams) (stops_sqlc.Stop, error) {
	if s.failOn == "UpsertStop" {
		return stops_sqlc.Stop{}, s.err
	}
	return s.fakeStore.UpsertStop(ctx, arg)
}
