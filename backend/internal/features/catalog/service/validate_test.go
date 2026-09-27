package catalog_service

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"strings"
	"testing"

	"github.com/jackc/pgx/v5"

	catalog_osm_repository "github.com/r0mbeg/TramCast/backend/internal/features/catalog/repository/osm"
	catalog_xlsx_repository "github.com/r0mbeg/TramCast/backend/internal/features/catalog/repository/xlsx"
)

type rowValues = map[string]string

func routeRow(sourceID, number, name string) rowValues {
	return rowValues{"route_id": sourceID, "route_short_name": number, "route_long_name": name,
		"route_type": "Тм", "status": "открыт", "route_date_end": ""}
}

func stopRow(sourceID, name, latitude, longitude string) rowValues {
	return rowValues{"stop_id": sourceID, "stop_name": name, "stop_lat": latitude, "stop_lon": longitude, "is_deleted": "0"}
}

func positionRow(routeID, number, tripID, direction, sequence, stopID string) rowValues {
	return rowValues{"route_id": routeID, "route_short_name": number, "trip_id": tripID, "direction_id": direction,
		"stop_sequence": sequence, "stop_id": stopID, "end_date": "", "is_addpoint": "0"}
}

// validSheets is a small consistent workbook; rows start at Excel row 3.
func validSheets() map[string][]rowValues {
	return map[string][]rowValues{
		RoutesSheet: {
			routeRow("4450", "1", "Чертаново Южное - Москворецкий рынок"),
			routeRow("3736", "5", ""),
		},
		StopsSheet: {
			stopRow("2594", "Чертаново Южное", "55.5946802200000008", "37.5908837699999978"),
			stopRow("2595", "Чертаново Южное", "55.5965768299999965", "37.5880750700000021"),
			stopRow("7879", "Поликлиника", "55.5987209800000031", "37.5885260499999987"),
		},
		PositionsSheet: {
			positionRow("4450", "1", "2040920", "0", "1", "2594"),
			positionRow("4450", "1", "2040920", "0", "2", "7879"),
			positionRow("4450", "1", "2040921", "1", "1", "7879"),
			positionRow("4450", "1", "2040921", "1", "2", "2595"),
		},
	}
}

func toSheets(values map[string][]rowValues) map[string]catalog_xlsx_repository.Sheet {
	sheets := make(map[string]catalog_xlsx_repository.Sheet, len(values))
	for name, rows := range values {
		sheet := catalog_xlsx_repository.Sheet{Name: name}
		for i, row := range rows {
			sheet.Rows = append(sheet.Rows, catalog_xlsx_repository.Row{Number: i + 3, Values: maps.Clone(row)})
		}
		sheets[name] = sheet
	}
	return sheets
}

func TestParseWorkbook(t *testing.T) {
	catalog, issues := parseWorkbook(toSheets(validSheets()))
	if len(issues) > 0 {
		t.Fatalf("issues = %+v", issues)
	}
	if len(catalog.Routes) != 2 || len(catalog.Stops) != 3 || len(catalog.Positions) != 4 {
		t.Fatalf("catalog = %+v", catalog)
	}
	if route := catalog.Routes[1]; route.Number != 5 || route.SourceID != "3736" || route.Name != "" {
		t.Errorf("route without a name = %+v", route)
	}
	// The stored text keeps float artefacts; parsing yields the exact value.
	if stop := catalog.Stops[0]; stop.Latitude != 55.59468022 || stop.Longitude != 37.59088377 {
		t.Errorf("stop coordinates = %v, %v", stop.Latitude, stop.Longitude)
	}
	if position := catalog.Positions[2]; position != (Position{RouteSourceID: "4450", PatternKey: "2040921", DirectionID: 1, StopSequence: 1, StopSourceID: "7879"}) {
		t.Errorf("position = %+v", position)
	}
}

func TestParseWorkbookProblems(t *testing.T) {
	// Changing an ID also leaves positions that reference the old ID; these
	// follow-up problems are expected after the first one.
	withConsequences := map[string]bool{"float route_id": true, "bad stop_id": true, "repeated stop_id": true}
	for _, tt := range []struct {
		name   string
		mutate func(map[string][]rowValues)
		sheet  string
		row    int
		want   string
	}{
		{"float route_id", func(s map[string][]rowValues) { s[RoutesSheet][0]["route_id"] = "4450.0" }, RoutesSheet, 3, `route_id "4450.0" is not a numeric identifier`},
		{"zero route number", func(s map[string][]rowValues) { s[RoutesSheet][0]["route_short_name"] = "0" }, RoutesSheet, 3, "not a positive route number"},
		{"repeated route_id", func(s map[string][]rowValues) { s[RoutesSheet][1]["route_id"] = "4450" }, RoutesSheet, 4, "route_id 4450 repeats row 3"},
		{"repeated route number", func(s map[string][]rowValues) { s[RoutesSheet][1]["route_short_name"] = "1" }, RoutesSheet, 4, "route number 1 repeats row 3"},
		{"not a tram", func(s map[string][]rowValues) { s[RoutesSheet][0]["route_type"] = "Ав" }, RoutesSheet, 3, `route_type is "Ав"`},
		{"closed route", func(s map[string][]rowValues) { s[RoutesSheet][0]["status"] = "закрыт" }, RoutesSheet, 3, `status is "закрыт"`},
		{"route end date", func(s map[string][]rowValues) { s[RoutesSheet][0]["route_date_end"] = "2026-01-01" }, RoutesSheet, 3, "routes with an end date are not supported"},
		{"bad stop_id", func(s map[string][]rowValues) { s[StopsSheet][0]["stop_id"] = "A1" }, StopsSheet, 3, `stop_id "A1" is not a numeric identifier`},
		{"repeated stop_id", func(s map[string][]rowValues) { s[StopsSheet][1]["stop_id"] = "2594" }, StopsSheet, 4, "stop_id 2594 repeats row 3"},
		{"empty stop name", func(s map[string][]rowValues) { s[StopsSheet][0]["stop_name"] = "" }, StopsSheet, 3, "stop_name is empty"},
		{"latitude text", func(s map[string][]rowValues) { s[StopsSheet][0]["stop_lat"] = "north" }, StopsSheet, 3, `stop_lat "north" is not a finite number`},
		{"latitude range", func(s map[string][]rowValues) { s[StopsSheet][0]["stop_lat"] = "91" }, StopsSheet, 3, `stop_lat "91" is outside [-90, 90]`},
		{"longitude NaN", func(s map[string][]rowValues) { s[StopsSheet][0]["stop_lon"] = "NaN" }, StopsSheet, 3, `stop_lon "NaN" is not a finite number`},
		{"empty longitude", func(s map[string][]rowValues) { s[StopsSheet][0]["stop_lon"] = "" }, StopsSheet, 3, "stop_lon is empty"},
		{"deleted stop", func(s map[string][]rowValues) { s[StopsSheet][2]["is_deleted"] = "1" }, StopsSheet, 5, "deleted stops are not supported"},
		{"unknown route", func(s map[string][]rowValues) { s[PositionsSheet][0]["route_id"] = "9999" }, PositionsSheet, 3, `route_id "9999" is not in sheet`},
		{"number mismatch", func(s map[string][]rowValues) { s[PositionsSheet][0]["route_short_name"] = "2" }, PositionsSheet, 3, `route_short_name "2" does not match route 1`},
		{"bad trip_id", func(s map[string][]rowValues) { s[PositionsSheet][0]["trip_id"] = "" }, PositionsSheet, 3, `trip_id "" is not a numeric identifier`},
		{"bad direction", func(s map[string][]rowValues) { s[PositionsSheet][0]["direction_id"] = "2" }, PositionsSheet, 3, `direction_id is "2"`},
		{"negative sequence", func(s map[string][]rowValues) { s[PositionsSheet][0]["stop_sequence"] = "-1" }, PositionsSheet, 3, `stop_sequence "-1" is not a non-negative integer`},
		{"unknown stop", func(s map[string][]rowValues) { s[PositionsSheet][0]["stop_id"] = "1" }, PositionsSheet, 3, `stop_id "1" is not in sheet`},
		{"pattern end date", func(s map[string][]rowValues) { s[PositionsSheet][0]["end_date"] = "2026-01-01" }, PositionsSheet, 3, "patterns with an end date are not supported"},
		{"additional point", func(s map[string][]rowValues) { s[PositionsSheet][0]["is_addpoint"] = "1" }, PositionsSheet, 3, "additional points are not supported"},
		{"repeated sequence", func(s map[string][]rowValues) { s[PositionsSheet][1]["stop_sequence"] = "1" }, PositionsSheet, 4, "stop_sequence 1 of trip_id 2040920 repeats row 3"},
		{"mixed direction", func(s map[string][]rowValues) { s[PositionsSheet][1]["direction_id"] = "1" }, PositionsSheet, 4, "trip_id 2040920 has direction_id 1 here but 0 in row 3"},
		{"trip in two routes", func(s map[string][]rowValues) {
			s[PositionsSheet] = append(s[PositionsSheet], positionRow("3736", "5", "2040920", "0", "9", "2594"))
		}, PositionsSheet, 7, "trip_id 2040920 belongs to route_id 3736 here but to 4450 in row 3"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			values := validSheets()
			tt.mutate(values)
			_, issues := parseWorkbook(toSheets(values))
			if len(issues) == 0 || len(issues) != 1 && !withConsequences[tt.name] {
				t.Fatalf("issues = %+v, want exactly one", issues)
			}
			issue := issues[0]
			if issue.OSM || issue.Sheet != tt.sheet || issue.Row != tt.row || !strings.Contains(issue.Message, tt.want) {
				t.Fatalf("issue = %+v, want sheet %q row %d containing %q", issue, tt.sheet, tt.row, tt.want)
			}
		})
	}
}

func TestParseWorkbookRejectsEmptySheets(t *testing.T) {
	values := validSheets()
	values[StopsSheet] = nil
	values[PositionsSheet] = nil
	_, issues := parseWorkbook(toSheets(values))
	var empty []string
	for _, issue := range issues {
		if issue.Row == 0 && issue.Message == "sheet has no data rows" {
			empty = append(empty, issue.Sheet)
		}
	}
	if strings.Join(empty, ",") != StopsSheet+","+PositionsSheet {
		t.Fatalf("empty-sheet issues = %q, issues = %+v", empty, issues)
	}
	err := &ValidationError{Issues: issues}
	if !strings.Contains(err.Error(), `first: sheet "`+StopsSheet+`": sheet has no data rows`) {
		t.Errorf("summary = %q", err)
	}
}

func TestParseWorkbookCollectsAllProblems(t *testing.T) {
	values := validSheets()
	values[RoutesSheet][0]["status"] = "закрыт"
	values[StopsSheet][0]["stop_lat"] = "north"
	values[PositionsSheet][3]["is_addpoint"] = "1"
	_, issues := parseWorkbook(toSheets(values))
	if len(issues) != 3 {
		t.Fatalf("issues = %+v, want three problems", issues)
	}
	if err := (&ValidationError{Issues: issues}); !strings.HasPrefix(err.Error(), "catalog has 3 problem(s); first: ") {
		t.Errorf("summary = %q", err)
	}
}

func TestIssueLocation(t *testing.T) {
	for _, tt := range []struct {
		issue Issue
		want  string
	}{
		{Issue{Sheet: StopsSheet}, `sheet "Остановки GTFS_STOPS"`},
		{Issue{Sheet: StopsSheet, Row: 7}, `sheet "Остановки GTFS_STOPS" row 7`},
		{Issue{OSM: true}, "OSM snapshot"},
		{Issue{OSM: true, Relation: 540033}, "OSM relation 540033"},
		{Issue{OSM: true, Relation: 540033, Member: 12}, "OSM relation 540033 member 12"},
	} {
		if got := tt.issue.Location(); got != tt.want {
			t.Errorf("Location(%+v) = %q, want %q", tt.issue, got, tt.want)
		}
	}
}

type fakeReader struct {
	sheets map[string]catalog_xlsx_repository.Sheet
	err    error
	// path, when set, is the only workbook path the reader accepts.
	path string
}

func (r fakeReader) ReadSheets(_ context.Context, path string, _ []catalog_xlsx_repository.SheetSpec) (map[string]catalog_xlsx_repository.Sheet, error) {
	if r.path != "" && path != r.path {
		return nil, fmt.Errorf("workbook path %q, want %q", path, r.path)
	}
	return r.sheets, r.err
}

type fakeOSMReader struct {
	snapshot catalog_osm_repository.Snapshot
	err      error
}

func (r fakeOSMReader) ReadSnapshot(context.Context, string) (catalog_osm_repository.Snapshot, error) {
	return r.snapshot, r.err
}

type unusedDB struct{ t *testing.T }

func (db unusedDB) Begin(context.Context) (pgx.Tx, error) {
	db.t.Fatal("the database must not be touched")
	return nil, nil
}

func TestImportStopsBeforeTheDatabase(t *testing.T) {
	// The real repository reads the committed snapshot only from sources.OSM.
	committed := catalog_osm_repository.New()
	sources := Sources{Workbook: "catalog.xlsx", OSM: committedSnapshot}
	importWith := func(t *testing.T, workbook WorkbookReader, osm OSMReader) error {
		t.Helper()
		_, err := NewService(workbook, osm, unusedDB{t}).Import(context.Background(), sources, false)
		return err
	}
	t.Run("workbook read error", func(t *testing.T) {
		failure := errors.New("sheet not found")
		if err := importWith(t, fakeReader{err: failure}, committed); !errors.Is(err, failure) {
			t.Fatalf("error = %v, want the read error", err)
		}
	})
	t.Run("snapshot read error", func(t *testing.T) {
		failure := errors.New("decode Overpass JSON")
		err := importWith(t, fakeReader{sheets: toSheets(validSheets())}, fakeOSMReader{err: failure})
		if !errors.Is(err, failure) || !strings.Contains(err.Error(), "read OSM snapshot") {
			t.Fatalf("error = %v, want the wrapped read error", err)
		}
	})
	t.Run("invalid workbook", func(t *testing.T) {
		values := validSheets()
		values[StopsSheet][0]["is_deleted"] = "1"
		var validation *ValidationError
		// Each reader gets its own path: swapped paths fail with a read error.
		err := importWith(t, fakeReader{sheets: toSheets(values), path: sources.Workbook}, committed)
		if !errors.As(err, &validation) || len(validation.Issues) != 1 || validation.Issues[0].OSM {
			t.Fatalf("error = %v, want ValidationError with the workbook problem only", err)
		}
	})
	t.Run("invalid snapshot", func(t *testing.T) {
		snapshot := catalog_osm_repository.Snapshot{BaseTimestamp: "2026-09-26T22:39:54Z", Remark: "runtime error: Query timed out"}
		var validation *ValidationError
		err := importWith(t, fakeReader{sheets: toSheets(validSheets())}, fakeOSMReader{snapshot: snapshot})
		if !errors.As(err, &validation) || !validation.Issues[0].OSM {
			t.Fatalf("error = %v, want ValidationError with OSM issues", err)
		}
		if !strings.HasPrefix(err.Error(), `catalog has `) || !strings.Contains(err.Error(), `first: OSM snapshot: Overpass remark "runtime error: Query timed out"`) {
			t.Errorf("summary = %q", err)
		}
	})
	t.Run("problems of both sources together", func(t *testing.T) {
		values := validSheets()
		values[StopsSheet][0]["is_deleted"] = "1"
		var validation *ValidationError
		err := importWith(t, fakeReader{sheets: toSheets(values)}, fakeOSMReader{})
		if !errors.As(err, &validation) {
			t.Fatalf("error = %v, want ValidationError", err)
		}
		// Workbook problems come first, then the snapshot's.
		first, last := validation.Issues[0], validation.Issues[len(validation.Issues)-1]
		if first.OSM || first.Sheet != StopsSheet || !last.OSM {
			t.Fatalf("issues = %+v, want the workbook problem followed by snapshot problems", validation.Issues)
		}
	})
}
