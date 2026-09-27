package catalog_service

import (
	"fmt"
	"math"
	"regexp"
	"strconv"
	"strings"

	catalog_xlsx_repository "github.com/r0mbeg/TramCast/backend/internal/features/catalog/repository/xlsx"
)

const (
	RoutesSheet    = "Маршруты GTFS_ROUTES"
	StopsSheet     = "Остановки GTFS_STOPS"
	PositionsSheet = "Порядок_остановок GTFS_TRIPS_ST"

	tramRouteType   = "Тм"
	openRouteStatus = "открыт"
)

// sheetSpecs lists the sheets and columns the import reads. Other sheets,
// including the derived "Порядок_с_координатами", are ignored.
var sheetSpecs = []catalog_xlsx_repository.SheetSpec{
	{Name: RoutesSheet, Columns: []string{"route_id", "route_short_name", "route_long_name", "route_type", "status", "route_date_end"}},
	{Name: StopsSheet, Columns: []string{"stop_id", "stop_name", "stop_lat", "stop_lon", "is_deleted"}},
	{Name: PositionsSheet, Columns: []string{"route_id", "route_short_name", "trip_id", "direction_id", "stop_sequence", "stop_id", "end_date", "is_addpoint"}},
}

// digits accepts source identifiers stored as text; it rejects float
// artefacts such as "4450.0" that a numeric cell could produce.
var digits = regexp.MustCompile(`^[0-9]+$`)

type Route struct {
	Number int16
	// Name is empty when the workbook has none; it is stored as NULL.
	Name     string
	SourceID string
}

type Stop struct {
	SourceID  string
	Name      string
	Latitude  float64
	Longitude float64
}

type Position struct {
	RouteSourceID string
	PatternKey    string
	DirectionID   int16
	StopSequence  int32
	StopSourceID  string
}

// Catalog is validated content of the workbook, of the OSM snapshot, or of
// both merged.
type Catalog struct {
	Routes    []Route
	Stops     []Stop
	Positions []Position
}

// Issue is one problem of a catalog source. A workbook problem is located by
// sheet and Excel row, an OSM snapshot problem by relation and member.
type Issue struct {
	Sheet string
	// Row is 0 for a problem of the whole sheet.
	Row int
	// OSM marks a problem of the OSM snapshot. Relation is 0 for a problem of
	// the whole snapshot; Member is the 1-based position in the relation's
	// member list, or 0 for a problem of the relation itself.
	OSM      bool
	Relation int64
	Member   int
	Message  string
}

// Location formats the most precise known place of the problem.
func (i Issue) Location() string {
	switch {
	case i.OSM && i.Relation == 0:
		return "OSM snapshot"
	case i.OSM && i.Member == 0:
		return fmt.Sprintf("OSM relation %d", i.Relation)
	case i.OSM:
		return fmt.Sprintf("OSM relation %d member %d", i.Relation, i.Member)
	case i.Row == 0:
		return fmt.Sprintf("sheet %q", i.Sheet)
	default:
		return fmt.Sprintf("sheet %q row %d", i.Sheet, i.Row)
	}
}

// ValidationError lists every problem of both sources; nothing is written to
// the database.
type ValidationError struct {
	Issues []Issue
}

func (e *ValidationError) Error() string {
	first := e.Issues[0]
	return fmt.Sprintf("catalog has %d problem(s); first: %s: %s", len(e.Issues), first.Location(), first.Message)
}

type validator struct {
	issues []Issue
}

func (v *validator) add(sheet string, row int, format string, args ...any) {
	v.issues = append(v.issues, Issue{Sheet: sheet, Row: row, Message: fmt.Sprintf(format, args...)})
}

// parseWorkbook checks the whole workbook before any write and returns the
// catalog only when there are no issues. Unknown situations, such as deleted
// stops or closed routes, are errors rather than guesses.
func parseWorkbook(sheets map[string]catalog_xlsx_repository.Sheet) (Catalog, []Issue) {
	var v validator
	// The workbook is part of a full snapshot, so an accidentally empty sheet
	// would delete its whole geography.
	for _, spec := range sheetSpecs {
		if len(sheets[spec.Name].Rows) == 0 {
			v.add(spec.Name, 0, "sheet has no data rows")
		}
	}
	routes, routeNumbers := v.routes(sheets[RoutesSheet])
	stops, stopIDs := v.stops(sheets[StopsSheet])
	positions := v.positions(sheets[PositionsSheet], routeNumbers, stopIDs)
	if len(v.issues) > 0 {
		return Catalog{}, v.issues
	}
	return Catalog{Routes: routes, Stops: stops, Positions: positions}, nil
}

func (v *validator) routes(sheet catalog_xlsx_repository.Sheet) ([]Route, map[string]int16) {
	var routes []Route
	numbers := make(map[string]int16)
	sourceRows := make(map[string]int)
	numberRows := make(map[int16]int)
	for _, row := range sheet.Rows {
		sourceID := row.Get("route_id")
		validSource := digits.MatchString(sourceID)
		if !validSource {
			v.add(RoutesSheet, row.Number, "route_id %q is not a numeric identifier", sourceID)
		}
		number, err := parseInteger(row.Get("route_short_name"), 16)
		validNumber := err == nil && number > 0
		if !validNumber {
			v.add(RoutesSheet, row.Number, "route_short_name %q is not a positive route number", row.Get("route_short_name"))
		}
		if value := row.Get("route_type"); value != tramRouteType {
			v.add(RoutesSheet, row.Number, "route_type is %q; only trams (%q) are supported", value, tramRouteType)
		}
		if value := row.Get("status"); value != openRouteStatus {
			v.add(RoutesSheet, row.Number, "status is %q; only open routes (%q) are supported", value, openRouteStatus)
		}
		if value := row.Get("route_date_end"); value != "" {
			v.add(RoutesSheet, row.Number, "route_date_end is %q; routes with an end date are not supported", value)
		}
		if validSource {
			if first, repeated := sourceRows[sourceID]; repeated {
				v.add(RoutesSheet, row.Number, "route_id %s repeats row %d", sourceID, first)
				continue
			}
			sourceRows[sourceID] = row.Number
			// Known with number 0 until the number is valid, so positions do not
			// report a second, misleading "unknown route" problem.
			numbers[sourceID] = 0
		}
		if !validNumber {
			continue
		}
		if first, repeated := numberRows[int16(number)]; repeated {
			v.add(RoutesSheet, row.Number, "route number %d repeats row %d", number, first)
			continue
		}
		numberRows[int16(number)] = row.Number
		if !validSource {
			continue
		}
		numbers[sourceID] = int16(number)
		routes = append(routes, Route{Number: int16(number), Name: row.Get("route_long_name"), SourceID: sourceID})
	}
	return routes, numbers
}

func (v *validator) stops(sheet catalog_xlsx_repository.Sheet) ([]Stop, map[string]bool) {
	var stops []Stop
	known := make(map[string]bool)
	sourceRows := make(map[string]int)
	for _, row := range sheet.Rows {
		sourceID := row.Get("stop_id")
		valid := true
		if !digits.MatchString(sourceID) {
			v.add(StopsSheet, row.Number, "stop_id %q is not a numeric identifier", sourceID)
			valid = false
		} else if first, repeated := sourceRows[sourceID]; repeated {
			v.add(StopsSheet, row.Number, "stop_id %s repeats row %d", sourceID, first)
			valid = false
		}
		name := row.Get("stop_name")
		if name == "" {
			v.add(StopsSheet, row.Number, "stop_name is empty")
			valid = false
		}
		latitude, err := parseCoordinate(row.Get("stop_lat"), 90)
		if err != nil {
			v.add(StopsSheet, row.Number, "stop_lat %s", err)
			valid = false
		}
		longitude, err := parseCoordinate(row.Get("stop_lon"), 180)
		if err != nil {
			v.add(StopsSheet, row.Number, "stop_lon %s", err)
			valid = false
		}
		switch value := row.Get("is_deleted"); value {
		case "0":
		case "1":
			v.add(StopsSheet, row.Number, "stop is marked deleted (is_deleted=1); deleted stops are not supported")
			valid = false
		default:
			v.add(StopsSheet, row.Number, "is_deleted is %q, want 0", value)
			valid = false
		}
		if digits.MatchString(sourceID) {
			if _, repeated := sourceRows[sourceID]; !repeated {
				sourceRows[sourceID] = row.Number
			}
			// Keep the ID known even if other fields fail, so positions do not
			// report a second, misleading "unknown stop" problem.
			known[sourceID] = true
		}
		if valid {
			stops = append(stops, Stop{SourceID: sourceID, Name: name, Latitude: latitude, Longitude: longitude})
		}
	}
	return stops, known
}

func (v *validator) positions(sheet catalog_xlsx_repository.Sheet, routeNumbers map[string]int16, stopIDs map[string]bool) []Position {
	type patternKey struct{ route, trip string }
	type positionKey struct {
		pattern  patternKey
		sequence int32
	}
	type firstSeen struct {
		value string
		row   int
	}
	var positions []Position
	tripRoutes := make(map[string]firstSeen)
	directions := make(map[patternKey]firstSeen)
	sequences := make(map[positionKey]int)
	for _, row := range sheet.Rows {
		routeID, tripID, stopID := row.Get("route_id"), row.Get("trip_id"), row.Get("stop_id")
		valid := true
		if number, known := routeNumbers[routeID]; !known {
			v.add(PositionsSheet, row.Number, "route_id %q is not in sheet %q", routeID, RoutesSheet)
			valid = false
		} else if value := row.Get("route_short_name"); number > 0 && value != strconv.Itoa(int(number)) {
			v.add(PositionsSheet, row.Number, "route_short_name %q does not match route %d of route_id %s", value, number, routeID)
			valid = false
		}
		if !digits.MatchString(tripID) {
			v.add(PositionsSheet, row.Number, "trip_id %q is not a numeric identifier", tripID)
			valid = false
		}
		direction := row.Get("direction_id")
		if direction != "0" && direction != "1" {
			v.add(PositionsSheet, row.Number, "direction_id is %q, want 0 or 1", direction)
			valid = false
		}
		sequence, err := parseInteger(row.Get("stop_sequence"), 32)
		if err != nil {
			v.add(PositionsSheet, row.Number, "stop_sequence %q is not a non-negative integer", row.Get("stop_sequence"))
			valid = false
		}
		if !stopIDs[stopID] {
			v.add(PositionsSheet, row.Number, "stop_id %q is not in sheet %q", stopID, StopsSheet)
			valid = false
		}
		if value := row.Get("end_date"); value != "" {
			v.add(PositionsSheet, row.Number, "end_date is %q; patterns with an end date are not supported", value)
			valid = false
		}
		if value := row.Get("is_addpoint"); value != "0" {
			v.add(PositionsSheet, row.Number, "is_addpoint is %q; additional points are not supported", value)
			valid = false
		}
		if !valid {
			continue
		}

		pattern := patternKey{route: routeID, trip: tripID}
		if first, seen := tripRoutes[tripID]; seen && first.value != routeID {
			v.add(PositionsSheet, row.Number, "trip_id %s belongs to route_id %s here but to %s in row %d", tripID, routeID, first.value, first.row)
			continue
		} else if !seen {
			tripRoutes[tripID] = firstSeen{value: routeID, row: row.Number}
		}
		if first, seen := directions[pattern]; seen && first.value != direction {
			v.add(PositionsSheet, row.Number, "trip_id %s has direction_id %s here but %s in row %d", tripID, direction, first.value, first.row)
			continue
		} else if !seen {
			directions[pattern] = firstSeen{value: direction, row: row.Number}
		}
		key := positionKey{pattern: pattern, sequence: int32(sequence)}
		if first, repeated := sequences[key]; repeated {
			v.add(PositionsSheet, row.Number, "stop_sequence %d of trip_id %s repeats row %d", sequence, tripID, first)
			continue
		}
		sequences[key] = row.Number

		directionID := int16(0)
		if direction == "1" {
			directionID = 1
		}
		positions = append(positions, Position{
			RouteSourceID: routeID,
			PatternKey:    tripID,
			DirectionID:   directionID,
			StopSequence:  int32(sequence),
			StopSourceID:  stopID,
		})
	}
	return positions
}

// parseInteger accepts only decimal digits, so signs, spaces and "1.0" fail.
func parseInteger(value string, bitSize int) (int64, error) {
	if !digits.MatchString(value) {
		return 0, fmt.Errorf("not a non-negative integer")
	}
	return strconv.ParseInt(value, 10, bitSize)
}

// parseCoordinate parses the stored text exactly; the workbook keeps values
// such as "55.6754278100000022", which ParseFloat maps to 55.67542781.
func parseCoordinate(value string, limit float64) (float64, error) {
	if strings.TrimSpace(value) == "" {
		return 0, fmt.Errorf("is empty")
	}
	parsed, err := strconv.ParseFloat(value, 64)
	if err != nil || math.IsNaN(parsed) || math.IsInf(parsed, 0) {
		return 0, fmt.Errorf("%q is not a finite number", value)
	}
	if parsed < -limit || parsed > limit {
		return 0, fmt.Errorf("%q is outside [-%g, %g]", value, limit, limit)
	}
	return parsed, nil
}
