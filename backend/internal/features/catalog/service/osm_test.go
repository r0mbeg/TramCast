package catalog_service

import (
	"context"
	"slices"
	"strings"
	"testing"

	catalog_osm_repository "github.com/r0mbeg/TramCast/backend/internal/features/catalog/repository/osm"
)

type osmElement = catalog_osm_repository.Element

func stopNode(id int64, name string, latitude, longitude float64) osmElement {
	return osmElement{Type: "node", ID: id, Lat: &latitude, Lon: &longitude,
		Tags: map[string]string{"public_transport": "stop_position", "railway": "tram_stop", "name": name}}
}

func platformNode(id int64, latitude, longitude float64) osmElement {
	return osmElement{Type: "node", ID: id, Lat: &latitude, Lon: &longitude,
		Tags: map[string]string{"public_transport": "platform", "name": "Платформа"}}
}

func member(memberType string, ref int64, role string) catalog_osm_repository.Member {
	return catalog_osm_repository.Member{Type: memberType, Ref: ref, Role: role}
}

func routeRelation(id int64, ref string, members ...catalog_osm_repository.Member) osmElement {
	return osmElement{Type: "relation", ID: id, Members: members,
		Tags: map[string]string{"type": "route", "route": "tram", "public_transport:version": "2", "ref": ref}}
}

func routeMaster(id int64, ref string, routes ...int64) osmElement {
	master := osmElement{Type: "relation", ID: id, Tags: map[string]string{"type": "route_master", "route_master": "tram", "ref": ref}}
	for _, route := range routes {
		master.Members = append(master.Members, member("relation", route, ""))
	}
	return master
}

// testOSMRoutes plays the role of OSMRoutes for validSnapshot.
var testOSMRoutes = map[int16][2]int64{17: {101, 102}, 25: {201, 202}}

// validSnapshot is a small consistent snapshot. Routes 17 and 25 share nodes 1
// and 5; both directions of route 17 share the terminus 13725188937, an ID
// above 2^31. Relation 101 also has every kind of skipped member.
func validSnapshot() catalog_osm_repository.Snapshot {
	return catalog_osm_repository.Snapshot{
		Generator:     "Overpass API 0.7.62.4",
		BaseTimestamp: "2026-09-26T22:39:54Z",
		Elements: []osmElement{
			routeRelation(101, "17",
				member("node", 1, "stop_entry_only"),
				member("way", 900, "platform_entry_only"),
				member("node", 2, "stop"),
				member("node", 950, "platform"),
				member("way", 901, ""),
				member("relation", 960, "platform"),
				member("node", 13725188937, "stop_exit_only"),
				member("node", 951, "platform_exit_only"),
			),
			routeRelation(102, "17",
				member("node", 13725188937, "stop_entry_only"),
				member("node", 4, "stop"),
				member("node", 5, "stop_exit_only"),
				member("way", 902, ""),
			),
			routeRelation(201, "25", member("node", 1, "stop_entry_only"), member("node", 6, "stop_exit_only")),
			routeRelation(202, "25", member("node", 6, "stop_entry_only"), member("node", 5, "stop_exit_only")),
			stopNode(1, "Усадьба Останкино", 55.8200, 37.6100),
			stopNode(2, "улица Бориса Галушкина, 17", 55.8240, 37.6100),
			stopNode(4, "Улица Бориса Галушкина, 17", 55.8241, 37.6102),
			stopNode(5, "Усадьба Останкино", 55.8201, 37.6102),
			stopNode(6, "Метро «Сокольники»", 55.8200, 37.6300),
			platformNode(950, 55.8240, 37.6098),
			platformNode(951, 55.8280, 37.6098),
			stopNode(13725188937, "Метро «ВДНХ»", 55.8280, 37.6100),
			// Member order inside a route_master carries no direction.
			routeMaster(1001, "17", 102, 101),
			routeMaster(1002, "25", 201, 202),
		},
	}
}

func element(s *catalog_osm_repository.Snapshot, elementType string, id int64) *osmElement {
	for i := range s.Elements {
		if s.Elements[i].Type == elementType && s.Elements[i].ID == id {
			return &s.Elements[i]
		}
	}
	panic("no such element in the fixture")
}

func removeElement(s *catalog_osm_repository.Snapshot, elementType string, id int64) {
	s.Elements = slices.DeleteFunc(s.Elements, func(e osmElement) bool { return e.Type == elementType && e.ID == id })
}

func parseValidOSM(t *testing.T) osmCatalog {
	t.Helper()
	osm, issues := parseOSM(validSnapshot(), testOSMRoutes)
	if len(issues) > 0 {
		t.Fatalf("issues = %+v", issues)
	}
	return osm
}

func TestParseOSM(t *testing.T) {
	osm := parseValidOSM(t)
	if len(osm.Routes) != 2 || len(osm.Stops) != 6 || len(osm.Warnings) != 0 {
		t.Fatalf("catalog = %+v", osm)
	}
	route17, route25 := osm.Routes[0], osm.Routes[1]
	if route17.Route != (Route{Number: 17, Name: `Усадьба Останкино - Метро "ВДНХ"`, SourceID: "osm:relation/1001"}) ||
		route17.Relations != [2]int64{101, 102} {
		t.Errorf("route 17 = %+v", route17)
	}
	want := []Position{
		{RouteSourceID: "osm:relation/1001", PatternKey: "osm:relation/101", DirectionID: 0, StopSequence: 1, StopSourceID: "osm:node/1"},
		{RouteSourceID: "osm:relation/1001", PatternKey: "osm:relation/101", DirectionID: 0, StopSequence: 2, StopSourceID: "osm:node/2"},
		{RouteSourceID: "osm:relation/1001", PatternKey: "osm:relation/101", DirectionID: 0, StopSequence: 3, StopSourceID: "osm:node/13725188937"},
		{RouteSourceID: "osm:relation/1001", PatternKey: "osm:relation/102", DirectionID: 1, StopSequence: 1, StopSourceID: "osm:node/13725188937"},
		{RouteSourceID: "osm:relation/1001", PatternKey: "osm:relation/102", DirectionID: 1, StopSequence: 2, StopSourceID: "osm:node/4"},
		{RouteSourceID: "osm:relation/1001", PatternKey: "osm:relation/102", DirectionID: 1, StopSequence: 3, StopSourceID: "osm:node/5"},
	}
	if !slices.Equal(route17.Positions, want) {
		t.Errorf("positions of route 17 = %+v\nwant %+v", route17.Positions, want)
	}
	if route25.Route != (Route{Number: 25, Name: `Усадьба Останкино - Метро "Сокольники"`, SourceID: "osm:relation/1002"}) ||
		len(route25.Positions) != 4 {
		t.Errorf("route 25 = %+v", route25)
	}
	if stop := osm.Stops["osm:node/2"]; stop != (Stop{SourceID: "osm:node/2", Name: "Улица Бориса Галушкина, 17", Latitude: 55.824, Longitude: 37.61}) {
		t.Errorf("stop 2 = %+v", stop)
	}
	if _, found := osm.Stops["osm:node/950"]; found {
		t.Error("a platform node must not become a stop")
	}
}

func TestParseOSMProblems(t *testing.T) {
	// Some changes also invalidate what depends on them; these follow-up
	// problems are expected after the first one.
	withConsequences := map[string]bool{"no elements": true, "relation listed twice": true}
	for _, tt := range []struct {
		name     string
		mutate   func(*catalog_osm_repository.Snapshot)
		routes   map[int16][2]int64
		relation int64
		member   int
		want     string
	}{
		{name: "Overpass remark", mutate: func(s *catalog_osm_repository.Snapshot) { s.Remark = "runtime error: Query timed out" },
			want: `Overpass remark "runtime error: Query timed out"`},
		{name: "missing timestamp", mutate: func(s *catalog_osm_repository.Snapshot) { s.BaseTimestamp = "" },
			want: "osm3s.timestamp_osm_base is missing"},
		{name: "malformed timestamp", mutate: func(s *catalog_osm_repository.Snapshot) { s.BaseTimestamp = "26.09.2026" },
			want: `osm3s.timestamp_osm_base "26.09.2026" is not an RFC 3339 time`},
		{name: "no elements", mutate: func(s *catalog_osm_repository.Snapshot) { s.Elements = nil },
			want: "snapshot has no elements"},
		{name: "way element", mutate: func(s *catalog_osm_repository.Snapshot) {
			s.Elements = append(s.Elements, osmElement{Type: "way", ID: 900})
		}, want: "1 element(s) are neither nodes nor relations, first way 900"},
		{name: "repeated node", mutate: func(s *catalog_osm_repository.Snapshot) {
			s.Elements = append(s.Elements, *element(s, "node", 6))
		}, want: "node 6 appears more than once"},
		{name: "repeated relation", mutate: func(s *catalog_osm_repository.Snapshot) {
			s.Elements = append(s.Elements, *element(s, "relation", 201))
		}, relation: 201, want: "relation appears more than once"},
		{name: "relation listed twice", routes: map[int16][2]int64{17: {101, 101}, 25: {201, 202}},
			relation: 101, want: "relation is listed for route 17 direction 0 and route 17 direction 1"},
		{name: "unlisted route relation", mutate: func(s *catalog_osm_repository.Snapshot) {
			s.Elements = append(s.Elements, routeRelation(301, "99", member("node", 1, "stop"), member("node", 6, "stop")))
		}, relation: 301, want: `route relation with ref "99" is not listed in OSMRoutes`},
		{name: "missing relation", mutate: func(s *catalog_osm_repository.Snapshot) { removeElement(s, "relation", 102) },
			relation: 102, want: "relation of route 17 direction 1 is not in the snapshot"},
		{name: "wrong ref", mutate: func(s *catalog_osm_repository.Snapshot) { element(s, "relation", 101).Tags["ref"] = "18" },
			relation: 101, want: `tag ref is "18", want "17"`},
		{name: "not a tram", mutate: func(s *catalog_osm_repository.Snapshot) { element(s, "relation", 101).Tags["route"] = "bus" },
			relation: 101, want: `tag route is "bus", want "tram"`},
		{name: "not a route", mutate: func(s *catalog_osm_repository.Snapshot) { element(s, "relation", 101).Tags["type"] = "multipolygon" },
			relation: 101, want: `tag type is "multipolygon", want "route"`},
		{name: "PTv1 relation", mutate: func(s *catalog_osm_repository.Snapshot) {
			element(s, "relation", 101).Tags["public_transport:version"] = "1"
		}, relation: 101, want: `tag public_transport:version is "1", want "2"`},
		{name: "missing route_master", mutate: func(s *catalog_osm_repository.Snapshot) { removeElement(s, "relation", 1001) },
			relation: 101, want: "no route_master relation groups relations 101 and 102 of route 17"},
		{name: "two route_masters", mutate: func(s *catalog_osm_repository.Snapshot) {
			s.Elements = append(s.Elements, routeMaster(1003, "17", 102))
		}, relation: 101, want: "relations 101 and 102 of route 17 belong to 2 route_master relations [1001 1003]"},
		{name: "route_master misses a direction", mutate: func(s *catalog_osm_repository.Snapshot) {
			element(s, "relation", 1001).Members = []catalog_osm_repository.Member{member("relation", 101, "")}
		}, relation: 1001, want: "route_master does not group relation 102 of route 17"},
		{name: "route_master ref", mutate: func(s *catalog_osm_repository.Snapshot) { element(s, "relation", 1001).Tags["ref"] = "18" },
			relation: 1001, want: `tag ref is "18", want "17"`},
		{name: "route_master of buses", mutate: func(s *catalog_osm_repository.Snapshot) {
			element(s, "relation", 1001).Tags["route_master"] = "bus"
		}, relation: 1001, want: `tag route_master is "bus", want "tram"`},
		{name: "stop member that is a way", mutate: func(s *catalog_osm_repository.Snapshot) {
			element(s, "relation", 101).Members[2] = member("way", 2, "stop")
		}, relation: 101, member: 3, want: `role "stop" needs a node, got way 2`},
		{name: "unknown role on a node", mutate: func(s *catalog_osm_repository.Snapshot) {
			element(s, "relation", 101).Members[3].Role = "stop_area"
		}, relation: 101, member: 4, want: `node 950 has role "stop_area"`},
		{name: "empty role on a node", mutate: func(s *catalog_osm_repository.Snapshot) {
			element(s, "relation", 101).Members[3].Role = ""
		}, relation: 101, member: 4, want: `node 950 has role ""`},
		{name: "missing node", mutate: func(s *catalog_osm_repository.Snapshot) { removeElement(s, "node", 2) },
			relation: 101, member: 3, want: "node 2 is not in the snapshot"},
		{name: "node without name", mutate: func(s *catalog_osm_repository.Snapshot) { element(s, "node", 2).Tags["name"] = "  " },
			relation: 101, member: 3, want: "node 2 has no name"},
		{name: "platform node as stop", mutate: func(s *catalog_osm_repository.Snapshot) {
			element(s, "node", 2).Tags["public_transport"] = "platform"
		}, relation: 101, member: 3, want: `node 2 has public_transport="platform", want "stop_position"`},
		{name: "latitude out of range", mutate: func(s *catalog_osm_repository.Snapshot) {
			latitude := 91.0
			element(s, "node", 2).Lat = &latitude
		}, relation: 101, member: 3, want: "node 2 lat 91 is outside [-90, 90]"},
		{name: "missing longitude", mutate: func(s *catalog_osm_repository.Snapshot) { element(s, "node", 2).Lon = nil },
			relation: 101, member: 3, want: "node 2 lon is missing"},
		{name: "shared node is reported once", mutate: func(s *catalog_osm_repository.Snapshot) { delete(element(s, "node", 1).Tags, "name") },
			relation: 101, member: 1, want: "node 1 has no name"},
		{name: "fewer than two stops", mutate: func(s *catalog_osm_repository.Snapshot) {
			element(s, "relation", 202).Members = []catalog_osm_repository.Member{member("node", 6, "stop_entry_only")}
		}, relation: 202, want: "relation has 1 stop(s), want at least 2"},
		{name: "same node twice in a row", mutate: func(s *catalog_osm_repository.Snapshot) {
			element(s, "relation", 101).Members[2].Ref = 1
		}, relation: 101, member: 3, want: "node 1 repeats the previous stop"},
		{name: "gap over 3 km", mutate: func(s *catalog_osm_repository.Snapshot) {
			relation := element(s, "relation", 101)
			relation.Members = append(relation.Members, member("node", 7, "stop"))
			s.Elements = append(s.Elements, stopNode(7, "Далеко", 55.8600, 37.6100))
		}, relation: 101, member: 9, want: "node 7 is 3.6 km from the previous stop node 13725188937; the limit is 3 km"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			snapshot := validSnapshot()
			if tt.mutate != nil {
				tt.mutate(&snapshot)
			}
			routes := tt.routes
			if routes == nil {
				routes = testOSMRoutes
			}
			osm, issues := parseOSM(snapshot, routes)
			if len(issues) == 0 || len(issues) != 1 && !withConsequences[tt.name] {
				t.Fatalf("issues = %+v, want exactly one", issues)
			}
			if osm.Routes != nil || osm.Stops != nil {
				t.Errorf("catalog = %+v, want none with issues", osm)
			}
			issue := issues[0]
			if !issue.OSM || issue.Relation != tt.relation || issue.Member != tt.member || !strings.Contains(issue.Message, tt.want) {
				t.Fatalf("issue = %+v, want relation %d member %d containing %q", issue, tt.relation, tt.member, tt.want)
			}
		})
	}
}

func TestParseOSMWarnings(t *testing.T) {
	snapshot := validSnapshot()
	master := element(&snapshot, "relation", 1001)
	master.Members = append(master.Members, member("relation", 103, ""))
	snapshot.Elements = append(snapshot.Elements,
		osmElement{Type: "relation", ID: 5000, Tags: map[string]string{"type": "network"}},
		routeMaster(5001, "99", 999),
	)
	osm, issues := parseOSM(snapshot, testOSMRoutes)
	if len(issues) > 0 {
		t.Fatalf("issues = %+v", issues)
	}
	want := []string{
		`OSM relation 5000: type "network" is not used; it is ignored`,
		"OSM relation 5001: route_master groups none of the listed relations; it is ignored",
		"route 17: route_master 1001 also groups relation 103; ignored",
	}
	if !slices.Equal(osm.Warnings, want) {
		t.Fatalf("warnings = %q\nwant %q", osm.Warnings, want)
	}
}

func TestNormalizeOSMName(t *testing.T) {
	for input, want := range map[string]string{
		"Метро «Сокольники»":          `Метро "Сокольники"`,
		" Дом культуры „Компрессор“ ": `Дом культуры "Компрессор"`,
		"Метро ”Сокол”":               `Метро "Сокол"`,
		"улица Бориса Галушкина, 17":  "Улица Бориса Галушкина, 17",
		"ёлочная улица":               "Ёлочная улица",
		"Семёновская площадь":         "Семёновская площадь",
		"3-й Верхний Михайловский":    "3-й Верхний Михайловский",
		"\tПроспект Мира  ":           "Проспект Мира",
	} {
		if got := normalizeOSMName(input); got != want {
			t.Errorf("normalizeOSMName(%q) = %q, want %q", input, got, want)
		}
	}
}

func stopSourceIDs(stops []Stop) []string {
	ids := make([]string, len(stops))
	for i, stop := range stops {
		ids[i] = stop.SourceID
	}
	return ids
}

func TestMergeCatalogs(t *testing.T) {
	workbook := validCatalog(t)
	merged, warnings := mergeCatalogs(workbook, parseValidOSM(t))
	if len(warnings) != 0 {
		t.Fatalf("warnings = %q", warnings)
	}
	if len(merged.Routes) != 4 || merged.Routes[2].Number != 17 || merged.Routes[3].Number != 25 || len(merged.Positions) != 4+10 {
		t.Fatalf("merged = %+v", merged)
	}
	// Nodes shared by the directions of route 17 and by routes 17 and 25 become one stop each.
	want := []string{"2594", "2595", "7879", "osm:node/1", "osm:node/2", "osm:node/13725188937", "osm:node/4", "osm:node/5", "osm:node/6"}
	if got := stopSourceIDs(merged.Stops); !slices.Equal(got, want) {
		t.Fatalf("stops = %q, want %q", got, want)
	}
}

func TestMergeCatalogsGivesWorkbookPriority(t *testing.T) {
	positions := []Position{
		{RouteSourceID: "4417", PatternKey: "3000001", StopSequence: 1, StopSourceID: "2594"},
		{RouteSourceID: "4417", PatternKey: "3000001", StopSequence: 2, StopSourceID: "7879"},
	}
	for _, tt := range []struct {
		name      string
		positions []Position
		want      []string
	}{
		{"with positions", positions, []string{
			"route 17: the workbook has this route; OSM relations 101 and 102 are skipped",
		}},
		{"without positions", nil, []string{
			"route 17: the workbook has this route; OSM relations 101 and 102 are skipped",
			"route 17: the workbook has no positions for this route, so it has no geography although OSM has relations 101 and 102",
		}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			workbook := Catalog{
				Routes:    []Route{{Number: 17, Name: "Из книги", SourceID: "4417"}},
				Stops:     []Stop{{SourceID: "2594", Name: "А"}, {SourceID: "7879", Name: "Б"}},
				Positions: tt.positions,
			}
			merged, warnings := mergeCatalogs(workbook, parseValidOSM(t))
			if !slices.Equal(warnings, tt.want) {
				t.Fatalf("warnings = %q\nwant %q", warnings, tt.want)
			}
			if len(merged.Routes) != 2 || merged.Routes[0] != workbook.Routes[0] || merged.Routes[1].Number != 25 {
				t.Fatalf("routes = %+v", merged.Routes)
			}
			// Only route 25 is taken from OSM; it keeps the nodes it shares with
			// route 17, and no stop of route 17 alone is left behind.
			if got := stopSourceIDs(merged.Stops); !slices.Equal(got, []string{"2594", "7879", "osm:node/1", "osm:node/6", "osm:node/5"}) {
				t.Fatalf("stops = %q", got)
			}
			if len(merged.Positions) != len(tt.positions)+4 {
				t.Fatalf("positions = %+v", merged.Positions)
			}
		})
	}
}

// committedSnapshot is the snapshot in data/osm at the repository root,
// relative to this package directory.
const committedSnapshot = "../../../../../data/osm/tram_routes.json"

// committedSnapshotBase is timestamp_osm_base of the committed snapshot; it
// changes with every refresh.
const committedSnapshotBase = "2026-09-26T22:39:54Z"

// parseCommittedOSM reads the committed snapshot with the real repository and
// validates it against OSMRoutes.
func parseCommittedOSM(t *testing.T) (catalog_osm_repository.Snapshot, osmCatalog) {
	t.Helper()
	snapshot, err := catalog_osm_repository.New().ReadSnapshot(context.Background(), committedSnapshot)
	if err != nil {
		t.Fatal(err)
	}
	osm, issues := parseOSM(snapshot, OSMRoutes)
	if len(issues) > 0 {
		t.Fatalf("issues = %+v", issues)
	}
	return snapshot, osm
}

// TestParseCommittedSnapshot runs the committed snapshot through the real
// repository, validation and merge.
func TestParseCommittedSnapshot(t *testing.T) {
	snapshot, osm := parseCommittedOSM(t)
	catalog, warnings := mergeCatalogs(Catalog{}, osm)
	if len(warnings) != 0 {
		t.Errorf("warnings = %q", warnings)
	}
	if snapshot.BaseTimestamp != committedSnapshotBase {
		t.Errorf("timestamp_osm_base = %q", snapshot.BaseTimestamp)
	}

	wantRoutes := []Route{
		{Number: 17, Name: "Усадьба Останкино - Медведково", SourceID: "osm:relation/1371410"},
		{Number: 25, Name: `Усадьба Останкино - Метро "Сокольники"`, SourceID: "osm:relation/3186266"},
		{Number: 26, Name: `Метро "Университет" - Метро "Октябрьская"`, SourceID: "osm:relation/1689065"},
		{Number: 28, Name: `Проспект Маршала Жукова - Метро "Сокол"`, SourceID: "osm:relation/3184024"},
		{Number: 50, Name: `Дом культуры "Компрессор" - Метро "Новослободская"`, SourceID: "osm:relation/1538180"},
	}
	if !slices.Equal(catalog.Routes, wantRoutes) {
		t.Errorf("routes = %+v\nwant %+v", catalog.Routes, wantRoutes)
	}
	if len(catalog.Positions) != 257 || len(catalog.Stops) != 242 {
		t.Errorf("positions = %d, stops = %d; want 257 and 242", len(catalog.Positions), len(catalog.Stops))
	}
	patterns := make(map[string]int16)
	for _, position := range catalog.Positions {
		patterns[position.PatternKey] = position.DirectionID
	}
	if len(patterns) != 10 {
		t.Errorf("patterns = %v, want 10", patterns)
	}
	for _, key := range []string{"osm:relation/540033", "osm:relation/3186264", "osm:relation/1689026", "osm:relation/3184023", "osm:relation/1538169"} {
		if direction, found := patterns[key]; !found || direction != 0 {
			t.Errorf("pattern %s: direction %d, found %v; want direction 0", key, direction, found)
		}
	}
}
