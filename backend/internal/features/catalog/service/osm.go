package catalog_service

import (
	"errors"
	"fmt"
	"math"
	"slices"
	"strconv"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"

	core_domain "github.com/r0mbeg/TramCast/backend/internal/core/domain"
	catalog_osm_repository "github.com/r0mbeg/TramCast/backend/internal/features/catalog/repository/osm"
)

// maxStopGapMeters rejects a relation whose consecutive stops are further
// apart, a sign of a missing or misplaced stop: the snapshot's largest gap is
// 841 m, the workbook's 2.1 km.
const maxStopGapMeters = 3000

// earthRadiusMeters is the mean Earth radius used by distanceMeters.
const earthRadiusMeters = 6_371_008.8

var (
	// osmStopRoles mark the stop positions of a PTv2 route, in travel order.
	osmStopRoles = []string{"stop", "stop_entry_only", "stop_exit_only"}
	// osmPlatformRoles are skipped: platforms of any member type duplicate the
	// stops next to them.
	osmPlatformRoles = []string{"platform", "platform_entry_only", "platform_exit_only"}
)

// osmNameQuotes turns the typographic quotes of OSM names into the ASCII
// quotes the workbook uses.
var osmNameQuotes = strings.NewReplacer("«", `"`, "»", `"`, "„", `"`, "“", `"`, "”", `"`)

// normalizeOSMName aligns an OSM name with the workbook style: ASCII quotes,
// no surrounding spaces and a capital first letter. Other letters, including
// ё, are kept.
func normalizeOSMName(name string) string {
	name = strings.TrimSpace(osmNameQuotes.Replace(name))
	first, size := utf8.DecodeRuneInString(name)
	if unicode.IsLower(first) {
		return string(unicode.ToUpper(first)) + name[size:]
	}
	return name
}

// osmRoute is one validated route of the snapshot with both directions.
type osmRoute struct {
	// Route is named after the first and last stop of direction 0, as in the
	// workbook; SourceID is the route_master relation.
	Route     Route
	Relations [2]int64
	Positions []Position
}

// osmCatalog is the validated content of the snapshot.
type osmCatalog struct {
	// Routes are ordered by route number.
	Routes []osmRoute
	// Stops holds every stop node of the listed relations by source ID; the
	// merge keeps only the ones its positions reference.
	Stops    map[string]Stop
	Warnings []string
}

type osmParser struct {
	issues   []Issue
	warnings []string
	nodes    map[int64]catalog_osm_repository.Element
	// relations are keyed by ID; relationIDs keeps the snapshot order, so the
	// output does not depend on map iteration.
	relations   map[int64]catalog_osm_repository.Element
	relationIDs []int64
	stops       map[string]Stop
	// badNodes are nodes already reported, so a node shared by several
	// relations is reported once.
	badNodes map[int64]bool
}

func (p *osmParser) add(relation int64, member int, format string, args ...any) {
	p.issues = append(p.issues, Issue{OSM: true, Relation: relation, Member: member, Message: fmt.Sprintf(format, args...)})
}

func (p *osmParser) warn(format string, args ...any) {
	p.warnings = append(p.warnings, fmt.Sprintf(format, args...))
}

// parseOSM checks the whole snapshot against routes (route number → relation
// IDs by direction_id) and returns the catalog only when there are no issues.
// Routes the workbook takes priority for are checked as well: the snapshot is
// a committed file, so any problem needs a fix rather than a skip.
func parseOSM(snapshot catalog_osm_repository.Snapshot, routes map[int16][2]int64) (osmCatalog, []Issue) {
	p := osmParser{
		nodes:     make(map[int64]catalog_osm_repository.Element),
		relations: make(map[int64]catalog_osm_repository.Element),
		stops:     make(map[string]Stop),
		badNodes:  make(map[int64]bool),
	}
	// Overpass answers a failed or timed-out query with HTTP 200, a remark
	// and partial elements.
	if snapshot.Remark != "" {
		p.add(0, 0, "Overpass remark %q: the query failed or was cut short", snapshot.Remark)
	}
	if snapshot.BaseTimestamp == "" {
		p.add(0, 0, "osm3s.timestamp_osm_base is missing")
	} else if _, err := time.Parse(time.RFC3339, snapshot.BaseTimestamp); err != nil {
		p.add(0, 0, "osm3s.timestamp_osm_base %q is not an RFC 3339 time", snapshot.BaseTimestamp)
	}
	if len(snapshot.Elements) == 0 {
		p.add(0, 0, "snapshot has no elements")
	}
	p.index(snapshot.Elements)

	numbers := make([]int16, 0, len(routes))
	for number := range routes {
		numbers = append(numbers, number)
	}
	slices.Sort(numbers)
	p.checkListing(numbers, routes)

	var catalog osmCatalog
	for _, number := range numbers {
		if route, ok := p.route(number, routes[number]); ok {
			catalog.Routes = append(catalog.Routes, route)
		}
	}
	if len(p.issues) > 0 {
		return osmCatalog{}, p.issues
	}
	catalog.Stops, catalog.Warnings = p.stops, p.warnings
	return catalog, nil
}

// index keys nodes and relations by ID. The query returns nothing else, so
// other element types mean a changed query the importer does not support.
func (p *osmParser) index(elements []catalog_osm_repository.Element) {
	var others []string
	for _, element := range elements {
		switch element.Type {
		case "node":
			if _, repeated := p.nodes[element.ID]; repeated {
				p.add(0, 0, "node %d appears more than once", element.ID)
				continue
			}
			p.nodes[element.ID] = element
		case "relation":
			if _, repeated := p.relations[element.ID]; repeated {
				p.add(element.ID, 0, "relation appears more than once")
				continue
			}
			p.relations[element.ID] = element
			p.relationIDs = append(p.relationIDs, element.ID)
		default:
			others = append(others, fmt.Sprintf("%s %d", element.Type, element.ID))
		}
	}
	if len(others) > 0 {
		p.add(0, 0, "%d element(s) are neither nodes nor relations, first %s", len(others), others[0])
	}
}

// checkListing requires every route relation of the snapshot to be listed
// exactly once. Other relations are ignored with a warning, except
// route_master relations that group a listed relation: master checks those.
func (p *osmParser) checkListing(numbers []int16, routes map[int16][2]int64) {
	type listing struct{ number, direction int }
	listed := make(map[int64]listing)
	for _, number := range numbers {
		for direction, id := range routes[number] {
			if first, repeated := listed[id]; repeated {
				p.add(id, 0, "relation is listed for route %d direction %d and route %d direction %d",
					first.number, first.direction, number, direction)
				continue
			}
			listed[id] = listing{number: int(number), direction: direction}
		}
	}
	for _, id := range p.relationIDs {
		if _, found := listed[id]; found {
			continue
		}
		relation := p.relations[id]
		switch relation.Tags["type"] {
		case "route":
			p.add(id, 0, "route relation with ref %q is not listed in OSMRoutes", relation.Tags["ref"])
		case "route_master":
			if !slices.ContainsFunc(relation.Members, func(m catalog_osm_repository.Member) bool {
				_, found := listed[m.Ref]
				return m.Type == "relation" && found
			}) {
				p.warn("OSM relation %d: route_master groups none of the listed relations; it is ignored", id)
			}
		default:
			p.warn("OSM relation %d: type %q is not used; it is ignored", id, relation.Tags["type"])
		}
	}
}

// route validates both directions of one route and its route_master, then
// builds positions with stop_sequence from 1 in member order.
func (p *osmParser) route(number int16, ids [2]int64) (osmRoute, bool) {
	issuesBefore := len(p.issues)
	valid := true
	var stops [2][]Stop
	for direction, id := range ids {
		relation, found := p.relations[id]
		if !found {
			p.add(id, 0, "relation of route %d direction %d is not in the snapshot", number, direction)
			valid = false
			continue
		}
		p.checkTags(relation, map[string]string{
			"type":                     "route",
			"route":                    "tram",
			"public_transport:version": "2",
			"ref":                      strconv.Itoa(int(number)),
		})
		var complete bool
		stops[direction], complete = p.stopsOf(relation)
		// Problems of stop members are already reported; the count matters only
		// for a relation without them.
		switch {
		case !complete:
			valid = false
		case len(stops[direction]) < 2:
			p.add(id, 0, "relation has %d stop(s), want at least 2", len(stops[direction]))
		}
	}
	master, masterOK := p.master(number, ids)
	// A node reported for another relation adds no issue here, hence valid.
	if !valid || !masterOK || len(p.issues) > issuesBefore {
		return osmRoute{}, false
	}

	routeSourceID := core_domain.OSMRelationID(master)
	route := osmRoute{Relations: ids}
	for direction, id := range ids {
		patternKey := core_domain.OSMRelationID(id)
		for i, stop := range stops[direction] {
			route.Positions = append(route.Positions, Position{
				RouteSourceID: routeSourceID,
				PatternKey:    patternKey,
				DirectionID:   int16(direction),
				StopSequence:  int32(i + 1),
				StopSourceID:  stop.SourceID,
			})
		}
	}
	first, last := stops[0][0], stops[0][len(stops[0])-1]
	route.Route = Route{Number: number, Name: first.Name + " - " + last.Name, SourceID: routeSourceID}
	return route, true
}

// checkTags reports every tag of the relation that differs from want, in key order.
func (p *osmParser) checkTags(relation catalog_osm_repository.Element, want map[string]string) {
	keys := make([]string, 0, len(want))
	for key := range want {
		keys = append(keys, key)
	}
	slices.Sort(keys)
	for _, key := range keys {
		if value := relation.Tags[key]; value != want[key] {
			p.add(relation.ID, 0, "tag %s is %q, want %q", key, value, want[key])
		}
	}
}

// stopsOf returns the valid stops of a route relation in member order and
// whether every member was valid. Platforms and track ways are skipped; any
// other member is an issue.
func (p *osmParser) stopsOf(relation catalog_osm_repository.Element) ([]Stop, bool) {
	var stops []Stop
	complete := true
	// previousRef is the node of the previous stop member, if any; previous is
	// set only when that node is valid, so gaps are measured between valid stops.
	var previousRef *int64
	var previous *Stop
	for i, member := range relation.Members {
		position := i + 1
		switch {
		case slices.Contains(osmStopRoles, member.Role):
			if member.Type != "node" {
				p.add(relation.ID, position, "role %q needs a node, got %s %d", member.Role, member.Type, member.Ref)
				complete = false
				continue
			}
			if previousRef != nil && *previousRef == member.Ref {
				p.add(relation.ID, position, "node %d repeats the previous stop", member.Ref)
				complete = false
				continue
			}
			stop, ok := p.stop(relation.ID, position, member.Ref)
			if ok && previous != nil {
				if gap := distanceMeters(*previous, stop); gap > maxStopGapMeters {
					p.add(relation.ID, position, "node %d is %.1f km from the previous stop node %d; the limit is %d km",
						member.Ref, gap/1000, *previousRef, maxStopGapMeters/1000)
					complete = false
				}
			}
			previousRef, previous = &member.Ref, nil
			if !ok {
				complete = false
				continue
			}
			stops = append(stops, stop)
			previous = &stop
		case slices.Contains(osmPlatformRoles, member.Role):
		case member.Role == "" && member.Type == "way":
			// Track ways: the MVP draws lines between stops, not rail geometry.
		default:
			p.add(relation.ID, position, "%s %d has role %q; only stops, platforms and track ways are supported",
				member.Type, member.Ref, member.Role)
			complete = false
		}
	}
	return stops, complete
}

// stop validates a stop node once. A problem is reported at the first member
// that references the node; later references reuse the result.
func (p *osmParser) stop(relation int64, member int, id int64) (Stop, bool) {
	sourceID := core_domain.OSMNodeID(id)
	if stop, found := p.stops[sourceID]; found {
		return stop, true
	}
	if p.badNodes[id] {
		return Stop{}, false
	}
	issuesBefore := len(p.issues)
	node, found := p.nodes[id]
	if !found {
		p.add(relation, member, "node %d is not in the snapshot", id)
		p.badNodes[id] = true
		return Stop{}, false
	}
	if value := node.Tags["public_transport"]; value != "stop_position" {
		p.add(relation, member, "node %d has public_transport=%q, want %q", id, value, "stop_position")
	}
	name := normalizeOSMName(node.Tags["name"])
	if name == "" {
		p.add(relation, member, "node %d has no name", id)
	}
	latitude, err := osmCoordinate(node.Lat, 90)
	if err != nil {
		p.add(relation, member, "node %d lat %s", id, err)
	}
	longitude, err := osmCoordinate(node.Lon, 180)
	if err != nil {
		p.add(relation, member, "node %d lon %s", id, err)
	}
	if len(p.issues) > issuesBefore {
		p.badNodes[id] = true
		return Stop{}, false
	}
	stop := Stop{SourceID: sourceID, Name: name, Latitude: latitude, Longitude: longitude}
	p.stops[sourceID] = stop
	return stop, true
}

func osmCoordinate(value *float64, limit float64) (float64, error) {
	switch {
	case value == nil:
		return 0, errors.New("is missing")
	case math.IsNaN(*value) || math.IsInf(*value, 0):
		return 0, fmt.Errorf("%v is not a finite number", *value)
	case *value < -limit || *value > limit:
		return 0, fmt.Errorf("%v is outside [-%g, %g]", *value, limit, limit)
	}
	return *value, nil
}

// master finds the single route_master relation that groups both directions
// of a route and returns its ID. Other members of that relation are ignored
// with a warning.
func (p *osmParser) master(number int16, ids [2]int64) (int64, bool) {
	groups := func(relation catalog_osm_repository.Element, id int64) bool {
		return slices.ContainsFunc(relation.Members, func(m catalog_osm_repository.Member) bool {
			return m.Type == "relation" && m.Ref == id
		})
	}
	var masters []int64
	for _, id := range p.relationIDs {
		relation := p.relations[id]
		if relation.Tags["type"] == "route_master" && (groups(relation, ids[0]) || groups(relation, ids[1])) {
			masters = append(masters, id)
		}
	}
	switch len(masters) {
	case 0:
		p.add(ids[0], 0, "no route_master relation groups relations %d and %d of route %d", ids[0], ids[1], number)
		return 0, false
	case 1:
	default:
		p.add(ids[0], 0, "relations %d and %d of route %d belong to %d route_master relations %v",
			ids[0], ids[1], number, len(masters), masters)
		return 0, false
	}

	master := p.relations[masters[0]]
	issuesBefore := len(p.issues)
	for _, id := range ids {
		if !groups(master, id) {
			p.add(master.ID, 0, "route_master does not group relation %d of route %d", id, number)
		}
	}
	p.checkTags(master, map[string]string{"route_master": "tram", "ref": strconv.Itoa(int(number))})
	var extra []string
	for _, member := range master.Members {
		if member.Type != "relation" || (member.Ref != ids[0] && member.Ref != ids[1]) {
			extra = append(extra, fmt.Sprintf("%s %d", member.Type, member.Ref))
		}
	}
	if len(extra) > 0 {
		p.warn("route %d: route_master %d also groups %s; ignored", number, master.ID, strings.Join(extra, ", "))
	}
	return master.ID, len(p.issues) == issuesBefore
}

// distanceMeters is the great-circle distance between two stops.
func distanceMeters(a, b Stop) float64 {
	toRadians := func(degrees float64) float64 { return degrees * math.Pi / 180 }
	lat1, lat2 := toRadians(a.Latitude), toRadians(b.Latitude)
	dLat, dLon := lat2-lat1, toRadians(b.Longitude-a.Longitude)
	h := math.Sin(dLat/2)*math.Sin(dLat/2) + math.Cos(lat1)*math.Cos(lat2)*math.Sin(dLon/2)*math.Sin(dLon/2)
	return 2 * earthRadiusMeters * math.Asin(math.Sqrt(h))
}

// mergeCatalogs adds the OSM routes the workbook lacks. The workbook has
// priority by route number, even when it has no positions for the route. OSM
// stops come from the positions kept, so a skipped route leaves no orphan
// stops and a node shared by several relations becomes one stop.
func mergeCatalogs(workbook Catalog, osm osmCatalog) (Catalog, []string) {
	warnings := slices.Clone(osm.Warnings)
	workbookRoutes := make(map[int16]string, len(workbook.Routes))
	for _, route := range workbook.Routes {
		workbookRoutes[route.Number] = route.SourceID
	}
	withPositions := make(map[string]bool)
	for _, position := range workbook.Positions {
		withPositions[position.RouteSourceID] = true
	}

	merged := Catalog{
		Routes:    slices.Clone(workbook.Routes),
		Stops:     slices.Clone(workbook.Stops),
		Positions: slices.Clone(workbook.Positions),
	}
	added := make(map[string]bool)
	for _, route := range osm.Routes {
		number := route.Route.Number
		if sourceID, found := workbookRoutes[number]; found {
			warnings = append(warnings, fmt.Sprintf("route %d: the workbook has this route; OSM relations %d and %d are skipped",
				number, route.Relations[0], route.Relations[1]))
			if !withPositions[sourceID] {
				warnings = append(warnings, fmt.Sprintf("route %d: the workbook has no positions for this route, so it has no geography although OSM has relations %d and %d",
					number, route.Relations[0], route.Relations[1]))
			}
			continue
		}
		merged.Routes = append(merged.Routes, route.Route)
		for _, position := range route.Positions {
			merged.Positions = append(merged.Positions, position)
			if !added[position.StopSourceID] {
				added[position.StopSourceID] = true
				merged.Stops = append(merged.Stops, osm.Stops[position.StopSourceID])
			}
		}
	}
	return merged, warnings
}
