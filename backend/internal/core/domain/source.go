// Package core_domain holds business rules shared by features.
package core_domain

import (
	"strconv"
	"strings"
)

// OSMPrefix starts every source identifier taken from OpenStreetMap. Workbook
// identifiers are digits only, so the two namespaces cannot collide.
const OSMPrefix = "osm:"

// OSMNodeID is the source identifier of an OSM node, such as a stop position.
// OSM IDs exceed 32 bits, so they are formatted from int64.
func OSMNodeID(id int64) string {
	return OSMPrefix + "node/" + strconv.FormatInt(id, 10)
}

// OSMRelationID is the source identifier of an OSM relation: a route variant
// or the route_master that groups them.
func OSMRelationID(id int64) string {
	return OSMPrefix + "relation/" + strconv.FormatInt(id, 10)
}

// IsOSM reports whether a source identifier comes from OpenStreetMap.
func IsOSM(sourceID string) bool {
	return strings.HasPrefix(sourceID, OSMPrefix)
}
