package core_domain

import "testing"

func TestOSMIDs(t *testing.T) {
	// Node IDs already exceed 2^31; they must not turn into exponent notation.
	if got := OSMNodeID(13725188937); got != "osm:node/13725188937" {
		t.Errorf("node ID = %q", got)
	}
	if got := OSMRelationID(540033); got != "osm:relation/540033" {
		t.Errorf("relation ID = %q", got)
	}
}

func TestIsOSM(t *testing.T) {
	for sourceID, want := range map[string]bool{
		"osm:node/444863659":   true,
		"osm:relation/1371410": true,
		"2040920":              false,
		"":                     false,
		"OSM:node/1":           false,
	} {
		if got := IsOSM(sourceID); got != want {
			t.Errorf("IsOSM(%q) = %v, want %v", sourceID, got, want)
		}
	}
}
