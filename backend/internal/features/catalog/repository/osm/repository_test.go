package catalog_osm_repository

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// committedSnapshot is the snapshot in data/osm at the repository root,
// relative to this package directory.
const committedSnapshot = "../../../../../../data/osm/tram_routes.json"

func TestDecode(t *testing.T) {
	decoded, err := Decode(strings.NewReader(`{
		"version": 0.6,
		"generator": "Overpass API 0.7.62.4",
		"osm3s": {"timestamp_osm_base": "2026-09-26T22:39:54Z", "copyright": "ODbL"},
		"elements": [
			{"type": "relation", "id": 540033, "tags": {"type": "route", "ref": "17"},
			 "members": [{"type": "node", "ref": 13725188937, "role": "stop"}, {"type": "way", "ref": 1, "role": ""}]},
			{"type": "node", "id": 13725188937, "lat": 55.8205, "lon": 37.6158, "tags": {"name": "ВДНХ"}}
		]
	}`))
	if err != nil {
		t.Fatal(err)
	}
	if decoded.Generator != "Overpass API 0.7.62.4" || decoded.BaseTimestamp != "2026-09-26T22:39:54Z" ||
		decoded.Remark != "" || len(decoded.Elements) != 2 {
		t.Fatalf("snapshot = %+v", decoded)
	}
	relation, node := decoded.Elements[0], decoded.Elements[1]
	if relation.Type != "relation" || relation.ID != 540033 || relation.Tags["ref"] != "17" || relation.Lat != nil ||
		!slices.Equal(relation.Members, []Member{{Type: "node", Ref: 13725188937, Role: "stop"}, {Type: "way", Ref: 1}}) {
		t.Errorf("relation = %+v", relation)
	}
	// The ID exceeds 2^31 and must stay exact.
	if node.ID != 13725188937 || node.Lat == nil || *node.Lat != 55.8205 || node.Lon == nil || *node.Lon != 37.6158 ||
		node.Tags["name"] != "ВДНХ" || node.Members != nil {
		t.Errorf("node = %+v", node)
	}
}

func TestDecodeKeepsRemark(t *testing.T) {
	decoded, err := Decode(strings.NewReader(`{"elements": [], "remark": "runtime error: Query timed out"}`))
	if err != nil {
		t.Fatal(err)
	}
	if decoded.Remark != "runtime error: Query timed out" || len(decoded.Elements) != 0 {
		t.Fatalf("snapshot = %+v", decoded)
	}
}

func TestDecodeRejectsMalformedJSON(t *testing.T) {
	for name, input := range map[string]string{
		"truncated":       `{"elements": [`,
		"not JSON":        `<html>504 Gateway Timeout</html>`,
		"fractional ID":   `{"elements": [{"type": "node", "id": 1.5}]}`,
		"textual ID":      `{"elements": [{"type": "node", "id": "1"}]}`,
		"numeric tag":     `{"elements": [{"type": "node", "id": 1, "tags": {"ele": 150}}]}`,
		"second document": `{"elements": []} {"elements": []}`,
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := Decode(strings.NewReader(input)); err == nil {
				t.Fatal("error = nil, want a decoding error")
			}
		})
	}
}

func TestReadSnapshotMissingFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "missing.json")
	_, err := New().ReadSnapshot(context.Background(), path)
	if !errors.Is(err, os.ErrNotExist) || !strings.Contains(err.Error(), path) {
		t.Fatalf("error = %v, want a missing-file error naming the path", err)
	}
}

func TestReadSnapshotRejectsMalformedFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tram_routes.json")
	if err := os.WriteFile(path, []byte(`<html>504 Gateway Timeout</html>`), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := New().ReadSnapshot(context.Background(), path); err == nil || !strings.Contains(err.Error(), "decode Overpass JSON") {
		t.Fatalf("error = %v, want a decoding error", err)
	}
}

func TestCommittedSnapshot(t *testing.T) {
	decoded, err := New().ReadSnapshot(context.Background(), committedSnapshot)
	if err != nil {
		t.Fatal(err)
	}
	// 10 route relations, 5 route_masters and 352 nodes.
	if decoded.BaseTimestamp != "2026-09-26T22:39:54Z" || decoded.Remark != "" || len(decoded.Elements) != 367 {
		t.Fatalf("snapshot: base %q, remark %q, %d elements", decoded.BaseTimestamp, decoded.Remark, len(decoded.Elements))
	}
}

// The snapshot is fetched with "out body": contributor metadata such as user
// names would be personal data and must never be committed.
func TestCommittedSnapshotHasNoContributorMetadata(t *testing.T) {
	snapshot, err := os.ReadFile(committedSnapshot)
	if err != nil {
		t.Fatal(err)
	}
	var document struct {
		Elements []map[string]json.RawMessage `json:"elements"`
	}
	var top map[string]json.RawMessage
	if err := json.Unmarshal(snapshot, &top); err != nil {
		t.Fatal(err)
	}
	for key := range top {
		if !slices.Contains([]string{"version", "generator", "osm3s", "elements"}, key) {
			t.Errorf("unexpected top-level key %q", key)
		}
	}
	if err := json.Unmarshal(snapshot, &document); err != nil {
		t.Fatal(err)
	}
	allowed := []string{"type", "id", "lat", "lon", "tags", "members"}
	for _, element := range document.Elements {
		for key := range element {
			if !slices.Contains(allowed, key) {
				t.Fatalf("element %s has key %q; want only %q", element["id"], key, allowed)
			}
		}
	}
	for _, key := range []string{"user", "uid", "changeset", "remark"} {
		if strings.Contains(string(snapshot), `"`+key+`"`) {
			t.Errorf("snapshot contains %q", key)
		}
	}
}
