// Package catalog_osm_repository reads the OpenStreetMap snapshot of the tram
// routes that the reference workbook lacks. The snapshot is committed in
// data/osm at the repository root; data/osm/README.md describes its source,
// licence and refresh.
package catalog_osm_repository

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
)

// Snapshot is a decoded Overpass API JSON response.
type Snapshot struct {
	Generator string
	// BaseTimestamp is osm3s.timestamp_osm_base: the state of the OSM database
	// the response reflects.
	BaseTimestamp string
	// Remark is set by Overpass when the query failed or was cut short.
	Remark   string
	Elements []Element
}

// Element is a node or relation; ways are not part of the snapshot. IDs of
// different element types are independent and exceed 32 bits.
type Element struct {
	Type string `json:"type"`
	ID   int64  `json:"id"`
	// Lat and Lon are set for nodes only.
	Lat     *float64          `json:"lat"`
	Lon     *float64          `json:"lon"`
	Tags    map[string]string `json:"tags"`
	Members []Member          `json:"members"`
}

// Member is one entry of a relation's ordered member list.
type Member struct {
	Type string `json:"type"`
	Ref  int64  `json:"ref"`
	Role string `json:"role"`
}

type response struct {
	Generator string `json:"generator"`
	OSM3S     struct {
		TimestampOSMBase string `json:"timestamp_osm_base"`
	} `json:"osm3s"`
	Remark   string    `json:"remark"`
	Elements []Element `json:"elements"`
}

type Repository struct{}

func New() *Repository {
	return &Repository{}
}

// ReadSnapshot opens the Overpass JSON file at path and decodes it. Like the
// workbook reader it does no semantic checks; the catalog service validates
// the data.
func (r *Repository) ReadSnapshot(_ context.Context, path string) (Snapshot, error) {
	file, err := os.Open(path)
	if err != nil {
		// The *os.PathError already names the path.
		return Snapshot{}, err
	}
	defer file.Close()
	return Decode(file)
}

// Decode reads one Overpass JSON document. Unknown fields such as version and
// copyright are ignored; data after the document is an error.
func Decode(r io.Reader) (Snapshot, error) {
	decoder := json.NewDecoder(r)
	var decoded response
	if err := decoder.Decode(&decoded); err != nil {
		return Snapshot{}, fmt.Errorf("decode Overpass JSON: %w", err)
	}
	if _, err := decoder.Token(); !errors.Is(err, io.EOF) {
		return Snapshot{}, errors.New("decode Overpass JSON: unexpected data after the document")
	}
	return Snapshot{
		Generator:     decoded.Generator,
		BaseTimestamp: decoded.OSM3S.TimestampOSMBase,
		Remark:        decoded.Remark,
		Elements:      decoded.Elements,
	}, nil
}
