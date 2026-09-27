package forecasts_service

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"time"
)

// versionMetadata is the JSON printed by `service.py --describe` and stored in
// replay bundles as forecast_bundle.json. Other keys are ignored.
type versionMetadata struct {
	Timezone       string  `json:"timezone"`
	RouteNumbers   []int16 `json:"route_numbers"`
	HistoryEnd     string  `json:"history_end"`
	ForecastFrom   string  `json:"forecast_from"`
	ForecastTo     string  `json:"forecast_to"`
	ModelVersion   string  `json:"model_version"`
	DatasetVersion string  `json:"dataset_version"`
	// Only replay bundles have these keys; any value marks one.
	PredictionFile json.RawMessage `json:"prediction_file"`
	ServingMode    json.RawMessage `json:"serving_mode"`
}

var utf8BOM = []byte{0xEF, 0xBB, 0xBF}

// ParseVersionMetadata decodes ML version metadata. It checks the format only;
// CheckSpec checks the values. A UTF-8 BOM, which PowerShell 5.1 writes with
// Out-File -Encoding utf8, is skipped. UTF-16, what its `>` redirection
// writes, is rejected with a hint.
func ParseVersionMetadata(data []byte) (VersionSpec, error) {
	if bytes.HasPrefix(data, []byte{0xFF, 0xFE}) || bytes.HasPrefix(data, []byte{0xFE, 0xFF}) {
		return VersionSpec{}, errors.New("metadata is UTF-16 text; save it as UTF-8: redirect in cmd.exe, " +
			"or pipe to Out-File -Encoding utf8 in PowerShell")
	}
	var metadata versionMetadata
	if err := json.Unmarshal(bytes.TrimPrefix(data, utf8BOM), &metadata); err != nil {
		return VersionSpec{}, fmt.Errorf("decode metadata JSON: %w", err)
	}

	spec := VersionSpec{
		ModelVersion:   metadata.ModelVersion,
		DatasetVersion: metadata.DatasetVersion,
		Timezone:       metadata.Timezone,
		RouteNumbers:   metadata.RouteNumbers,
		Replay:         len(metadata.PredictionFile) > 0 || len(metadata.ServingMode) > 0,
	}
	for _, field := range []struct {
		name  string
		value string
		time  *time.Time
	}{
		{"history_end", metadata.HistoryEnd, &spec.HistoryEnd},
		{"forecast_from", metadata.ForecastFrom, &spec.ForecastFrom},
		{"forecast_to", metadata.ForecastTo, &spec.ForecastTo},
	} {
		parsed, err := time.Parse(time.RFC3339, field.value)
		if err != nil {
			return VersionSpec{}, fmt.Errorf("metadata %s %q is not an RFC 3339 timestamp", field.name, field.value)
		}
		*field.time = parsed
	}
	return spec, nil
}
