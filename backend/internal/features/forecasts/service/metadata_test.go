package forecasts_service

import (
	"os"
	"slices"
	"strings"
	"testing"
	"time"
	"unicode/utf16"
)

// describeJSON has the layout of `service.py --describe` (json.dumps with
// indent=2) for a GPU recipe.
const describeJSON = `{
  "timezone": "Europe/Moscow",
  "route_numbers": [
    1,
    5,
    7,
    11,
    12,
    17,
    25,
    26,
    28,
    50
  ],
  "history_end": "2025-11-01T00:00:00+03:00",
  "forecast_from": "2025-11-01T00:00:00+03:00",
  "forecast_to": "2026-01-01T00:00:00+03:00",
  "model_version": "tabpfn030-0123456789abcdef",
  "dataset_version": "prepared-844b17f7f0a0d34e203758ba3c1158ef820ae416c8c09b21fc4cd4c5c4c1898e"
}
`

// replayBundle is the committed replay manifest, read by a path relative to
// this package like the OSM snapshot tests.
const replayBundle = "../../../../../ml/bundles/030/forecast_bundle.json"

var (
	novemberStart = time.Date(2025, 11, 1, 0, 0, 0, 0, time.UTC).Add(-3 * time.Hour)
	januaryStart  = time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC).Add(-3 * time.Hour)
	targetRoutes  = []int16{1, 5, 7, 11, 12, 17, 25, 26, 28, 50}
)

func checkContestSpec(t *testing.T, spec VersionSpec) {
	t.Helper()
	if spec.Timezone != "Europe/Moscow" || !slices.Equal(spec.RouteNumbers, targetRoutes) ||
		!spec.HistoryEnd.Equal(novemberStart) || !spec.ForecastFrom.Equal(novemberStart) || !spec.ForecastTo.Equal(januaryStart) {
		t.Fatalf("spec = %+v, want the contest period and routes", spec)
	}
}

func TestParseVersionMetadataDescribe(t *testing.T) {
	// --warm-cache prints the same metadata on one line with a points count.
	warmCache := strings.Replace(strings.Join(strings.Fields(describeJSON), ""), "{", `{"points":14640,`, 1)
	for name, data := range map[string]string{"describe": describeJSON, "warm cache": warmCache} {
		t.Run(name, func(t *testing.T) {
			spec, err := ParseVersionMetadata([]byte(data))
			if err != nil {
				t.Fatal(err)
			}
			checkContestSpec(t, spec)
			if spec.ModelVersion != "tabpfn030-0123456789abcdef" || !strings.HasPrefix(spec.DatasetVersion, "prepared-844b") || spec.Replay {
				t.Fatalf("spec = %+v, want the recipe IDs without replay", spec)
			}
		})
	}
}

func TestParseVersionMetadataReplayBundle(t *testing.T) {
	data, err := os.ReadFile(replayBundle)
	if err != nil {
		t.Fatal(err)
	}
	spec, err := ParseVersionMetadata(data)
	if err != nil {
		t.Fatal(err)
	}
	checkContestSpec(t, spec)
	if spec.ModelVersion != "tabpfn030-zhores8477154-replay" ||
		spec.DatasetVersion != "prepared-844b17f7f0a0d34e203758ba3c1158ef820ae416c8c09b21fc4cd4c5c4c1898e" || !spec.Replay {
		t.Fatalf("spec = %+v, want the replay IDs marked as replay", spec)
	}

	// Either replay key marks the metadata, whatever its value.
	for _, key := range []string{`"serving_mode": "precomputed"`, `"prediction_file": "forecast.csv"`, `"serving_mode": null`} {
		spec, err := ParseVersionMetadata([]byte(strings.Replace(describeJSON, "{", "{"+key+",", 1)))
		if err != nil || !spec.Replay {
			t.Errorf("%s: replay = %t, error = %v; want replay", key, spec.Replay, err)
		}
	}
}

func TestParseVersionMetadataSkipsUTF8BOM(t *testing.T) {
	// PowerShell 5.1: Out-File -Encoding utf8 writes a BOM and CRLF.
	data := append([]byte{0xEF, 0xBB, 0xBF}, strings.ReplaceAll(describeJSON, "\n", "\r\n")...)
	spec, err := ParseVersionMetadata(data)
	if err != nil {
		t.Fatal(err)
	}
	checkContestSpec(t, spec)
}

func TestParseVersionMetadataRejectsUTF16(t *testing.T) {
	units := utf16.Encode([]rune(describeJSON))
	little := []byte{0xFF, 0xFE}
	big := []byte{0xFE, 0xFF}
	for _, unit := range units {
		little = append(little, byte(unit), byte(unit>>8))
		big = append(big, byte(unit>>8), byte(unit))
	}
	for name, data := range map[string][]byte{"little endian": little, "big endian": big} {
		t.Run(name, func(t *testing.T) {
			_, err := ParseVersionMetadata(data)
			if err == nil || !strings.Contains(err.Error(), "UTF-16") || !strings.Contains(err.Error(), "Out-File -Encoding utf8") {
				t.Fatalf("error = %v, want a UTF-16 hint", err)
			}
		})
	}
}

func TestParseVersionMetadataRejectsInvalid(t *testing.T) {
	for _, tt := range []struct {
		name string
		data string
		want string
	}{
		{"not JSON", "model_version=x", "decode metadata JSON"},
		{"array", "[]", "decode metadata JSON"},
		{"trailing data", describeJSON + "{}", "decode metadata JSON"},
		{"route out of range", strings.Replace(describeJSON, "50\n", "70000\n", 1), "decode metadata JSON"},
		{"timestamp as number", strings.Replace(describeJSON, `"2026-01-01T00:00:00+03:00"`, "1767214800", 1), "decode metadata JSON"},
		{"missing timestamp", strings.Replace(describeJSON, `"history_end"`, `"history"`, 1), "history_end"},
		{"date only", strings.Replace(describeJSON, `"2026-01-01T00:00:00+03:00"`, `"2026-01-01"`, 1), "forecast_to"},
		{"no offset", strings.Replace(describeJSON, `"forecast_from": "2025-11-01T00:00:00+03:00"`, `"forecast_from": "2025-11-01T00:00:00"`, 1), "forecast_from"},
		{"space separator", strings.Replace(describeJSON, `"2026-01-01T00:00:00+03:00"`, `"2026-01-01 00:00:00+03:00"`, 1), "forecast_to"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ParseVersionMetadata([]byte(tt.data))
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error = %v, want one mentioning %q", err, tt.want)
			}
		})
	}
}
