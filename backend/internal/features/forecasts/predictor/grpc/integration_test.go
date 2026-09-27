package forecasts_predictor_grpc

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// TestPredictReplayServer calls a running replay of recipe 030, for example
// after `make ml-up ML_MODE=replay`:
//
//	TRAMCAST_TEST_ML_ADDR=127.0.0.1:50051 go -C backend test -count=1 -run Replay ./internal/features/forecasts/predictor/grpc/
func TestPredictReplayServer(t *testing.T) {
	addr := os.Getenv("TRAMCAST_TEST_ML_ADDR")
	if addr == "" {
		t.Skip("TRAMCAST_TEST_ML_ADDR is not set")
	}
	data, err := os.ReadFile(filepath.FromSlash("../../../../../../ml/bundles/030/forecast_bundle.json"))
	if err != nil {
		t.Fatal(err)
	}
	var bundle struct {
		ModelVersion   string    `json:"model_version"`
		DatasetVersion string    `json:"dataset_version"`
		ForecastFrom   time.Time `json:"forecast_from"`
		ForecastTo     time.Time `json:"forecast_to"`
	}
	if err := json.Unmarshal(data, &bundle); err != nil {
		t.Fatal(err)
	}
	client, err := New(addr, time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()

	// Totals of ml/bundles/030/forecast.csv; route 5 is the structural zero.
	for _, tt := range []struct {
		route int16
		sum   int64
	}{{1, 1103274}, {5, 0}} {
		result, err := client.Predict(context.Background(), Request{
			RouteNumber: tt.route, From: bundle.ForecastFrom, To: bundle.ForecastTo,
			ModelVersion: bundle.ModelVersion, DatasetVersion: bundle.DatasetVersion,
		})
		if err != nil {
			t.Fatalf("route %d: %v", tt.route, err)
		}
		var sum int64
		for _, point := range result.Points {
			sum += point.Boardings
		}
		if len(result.Points) != 1464 || sum != tt.sum {
			t.Errorf("route %d: %d points with sum %d, want 1464 with sum %d", tt.route, len(result.Points), sum, tt.sum)
		}
	}
}
