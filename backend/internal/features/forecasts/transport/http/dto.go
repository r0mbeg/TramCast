package forecasts_transport_http

import (
	"time"

	core_domain "github.com/r0mbeg/TramCast/backend/internal/core/domain"
	forecasts_sqlc "github.com/r0mbeg/TramCast/backend/internal/features/forecasts/repository/postgres/sqlc"
	forecasts_service "github.com/r0mbeg/TramCast/backend/internal/features/forecasts/service"
)

// queryRequest is the body of POST /api/predictions/query. A missing
// route_id, from or to decodes as zero and fails validation.
type queryRequest struct {
	RouteID int64  `json:"route_id"`
	From    string `json:"from"`
	To      string `json:"to"`
	// ForecastVersionID is nil when absent or null: the active version.
	ForecastVersionID *string `json:"forecast_version_id"`
}

type versionResponse struct {
	ID             string `json:"id"`
	ModelVersion   string `json:"model_version"`
	DatasetVersion string `json:"dataset_version"`
	HistoryEnd     string `json:"history_end"`
	ForecastFrom   string `json:"forecast_from"`
	ForecastTo     string `json:"forecast_to"`
	Timezone       string `json:"timezone"`
}

type sliceResponse struct {
	ForecastVersionID string     `json:"forecast_version_id"`
	RouteID           int64      `json:"route_id"`
	Timezone          string     `json:"timezone"`
	From              string     `json:"from"`
	To                string     `json:"to"`
	Points            []pointDTO `json:"points"`
}

type pointDTO struct {
	Date string `json:"date"`
	// Weekday is ISO: 1 is Monday, 7 Sunday.
	Weekday   int   `json:"weekday"`
	Hour      int   `json:"hour"`
	Boardings int64 `json:"boardings"`
}

type jobResponse struct {
	JobID             string `json:"job_id"`
	ForecastVersionID string `json:"forecast_version_id"`
	RouteID           int64  `json:"route_id"`
	Status            string `json:"status"`
	// PollIntervalSeconds is null for a finished job.
	PollIntervalSeconds *int `json:"poll_interval_seconds"`
}

// failedResponse is the error of a query whose job failed for good.
type failedResponse struct {
	Error string `json:"error"`
	JobID string `json:"job_id"`
}

// moscow formats t in the forecast zone, whatever location pgx returns.
func moscow(t time.Time) string {
	return t.In(core_domain.Moscow).Format(time.RFC3339)
}

func newVersionResponse(version forecasts_sqlc.ForecastVersion) versionResponse {
	return versionResponse{
		ID:             version.ID.String(),
		ModelVersion:   version.ModelVersion,
		DatasetVersion: version.DatasetVersion,
		HistoryEnd:     moscow(version.HistoryEnd.Time),
		ForecastFrom:   moscow(version.ForecastFrom.Time),
		ForecastTo:     moscow(version.ForecastTo.Time),
		Timezone:       version.Timezone,
	}
}

func newSliceResponse(slice forecasts_service.Slice) sliceResponse {
	response := sliceResponse{
		ForecastVersionID: slice.Version.ID.String(),
		RouteID:           slice.RouteID,
		Timezone:          slice.Version.Timezone,
		From:              moscow(slice.From),
		To:                moscow(slice.To),
		Points:            make([]pointDTO, len(slice.Points)),
	}
	for i, point := range slice.Points {
		local := point.HourStart.In(core_domain.Moscow)
		weekday := int(local.Weekday())
		if weekday == 0 {
			weekday = 7
		}
		response.Points[i] = pointDTO{
			Date:      local.Format(time.DateOnly),
			Weekday:   weekday,
			Hour:      local.Hour(),
			Boardings: point.Boardings,
		}
	}
	return response
}

func newJobResponse(job forecasts_sqlc.PredictionJob) jobResponse {
	response := jobResponse{
		JobID:             job.ID.String(),
		ForecastVersionID: job.ForecastVersionID.String(),
		RouteID:           job.RouteID,
		Status:            job.Status,
	}
	if job.Status == "queued" || job.Status == "running" {
		seconds := int(pollInterval / time.Second)
		response.PollIntervalSeconds = &seconds
	}
	return response
}
