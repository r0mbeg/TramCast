// Package forecasts_transport_http exposes forecast versions, published
// slices and prediction jobs over HTTP.
package forecasts_transport_http

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"mime"
	"net/http"
	"strconv"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/jackc/pgx/v5/pgtype"

	core_http_server "github.com/r0mbeg/TramCast/backend/internal/core/transport/http"
	forecasts_sqlc "github.com/r0mbeg/TramCast/backend/internal/features/forecasts/repository/postgres/sqlc"
	forecasts_service "github.com/r0mbeg/TramCast/backend/internal/features/forecasts/service"
)

const (
	// pollInterval is how often a client reads a queued or running job.
	// ponytail: fixed; tune it once GPU compute times are measured.
	pollInterval = 2 * time.Second
	// queueFullRetryAfter is the Retry-After of a full queue in seconds.
	queueFullRetryAfter = "30"
	// maxQueryBody bounds the JSON body of a query.
	maxQueryBody = 4 << 10
)

type Service interface {
	ActiveVersion(ctx context.Context) (forecasts_sqlc.ForecastVersion, error)
	Query(ctx context.Context, q forecasts_service.SliceQuery) (forecasts_service.QueryResult, error)
	Slice(ctx context.Context, q forecasts_service.SliceQuery) (forecasts_service.Slice, error)
	Job(ctx context.Context, id pgtype.UUID) (forecasts_sqlc.PredictionJob, error)
}

type Handler struct {
	service Service
}

func NewHandler(service Service) *Handler {
	return &Handler{service: service}
}

// Register adds the handlers to the API group.
func (h *Handler) Register(router gin.IRouter) {
	router.GET("/forecast-versions/active", h.activeVersion)
	router.POST("/predictions/query", h.query)
	router.GET("/predictions", h.slice)
	router.GET("/prediction-jobs/:job_id", h.job)
}

func (h *Handler) activeVersion(c *gin.Context) {
	version, err := h.service.ActiveVersion(c.Request.Context())
	if err != nil {
		writeServiceError(c, err)
		return
	}
	c.JSON(http.StatusOK, newVersionResponse(version))
}

// query answers with the slice of a succeeded job, or with the job of the
// pair, admitting a new one when there is none.
func (h *Handler) query(c *gin.Context) {
	// A JSON-only endpoint needs a CORS preflight from other origins, so a
	// cross-site form cannot admit jobs with the browser's credentials.
	if mediaType, _, err := mime.ParseMediaType(c.GetHeader("Content-Type")); err != nil || mediaType != "application/json" {
		core_http_server.WriteError(c, http.StatusUnsupportedMediaType, "unsupported_media_type")
		return
	}
	var body queryRequest
	decoder := json.NewDecoder(http.MaxBytesReader(c.Writer, c.Request.Body, maxQueryBody))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&body); err != nil || decoder.Decode(&struct{}{}) != io.EOF {
		core_http_server.WriteError(c, http.StatusBadRequest, "invalid_request")
		return
	}
	q, code := sliceQuery(body.ForecastVersionID, body.RouteID, body.From, body.To)
	if code != "" {
		core_http_server.WriteError(c, http.StatusBadRequest, code)
		return
	}

	result, err := h.service.Query(c.Request.Context(), q)
	switch {
	case err != nil:
		writeServiceError(c, err)
	case result.Slice != nil:
		c.JSON(http.StatusOK, newSliceResponse(*result.Slice))
	case result.Job.Status == "failed":
		// Failed is terminal: the client must not poll it again.
		c.AbortWithStatusJSON(http.StatusConflict, failedResponse{Error: "prediction_failed", JobID: result.Job.ID.String()})
	default:
		c.JSON(http.StatusAccepted, newJobResponse(result.Job))
	}
}

// slice reads a published slice and never creates a job.
func (h *Handler) slice(c *gin.Context) {
	versionID, present := c.GetQuery("forecast_version_id")
	if !present {
		core_http_server.WriteError(c, http.StatusBadRequest, "invalid_forecast_version_id")
		return
	}
	routeID, err := strconv.ParseInt(c.Query("route_id"), 10, 64)
	if err != nil {
		core_http_server.WriteError(c, http.StatusBadRequest, "invalid_route_id")
		return
	}
	q, code := sliceQuery(&versionID, routeID, c.Query("from"), c.Query("to"))
	if code != "" {
		core_http_server.WriteError(c, http.StatusBadRequest, code)
		return
	}
	slice, err := h.service.Slice(c.Request.Context(), q)
	if err != nil {
		writeServiceError(c, err)
		return
	}
	c.JSON(http.StatusOK, newSliceResponse(slice))
}

// job reports the status of a job; reading it starts no work.
func (h *Handler) job(c *gin.Context) {
	id, err := forecasts_service.ParseUUID(c.Param("job_id"))
	if err != nil {
		core_http_server.WriteError(c, http.StatusBadRequest, "invalid_job_id")
		return
	}
	job, err := h.service.Job(c.Request.Context(), id)
	if err != nil {
		writeServiceError(c, err)
		return
	}
	c.JSON(http.StatusOK, newJobResponse(job))
}

// sliceQuery checks the syntax of the parameters and returns the error code
// of the first bad one. A nil versionID leaves the choice of the version to
// the service, which also checks the interval rules.
func sliceQuery(versionID *string, routeID int64, from, to string) (forecasts_service.SliceQuery, string) {
	q := forecasts_service.SliceQuery{RouteID: routeID}
	if versionID != nil {
		id, err := forecasts_service.ParseUUID(*versionID)
		if err != nil {
			return q, "invalid_forecast_version_id"
		}
		q.VersionID = id
	}
	if routeID <= 0 {
		return q, "invalid_route_id"
	}
	var err error
	if q.From, err = time.Parse(time.RFC3339, from); err != nil {
		return q, "invalid_interval"
	}
	if q.To, err = time.Parse(time.RFC3339, to); err != nil {
		return q, "invalid_interval"
	}
	return q, ""
}

// writeServiceError maps the errors of the service to API codes; any other
// error is internal and stays in the log.
func writeServiceError(c *gin.Context, err error) {
	switch {
	case errors.Is(err, forecasts_service.ErrInvalidInterval):
		core_http_server.WriteError(c, http.StatusBadRequest, "invalid_interval")
	case errors.Is(err, forecasts_service.ErrVersionNotFound):
		core_http_server.WriteError(c, http.StatusNotFound, "forecast_version_not_found")
	case errors.Is(err, forecasts_service.ErrRouteNotFound):
		core_http_server.WriteError(c, http.StatusNotFound, "route_not_found")
	case errors.Is(err, forecasts_service.ErrJobNotFound):
		core_http_server.WriteError(c, http.StatusNotFound, "prediction_job_not_found")
	case errors.Is(err, forecasts_service.ErrPredictionNotReady):
		core_http_server.WriteError(c, http.StatusConflict, "prediction_not_ready")
	case errors.Is(err, forecasts_service.ErrVersionInactive):
		core_http_server.WriteError(c, http.StatusConflict, "forecast_version_inactive")
	case errors.Is(err, forecasts_service.ErrForecastDisabled):
		core_http_server.WriteError(c, http.StatusUnprocessableEntity, "forecast_disabled")
	case errors.Is(err, forecasts_service.ErrQueueFull):
		c.Header("Retry-After", queueFullRetryAfter)
		core_http_server.WriteError(c, http.StatusServiceUnavailable, "queue_full")
	case errors.Is(err, forecasts_service.ErrNoActiveVersion):
		core_http_server.WriteError(c, http.StatusServiceUnavailable, "no_active_forecast_version")
	default:
		core_http_server.WriteInternalError(c, err)
	}
}
