// Package forecasts_predictor_grpc calls the Python forecast service and checks
// its answer against the gRPC contract (proto/README.md) before Go stores it.
package forecasts_predictor_grpc

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/backoff"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/timestamppb"

	core_domain "github.com/r0mbeg/TramCast/backend/internal/core/domain"
	forecastv1 "github.com/r0mbeg/TramCast/backend/internal/gen/tramcast/forecast/v1"
)

// MaxHours is the longest horizon of one call, the limit of the Python server:
// the 61 days of the MVP period.
const MaxHours = 1464

// Request asks for one route over the full horizon of a forecast version.
type Request struct {
	RouteNumber int16
	// From and To are the version horizon, [From, To), on whole Moscow hours.
	From, To time.Time
	// ModelVersion and DatasetVersion are the IDs of the pinned forecast
	// version; the server must report exactly these artifacts.
	ModelVersion   string
	DatasetVersion string
}

// Point is one forecast hour.
type Point struct {
	HourStart time.Time // in core_domain.Moscow
	Boardings int64
}

// Result is a validated response: every hour of [From, To) in order.
type Result struct {
	ModelVersion, DatasetVersion string
	Points                       []Point
}

// Error is a failed Predict call classified for the job queue. Code is stable
// and fit for last_error_code. Err holds diagnostics, possibly the server's
// message, which are for logs and never drive retry decisions.
type Error struct {
	Code      string
	Retryable bool
	Err       error
}

func (e *Error) Error() string {
	if e.Err == nil {
		return e.Code
	}
	return e.Code + ": " + e.Err.Error()
}

func (e *Error) Unwrap() error { return e.Err }

type Client struct {
	conn    *grpc.ClientConn
	client  forecastv1.ForecastServiceClient
	timeout time.Duration
}

// reconnectMaxDelay caps the delay between reconnection attempts, which grows
// to 120 s by default: after an ML restart calls reach it again within ~10 s.
const reconnectMaxDelay = 10 * time.Second

// New prepares a plain-text client for a loopback or private-network address.
// It does not connect: a server that is down surfaces as ml_unavailable from
// Predict. gRPC retries are disabled, since the attempt budget belongs to the
// PostgreSQL job queue. Every Predict call is bounded by timeout.
func New(addr string, timeout time.Duration, opts ...grpc.DialOption) (*Client, error) {
	if timeout <= 0 {
		return nil, errors.New("ML gRPC timeout must be positive")
	}
	reconnect := backoff.DefaultConfig
	reconnect.MaxDelay = reconnectMaxDelay
	options := append([]grpc.DialOption{
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDisableRetry(),
		// 20 s is the gRPC default; a zero value would shorten connect attempts.
		grpc.WithConnectParams(grpc.ConnectParams{Backoff: reconnect, MinConnectTimeout: 20 * time.Second}),
	}, opts...)
	conn, err := grpc.NewClient(addr, options...)
	if err != nil {
		return nil, fmt.Errorf("create ML gRPC client for %s: %w", addr, err)
	}
	return &Client{conn: conn, client: forecastv1.NewForecastServiceClient(conn), timeout: timeout}, nil
}

func (c *Client) Close() error {
	return c.conn.Close()
}

// Predict computes r and accepts the response only as a whole. Every error is
// an *Error. A request that breaks the contract fails as ml_invalid_request
// without reaching the server. The caller's context should not belong to an
// HTTP request: a cold computation may take minutes.
func (c *Client) Predict(ctx context.Context, r Request) (Result, error) {
	hours, err := r.check()
	if err != nil {
		return Result{}, &Error{Code: "ml_invalid_request", Err: err}
	}
	ctx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()
	response, err := c.client.Predict(ctx, &forecastv1.PredictRequest{
		RouteNumber:  int32(r.RouteNumber),
		ForecastFrom: timestamppb.New(r.From),
		ForecastTo:   timestamppb.New(r.To),
	})
	if err != nil {
		return Result{}, statusError(err)
	}
	return r.result(response, hours)
}

// check mirrors the server's validation and returns the number of hours in
// [From, To).
func (r Request) check() (int, error) {
	span := r.To.Sub(r.From)
	switch {
	case r.RouteNumber <= 0:
		return 0, fmt.Errorf("route number %d must be positive", r.RouteNumber)
	case r.ModelVersion == "" || r.DatasetVersion == "":
		return 0, errors.New("expected model and dataset versions must be set")
	case !r.From.Before(r.To):
		return 0, fmt.Errorf("period %s to %s is empty", r.From, r.To)
	case !core_domain.IsHourAligned(r.From) || !core_domain.IsHourAligned(r.To):
		return 0, fmt.Errorf("period %s to %s must start and end on whole Moscow hours", r.From, r.To)
	case span%time.Hour != 0 || span > MaxHours*time.Hour:
		return 0, fmt.Errorf("period of %s must be 1 to %d whole hours", span, MaxHours)
	case timestamppb.New(r.From).CheckValid() != nil || timestamppb.New(r.To).CheckValid() != nil:
		return 0, fmt.Errorf("period %s to %s is outside the protobuf Timestamp range", r.From, r.To)
	}
	return int(span / time.Hour), nil
}

// result checks the versions first, then the exact hourly grid of r.
func (r Request) result(response *forecastv1.PredictResponse, hours int) (Result, error) {
	for _, version := range []struct{ name, value string }{
		{"model_version", response.ModelVersion},
		{"dataset_version", response.DatasetVersion},
	} {
		if version.value == "" || strings.TrimSpace(version.value) != version.value {
			return Result{}, invalidResponse("%s %q must be non-blank without surrounding spaces", version.name, version.value)
		}
	}
	if response.ModelVersion != r.ModelVersion || response.DatasetVersion != r.DatasetVersion {
		return Result{}, &Error{Code: "ml_version_mismatch", Err: fmt.Errorf(
			"server artifacts are model %q and dataset %q, the forecast version expects %q and %q",
			response.ModelVersion, response.DatasetVersion, r.ModelVersion, r.DatasetVersion)}
	}
	if len(response.Points) != hours {
		return Result{}, invalidResponse("got %d points, want %d", len(response.Points), hours)
	}
	points := make([]Point, hours)
	for i, point := range response.Points {
		// One exact second per index rules out gaps, duplicates and disorder.
		// With zero nanos it also implies a valid Timestamp, as [From, To) is.
		want := r.From.Unix() + int64(i)*3600
		switch {
		case point == nil || point.HourStart == nil:
			return Result{}, invalidResponse("point %d has no hour_start", i)
		case point.HourStart.Seconds != want || point.HourStart.Nanos != 0:
			return Result{}, invalidResponse("point %d starts at %d s %d ns, want %d s",
				i, point.HourStart.Seconds, point.HourStart.Nanos, want)
		case point.Boardings == nil:
			return Result{}, invalidResponse("point %d has no boardings", i)
		case *point.Boardings < 0:
			return Result{}, invalidResponse("point %d has negative boardings %d", i, *point.Boardings)
		}
		points[i] = Point{HourStart: point.HourStart.AsTime().In(core_domain.Moscow), Boardings: *point.Boardings}
	}
	return Result{ModelVersion: response.ModelVersion, DatasetVersion: response.DatasetVersion, Points: points}, nil
}

func invalidResponse(format string, args ...any) *Error {
	return &Error{Code: "ml_invalid_response", Err: fmt.Errorf(format, args...)}
}

// statusError classifies a failed call by its gRPC code only; the message text
// is never parsed. Only UNAVAILABLE and DEADLINE_EXCEEDED are worth another
// attempt through the queue.
func statusError(err error) *Error {
	e := &Error{Code: "ml_internal", Err: err}
	switch status.Code(err) {
	case codes.Unavailable:
		e.Code, e.Retryable = "ml_unavailable", true
	case codes.DeadlineExceeded:
		e.Code, e.Retryable = "ml_deadline_exceeded", true
	case codes.Canceled:
		// The worker tells its own shutdown or lost lease from an unexplained
		// cancellation, which is terminal.
		e.Code = "ml_cancelled"
	case codes.InvalidArgument:
		e.Code = "ml_invalid_argument"
	case codes.NotFound:
		e.Code = "ml_not_found"
	case codes.FailedPrecondition:
		e.Code = "ml_failed_precondition"
	case codes.ResourceExhausted:
		e.Code = "ml_resource_exhausted"
	}
	return e
}
