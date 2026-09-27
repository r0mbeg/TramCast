package forecasts_predictor_grpc

import (
	"context"
	"errors"
	"math"
	"net"
	"strings"
	"sync"
	"testing"
	"testing/synctest"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/test/bufconn"
	"google.golang.org/protobuf/types/known/timestamppb"

	core_domain "github.com/r0mbeg/TramCast/backend/internal/core/domain"
	forecastv1 "github.com/r0mbeg/TramCast/backend/internal/gen/tramcast/forecast/v1"
)

const (
	testModel   = "tabpfn030-test"
	testDataset = "prepared-test"
)

var horizonStart = time.Date(2025, 11, 1, 0, 0, 0, 0, core_domain.Moscow)

// fakeServer records calls and answers with predict, or with a valid grid
// of zeros when predict is nil.
type fakeServer struct {
	forecastv1.UnimplementedForecastServiceServer
	predict func(ctx context.Context, request *forecastv1.PredictRequest) (*forecastv1.PredictResponse, error)

	mu          sync.Mutex
	calls       int
	request     *forecastv1.PredictRequest
	hasDeadline bool
	deadline    time.Time
}

func (s *fakeServer) Predict(ctx context.Context, request *forecastv1.PredictRequest) (*forecastv1.PredictResponse, error) {
	s.mu.Lock()
	s.calls++
	s.request = request
	s.deadline, s.hasDeadline = ctx.Deadline()
	s.mu.Unlock()
	if s.predict == nil {
		return grid(request, func(int) int64 { return 0 }), nil
	}
	return s.predict(ctx, request)
}

func (s *fakeServer) callCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.calls
}

// last returns the last request and its deadline as the server saw them.
func (s *fakeServer) last() (request *forecastv1.PredictRequest, deadline time.Time, hasDeadline bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.request, s.deadline, s.hasDeadline
}

// grid is a valid response for request: every hour, in order.
func grid(request *forecastv1.PredictRequest, boardings func(i int) int64) *forecastv1.PredictResponse {
	response := &forecastv1.PredictResponse{ModelVersion: testModel, DatasetVersion: testDataset}
	from, to := request.ForecastFrom.Seconds, request.ForecastTo.Seconds
	for i, second := 0, from; second < to; i, second = i+1, second+3600 {
		value := boardings(i)
		response.Points = append(response.Points, &forecastv1.PredictionPoint{
			HourStart: &timestamppb.Timestamp{Seconds: second},
			Boardings: &value,
		})
	}
	return response
}

func newTestClient(t *testing.T, server *fakeServer, timeout time.Duration) *Client {
	t.Helper()
	listener := bufconn.Listen(1 << 20)
	grpcServer := grpc.NewServer()
	forecastv1.RegisterForecastServiceServer(grpcServer, server)
	go func() { _ = grpcServer.Serve(listener) }()
	t.Cleanup(grpcServer.Stop)

	client, err := New("passthrough:///bufnet", timeout, grpc.WithContextDialer(
		func(ctx context.Context, _ string) (net.Conn, error) { return listener.DialContext(ctx) }))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := client.Close(); err != nil {
			t.Error(err)
		}
	})
	return client
}

func testRequest(route int16, hours int) Request {
	return Request{
		RouteNumber: route, From: horizonStart, To: horizonStart.Add(time.Duration(hours) * time.Hour),
		ModelVersion: testModel, DatasetVersion: testDataset,
	}
}

// fullHorizon is November and December 2025, the MVP forecast version.
func fullHorizon(route int16) Request {
	r := testRequest(route, 0)
	r.To = time.Date(2026, 1, 1, 0, 0, 0, 0, core_domain.Moscow)
	return r
}

func requireError(t *testing.T, err error, code string, retryable bool) {
	t.Helper()
	var predictErr *Error
	if !errors.As(err, &predictErr) {
		t.Fatalf("error = %v, want *Error %s", err, code)
	}
	if predictErr.Code != code || predictErr.Retryable != retryable {
		t.Fatalf("error = %s retryable=%v, want %s retryable=%v", predictErr, predictErr.Retryable, code, retryable)
	}
	if !strings.HasPrefix(predictErr.Error(), code+": ") || predictErr.Unwrap() == nil {
		t.Fatalf("error %q must start with its code and keep the cause", predictErr)
	}
}

func TestPredictFullHorizon(t *testing.T) {
	server := &fakeServer{predict: func(_ context.Context, request *forecastv1.PredictRequest) (*forecastv1.PredictResponse, error) {
		return grid(request, func(i int) int64 {
			switch i {
			case 1:
				return math.MaxInt64
			case 2:
				return 17
			}
			return 0
		}), nil
	}}
	result, err := newTestClient(t, server, time.Minute).Predict(context.Background(), fullHorizon(1))
	if err != nil {
		t.Fatal(err)
	}

	sent, deadline, hasDeadline := server.last()
	wantFrom := time.Date(2025, 10, 31, 21, 0, 0, 0, time.UTC)
	wantTo := time.Date(2025, 12, 31, 21, 0, 0, 0, time.UTC)
	if sent.RouteNumber != 1 || !sent.ForecastFrom.AsTime().Equal(wantFrom) || !sent.ForecastTo.AsTime().Equal(wantTo) ||
		sent.ForecastFrom.Nanos != 0 || sent.ForecastTo.Nanos != 0 {
		t.Fatalf("request = %v, want route 1 from %s to %s", sent, wantFrom, wantTo)
	}
	if !hasDeadline || time.Until(deadline) > time.Minute {
		t.Fatalf("server deadline = %v (set %v), want at most the client timeout", deadline, hasDeadline)
	}

	if result.ModelVersion != testModel || result.DatasetVersion != testDataset || len(result.Points) != 1464 {
		t.Fatalf("result = %s/%s with %d points", result.ModelVersion, result.DatasetVersion, len(result.Points))
	}
	for i, want := range []Point{
		{HourStart: horizonStart, Boardings: 0},
		{HourStart: horizonStart.Add(time.Hour), Boardings: math.MaxInt64},
		{HourStart: horizonStart.Add(2 * time.Hour), Boardings: 17},
	} {
		if got := result.Points[i]; !got.HourStart.Equal(want.HourStart) || got.Boardings != want.Boardings {
			t.Errorf("point %d = %+v, want %+v", i, got, want)
		}
	}
	last := result.Points[len(result.Points)-1]
	if want := time.Date(2025, 12, 31, 23, 0, 0, 0, core_domain.Moscow); !last.HourStart.Equal(want) || last.HourStart.Hour() != 23 {
		t.Errorf("last point starts at %s, want %s", last.HourStart, want)
	}
	for i, point := range result.Points {
		if point.HourStart.Location() != core_domain.Moscow {
			t.Fatalf("point %d is in %s, want %s", i, point.HourStart.Location(), core_domain.Moscow)
		}
	}
}

func TestPredictOneHour(t *testing.T) {
	server := &fakeServer{}
	result, err := newTestClient(t, server, time.Minute).Predict(context.Background(), testRequest(5, 1))
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Points) != 1 || !result.Points[0].HourStart.Equal(horizonStart) || result.Points[0].Boardings != 0 {
		t.Fatalf("points = %+v, want one zero hour", result.Points)
	}
}

func TestPredictRejectsInvalidRequests(t *testing.T) {
	india := time.FixedZone("UTC+05:30", 5*60*60+30*60)
	for _, tt := range []struct {
		name   string
		change func(*Request)
	}{
		{"zero route", func(r *Request) { r.RouteNumber = 0 }},
		{"negative route", func(r *Request) { r.RouteNumber = -1 }},
		{"no expected model", func(r *Request) { r.ModelVersion = "" }},
		{"no expected dataset", func(r *Request) { r.DatasetVersion = "" }},
		{"zero times", func(r *Request) { r.From, r.To = time.Time{}, time.Time{} }},
		{"empty period", func(r *Request) { r.To = r.From }},
		{"reversed period", func(r *Request) { r.From, r.To = r.To, r.From }},
		{"start half past", func(r *Request) { r.From = r.From.Add(30 * time.Minute) }},
		{"end with nanoseconds", func(r *Request) { r.To = r.To.Add(time.Nanosecond) }},
		{"whole hour only in another zone", func(r *Request) { r.From = time.Date(2025, 11, 1, 2, 0, 0, 0, india) }},
		{"longer than 1464 hours", func(r *Request) { r.To = r.From.Add(1465 * time.Hour) }},
		{"beyond the Timestamp range", func(r *Request) {
			r.From = time.Date(10000, 1, 2, 0, 0, 0, 0, core_domain.Moscow)
			r.To = r.From.Add(time.Hour)
		}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			server := &fakeServer{}
			r := testRequest(1, 24)
			tt.change(&r)
			_, err := newTestClient(t, server, time.Minute).Predict(context.Background(), r)
			requireError(t, err, "ml_invalid_request", false)
			if calls := server.callCount(); calls != 0 {
				t.Fatalf("server calls = %d, want none", calls)
			}
		})
	}
}

func TestPredictRejectsInvalidResponses(t *testing.T) {
	start := horizonStart.Unix()
	negative := int64(-1)
	for _, tt := range []struct {
		name   string
		change func(*forecastv1.PredictResponse)
	}{
		{"blank model version", func(r *forecastv1.PredictResponse) { r.ModelVersion = "" }},
		{"space model version", func(r *forecastv1.PredictResponse) { r.ModelVersion = " " }},
		{"padded model version", func(r *forecastv1.PredictResponse) { r.ModelVersion = testModel + "\n" }},
		{"blank dataset version", func(r *forecastv1.PredictResponse) { r.DatasetVersion = "" }},
		{"padded dataset version", func(r *forecastv1.PredictResponse) { r.DatasetVersion = " " + testDataset }},
		{"no points", func(r *forecastv1.PredictResponse) { r.Points = nil }},
		{"missing last hour", func(r *forecastv1.PredictResponse) { r.Points = r.Points[:2] }},
		{"extra hour", func(r *forecastv1.PredictResponse) {
			r.Points = append(r.Points, &forecastv1.PredictionPoint{
				HourStart: &timestamppb.Timestamp{Seconds: start + 3*3600}, Boardings: r.Points[0].Boardings,
			})
		}},
		{"nil point", func(r *forecastv1.PredictResponse) { r.Points[1] = nil }},
		{"no hour_start", func(r *forecastv1.PredictResponse) { r.Points[1].HourStart = nil }},
		{"duplicate hour", func(r *forecastv1.PredictResponse) { r.Points[1].HourStart = r.Points[0].HourStart }},
		{"gap", func(r *forecastv1.PredictResponse) { r.Points[2].HourStart.Seconds += 3600 }},
		{"descending order", func(r *forecastv1.PredictResponse) { r.Points[0], r.Points[2] = r.Points[2], r.Points[0] }},
		{"half hour", func(r *forecastv1.PredictResponse) { r.Points[1].HourStart.Seconds += 1800 }},
		{"nanoseconds", func(r *forecastv1.PredictResponse) { r.Points[1].HourStart.Nanos = 1 }},
		{"out-of-range nanoseconds", func(r *forecastv1.PredictResponse) { r.Points[1].HourStart.Nanos = 1_000_000_000 }},
		{"no boardings", func(r *forecastv1.PredictResponse) { r.Points[2].Boardings = nil }},
		{"negative boardings", func(r *forecastv1.PredictResponse) { r.Points[2].Boardings = &negative }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			server := &fakeServer{predict: func(_ context.Context, request *forecastv1.PredictRequest) (*forecastv1.PredictResponse, error) {
				response := grid(request, func(int) int64 { return 5 })
				tt.change(response)
				return response, nil
			}}
			_, err := newTestClient(t, server, time.Minute).Predict(context.Background(), testRequest(1, 3))
			requireError(t, err, "ml_invalid_response", false)
		})
	}
}

func TestPredictRejectsVersionMismatch(t *testing.T) {
	for _, tt := range []struct {
		name   string
		change func(*forecastv1.PredictResponse)
	}{
		{"other model", func(r *forecastv1.PredictResponse) { r.ModelVersion = "tabpfn030-zhores8477154-replay" }},
		{"model in another case", func(r *forecastv1.PredictResponse) { r.ModelVersion = strings.ToUpper(testModel) }},
		{"other dataset", func(r *forecastv1.PredictResponse) { r.DatasetVersion = "prepared-other" }},
		// Versions are checked first: points of other artifacts do not matter.
		{"other model without points", func(r *forecastv1.PredictResponse) {
			r.ModelVersion = "chronos002"
			r.Points = nil
		}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			server := &fakeServer{predict: func(_ context.Context, request *forecastv1.PredictRequest) (*forecastv1.PredictResponse, error) {
				response := grid(request, func(int) int64 { return 1 })
				tt.change(response)
				return response, nil
			}}
			_, err := newTestClient(t, server, time.Minute).Predict(context.Background(), testRequest(1, 3))
			requireError(t, err, "ml_version_mismatch", false)
		})
	}
}

func TestPredictMapsStatusCodes(t *testing.T) {
	for _, tt := range []struct {
		code      codes.Code
		want      string
		retryable bool
	}{
		{codes.Unavailable, "ml_unavailable", true},
		{codes.DeadlineExceeded, "ml_deadline_exceeded", true},
		{codes.Canceled, "ml_cancelled", false},
		{codes.InvalidArgument, "ml_invalid_argument", false},
		{codes.NotFound, "ml_not_found", false},
		{codes.FailedPrecondition, "ml_failed_precondition", false},
		{codes.ResourceExhausted, "ml_resource_exhausted", false},
		{codes.Internal, "ml_internal", false},
		{codes.Unknown, "ml_internal", false},
		{codes.Unimplemented, "ml_internal", false},
		{codes.PermissionDenied, "ml_internal", false},
		{codes.Aborted, "ml_internal", false},
		{codes.DataLoss, "ml_internal", false},
	} {
		t.Run(tt.code.String(), func(t *testing.T) {
			server := &fakeServer{predict: func(context.Context, *forecastv1.PredictRequest) (*forecastv1.PredictResponse, error) {
				// The message must not influence the classification.
				return nil, status.Error(tt.code, "Forecast worker is busy, UNAVAILABLE, retry later")
			}}
			_, err := newTestClient(t, server, time.Minute).Predict(context.Background(), testRequest(1, 3))
			requireError(t, err, tt.want, tt.retryable)
			if got := status.Code(err); got != tt.code {
				t.Fatalf("wrapped status = %s, want %s", got, tt.code)
			}
		})
	}
}

func TestPredictUnreachableServerIsRetryable(t *testing.T) {
	client, err := New("passthrough:///down", time.Minute, grpc.WithContextDialer(
		func(context.Context, string) (net.Conn, error) { return nil, errors.New("connection refused") }))
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()
	_, err = client.Predict(context.Background(), testRequest(1, 3))
	requireError(t, err, "ml_unavailable", true)
}

// blockingServer waits for the call to end and reports that it saw the end.
func blockingServer(started, done chan struct{}) *fakeServer {
	return &fakeServer{predict: func(ctx context.Context, _ *forecastv1.PredictRequest) (*forecastv1.PredictResponse, error) {
		close(started)
		<-ctx.Done()
		close(done)
		return nil, ctx.Err()
	}}
}

func waitClosed(t *testing.T, done chan struct{}, what string) {
	t.Helper()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatalf("server did not observe %s", what)
	}
}

func TestPredictClientDeadline(t *testing.T) {
	started, done := make(chan struct{}), make(chan struct{})
	client := newTestClient(t, blockingServer(started, done), 200*time.Millisecond)
	_, err := client.Predict(context.Background(), testRequest(1, 3))
	requireError(t, err, "ml_deadline_exceeded", true)
	waitClosed(t, done, "the client deadline")
}

func TestPredictCallerCancellation(t *testing.T) {
	started, done := make(chan struct{}), make(chan struct{})
	client := newTestClient(t, blockingServer(started, done), time.Minute)
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		<-started
		cancel()
	}()
	_, err := client.Predict(ctx, testRequest(1, 3))
	requireError(t, err, "ml_cancelled", false)
	waitClosed(t, done, "the cancellation")
}

func TestNewDoesNotConnect(t *testing.T) {
	dialed := make(chan struct{}, 1)
	client, err := New("passthrough:///idle", time.Minute, grpc.WithContextDialer(
		func(context.Context, string) (net.Conn, error) {
			dialed <- struct{}{}
			return nil, errors.New("connection refused")
		}))
	if err != nil {
		t.Fatal(err)
	}
	if state := client.conn.GetState(); state != connectivity.Idle {
		t.Errorf("state = %s, want Idle until the first call", state)
	}
	if err := client.Close(); err != nil {
		t.Fatal(err)
	}
	select {
	case <-dialed:
		t.Fatal("New must not dial")
	default:
	}
}

// TestReconnectDelayIsCapped keeps the server down and records every dial in
// fake time. The default gRPC backoff grows to 120 s (about 43 s by the ninth
// attempt); the client must retry at least every reconnectMaxDelay plus 20%
// jitter, so an ML restart is noticed within about 10 s.
func TestReconnectDelayIsCapped(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		var mu sync.Mutex
		var dials []time.Time
		client, err := New("passthrough:///down", time.Minute, grpc.WithContextDialer(
			func(context.Context, string) (net.Conn, error) {
				mu.Lock()
				dials = append(dials, time.Now())
				mu.Unlock()
				return nil, errors.New("connection refused")
			}))
		if err != nil {
			t.Fatal(err)
		}
		_, err = client.Predict(context.Background(), testRequest(1, 3))
		requireError(t, err, "ml_unavailable", true)
		time.Sleep(3 * time.Minute)
		if err := client.Close(); err != nil {
			t.Fatal(err)
		}
		synctest.Wait()

		mu.Lock()
		defer mu.Unlock()
		if len(dials) < 15 {
			t.Fatalf("dials = %d in 3 minutes, want at least 15", len(dials))
		}
		for i := 1; i < len(dials); i++ {
			if gap := dials[i].Sub(dials[i-1]); gap > reconnectMaxDelay*6/5 {
				t.Fatalf("dial %d waited %s after the previous one, want at most %s", i, gap, reconnectMaxDelay*6/5)
			}
		}
	})
}

func TestNewRejectsNonPositiveTimeout(t *testing.T) {
	for _, timeout := range []time.Duration{0, -time.Second} {
		if _, err := New("127.0.0.1:50051", timeout); err == nil {
			t.Errorf("New with timeout %s must fail", timeout)
		}
	}
}
