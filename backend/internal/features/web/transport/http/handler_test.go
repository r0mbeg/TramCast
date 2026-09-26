package web_transport_http

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"

	core_config "github.com/r0mbeg/TramCast/backend/internal/core/config"
	core_http_server "github.com/r0mbeg/TramCast/backend/internal/core/transport/http"
	web_service "github.com/r0mbeg/TramCast/backend/internal/features/web/service"
)

const indexBody = "<!doctype html><title>TramCast</title>"

// newTestServer builds root/dist with a small frontend build and a secret file
// outside it, then serves dist through the real router fallback.
func newTestServer(t *testing.T, withBuild bool) http.Handler {
	t.Helper()
	root := t.TempDir()
	dist := filepath.Join(root, "dist")
	writeFile(t, filepath.Join(root, "secret.txt"), "outside-secret")
	if withBuild {
		writeFile(t, filepath.Join(dist, web_service.IndexFile), indexBody)
		writeFile(t, filepath.Join(dist, "assets", "app-3f2a.js"), "console.log('app')")
		writeFile(t, filepath.Join(dist, "assets", "app-3f2a.css"), "body{color:#fff}")
		writeFile(t, filepath.Join(dist, "data", "lines.json"), `{"lines":[]}`)
		writeFile(t, filepath.Join(dist, "favicon.ico"), "\x00\x00\x01\x00")
	}
	server := core_http_server.New(core_config.HTTPConfig{}, slog.New(slog.NewTextHandler(io.Discard, nil)), func(context.Context) error { return nil })
	server.Router().GET("/api/known", func(c *gin.Context) { c.JSON(http.StatusOK, gin.H{"ok": true}) })
	server.Router().NoRoute(NewHandler(web_service.NewService(os.DirFS(dist))).Serve)
	return server.Router()
}

func writeFile(t *testing.T, name, contents string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(name), 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(name, []byte(contents), 0600); err != nil {
		t.Fatal(err)
	}
}

func do(handler http.Handler, method, target string, headers ...string) *httptest.ResponseRecorder {
	request := httptest.NewRequest(method, target, nil)
	for i := 0; i+1 < len(headers); i += 2 {
		request.Header.Set(headers[i], headers[i+1])
	}
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	return response
}

func TestServesFiles(t *testing.T) {
	handler := newTestServer(t, true)
	for _, tt := range []struct {
		target, body, contentType, cacheControl string
	}{
		{"/", indexBody, "text/html; charset=utf-8", "no-cache"},
		{"/assets/app-3f2a.js", "console.log('app')", "text/javascript; charset=utf-8", "public, max-age=31536000, immutable"},
		{"/assets/app-3f2a.css", "body{color:#fff}", "text/css; charset=utf-8", "public, max-age=31536000, immutable"},
		{"/favicon.ico", "\x00\x00\x01\x00", "image/x-icon", "no-cache"},
	} {
		t.Run(tt.target, func(t *testing.T) {
			response := do(handler, http.MethodGet, tt.target)
			if response.Code != http.StatusOK || response.Body.String() != tt.body {
				t.Fatalf("response = %d %q", response.Code, response.Body.String())
			}
			if got := response.Header().Get("Content-Type"); got != tt.contentType {
				t.Errorf("Content-Type = %q, want %q", got, tt.contentType)
			}
			if got := response.Header().Get("Cache-Control"); got != tt.cacheControl {
				t.Errorf("Cache-Control = %q, want %q", got, tt.cacheControl)
			}
		})
	}
}

func TestClientRoutesAndDirectoriesGetIndex(t *testing.T) {
	handler := newTestServer(t, true)
	for _, target := range []string{"/routes/12", "/routes/12?date=2025-11-01", "/assets", "/assets/", "/data/"} {
		t.Run(target, func(t *testing.T) {
			response := do(handler, http.MethodGet, target)
			if response.Code != http.StatusOK || response.Body.String() != indexBody {
				t.Fatalf("response = %d %q, want index.html", response.Code, response.Body.String())
			}
			if got := response.Header().Get("Cache-Control"); got != "no-cache" {
				t.Errorf("Cache-Control = %q, want no-cache", got)
			}
		})
	}
}

func TestNotFoundCases(t *testing.T) {
	handler := newTestServer(t, true)
	for _, tt := range []struct {
		name, method, target string
		headers              []string
	}{
		{"missing asset", http.MethodGet, "/assets/missing.js", nil},
		{"missing root file", http.MethodGet, "/robots.txt", nil},
		{"unknown API path", http.MethodGet, "/api/unknown", nil},
		{"API root", http.MethodGet, "/api", nil},
		{"API path with duplicate slash", http.MethodGet, "//api/unknown", nil},
		{"wrong method for API route", http.MethodPost, "/api/known", nil},
		{"POST to page path", http.MethodPost, "/routes/12", nil},
		{"JSON client on page path", http.MethodGet, "/routes/12", []string{"Accept", "application/json"}},
		{"path through a file", http.MethodGet, "/favicon.ico/extra.png", nil},
	} {
		t.Run(tt.name, func(t *testing.T) {
			response := do(handler, tt.method, tt.target, tt.headers...)
			if response.Code != http.StatusNotFound || response.Body.String() != `{"error":"not_found"}` {
				t.Fatalf("response = %d %q", response.Code, response.Body.String())
			}
		})
	}
}

func TestExistingFileIsServedToJSONClient(t *testing.T) {
	response := do(newTestServer(t, true), http.MethodGet, "/data/lines.json", "Accept", "application/json")
	if response.Code != http.StatusOK || response.Body.String() != `{"lines":[]}` || response.Header().Get("Content-Type") != "application/json" {
		t.Fatalf("response = %d %q %q", response.Code, response.Header().Get("Content-Type"), response.Body.String())
	}
}

func TestDoesNotEscapeBuildDirectory(t *testing.T) {
	handler := newTestServer(t, true)
	for _, target := range []string{"/../secret.txt", "/assets/../../secret.txt", "/%2e%2e/secret.txt"} {
		t.Run(target, func(t *testing.T) {
			response := do(handler, http.MethodGet, target)
			if response.Code == http.StatusOK || strings.Contains(response.Body.String(), "outside-secret") {
				t.Fatalf("response = %d %q", response.Code, response.Body.String())
			}
		})
	}
}

func TestHeadAndConditionalRequests(t *testing.T) {
	handler := newTestServer(t, true)

	head := do(handler, http.MethodHead, "/")
	if head.Code != http.StatusOK || head.Body.Len() != 0 || head.Header().Get("Content-Length") != "38" {
		t.Fatalf("HEAD = %d, body %d bytes, Content-Length %q", head.Code, head.Body.Len(), head.Header().Get("Content-Length"))
	}

	first := do(handler, http.MethodGet, "/assets/app-3f2a.js")
	lastModified := first.Header().Get("Last-Modified")
	if lastModified == "" {
		t.Fatal("Last-Modified is missing")
	}
	conditional := do(handler, http.MethodGet, "/assets/app-3f2a.js", "If-Modified-Since", lastModified)
	if conditional.Code != http.StatusNotModified || conditional.Body.Len() != 0 {
		t.Fatalf("conditional GET = %d, body %q", conditional.Code, conditional.Body.String())
	}
}

func TestRollbackToOlderIndexIsNotServedFromCache(t *testing.T) {
	dist := t.TempDir()
	index := filepath.Join(dist, web_service.IndexFile)
	writeFile(t, index, `<script src="/assets/app-new.js"></script>`)
	server := core_http_server.New(core_config.HTTPConfig{}, slog.New(slog.NewTextHandler(io.Discard, nil)), func(context.Context) error { return nil })
	server.Router().NoRoute(NewHandler(web_service.NewService(os.DirFS(dist))).Serve)
	handler := server.Router()

	first := do(handler, http.MethodGet, "/")
	etag, lastModified := first.Header().Get("ETag"), first.Header().Get("Last-Modified")
	if etag == "" || lastModified == "" {
		t.Fatalf("validators are missing: ETag %q, Last-Modified %q", etag, lastModified)
	}
	if cached := do(handler, http.MethodGet, "/", "If-None-Match", etag, "If-Modified-Since", lastModified); cached.Code != http.StatusNotModified {
		t.Fatalf("unchanged page = %d, want 304", cached.Code)
	}

	// Roll back to an older build: different content with an earlier file date.
	writeFile(t, index, `<script src="/assets/app-old.js"></script>`)
	older := time.Now().Add(-24 * time.Hour)
	if err := os.Chtimes(index, older, older); err != nil {
		t.Fatal(err)
	}
	response := do(handler, http.MethodGet, "/", "If-None-Match", etag, "If-Modified-Since", lastModified)
	if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), "app-old.js") {
		t.Fatalf("rolled back page = %d %q, want 200 with the older build", response.Code, response.Body.String())
	}
}

func TestIndexPathServesMainPage(t *testing.T) {
	response := do(newTestServer(t, true), http.MethodGet, "/index.html")
	if response.Code != http.StatusOK || response.Body.String() != indexBody || response.Header().Get("Cache-Control") != "no-cache" {
		t.Fatalf("response = %d %q, Cache-Control %q", response.Code, response.Body.String(), response.Header().Get("Cache-Control"))
	}
}

type stubService struct {
	err                  error
	fileCalls, pageCalls int
}

func (s *stubService) GetFile(context.Context, string) (web_service.File, error) {
	s.fileCalls++
	return web_service.File{}, s.err
}

func (s *stubService) GetPage(context.Context, string) (web_service.File, error) {
	s.pageCalls++
	return web_service.File{}, s.err
}

func serveStub(t *testing.T, service *stubService, headers ...string) (*httptest.ResponseRecorder, string) {
	t.Helper()
	var logs strings.Builder
	server := core_http_server.New(core_config.HTTPConfig{}, slog.New(slog.NewJSONHandler(&logs, nil)), func(context.Context) error { return nil })
	server.Router().NoRoute(NewHandler(service).Serve)
	return do(server.Router(), http.MethodGet, "/routes/12", headers...), logs.String()
}

func TestServiceErrorIsNotExposed(t *testing.T) {
	response, logs := serveStub(t, &stubService{err: errors.New("disk-detail-secret")})
	if response.Code != http.StatusInternalServerError || response.Body.String() != `{"error":"internal_server_error"}` {
		t.Fatalf("response = %d %q", response.Code, response.Body.String())
	}
	if !strings.Contains(logs, "disk-detail-secret") {
		t.Fatalf("internal error must be logged: %s", logs)
	}
}

func TestClientKindSelectsServiceMethod(t *testing.T) {
	browser := &stubService{err: web_service.ErrNotFound}
	serveStub(t, browser, "Accept", "text/html,application/xhtml+xml")
	if browser.pageCalls != 1 || browser.fileCalls != 0 {
		t.Errorf("browser navigation: GetPage %d, GetFile %d calls", browser.pageCalls, browser.fileCalls)
	}
	jsonClient := &stubService{err: web_service.ErrNotFound}
	serveStub(t, jsonClient, "Accept", "application/json")
	if jsonClient.fileCalls != 1 || jsonClient.pageCalls != 0 {
		t.Errorf("JSON client: GetPage %d, GetFile %d calls", jsonClient.pageCalls, jsonClient.fileCalls)
	}
}

func TestMissingBuild(t *testing.T) {
	handler := newTestServer(t, false)
	for _, target := range []string{"/", "/routes/12", "/assets/app-3f2a.js"} {
		t.Run(target, func(t *testing.T) {
			response := do(handler, http.MethodGet, target)
			if response.Code != http.StatusNotFound || response.Body.String() != `{"error":"not_found"}` {
				t.Fatalf("response = %d %q", response.Code, response.Body.String())
			}
		})
	}
}
