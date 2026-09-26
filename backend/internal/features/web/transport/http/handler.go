// Package web_transport_http serves the frontend build over HTTP.
package web_transport_http

import (
	"context"
	"errors"
	"net/http"
	"path"
	"strings"

	"github.com/gin-gonic/gin"

	core_http_server "github.com/r0mbeg/TramCast/backend/internal/core/transport/http"
	web_service "github.com/r0mbeg/TramCast/backend/internal/features/web/service"
)

type Service interface {
	GetFile(ctx context.Context, requestPath string) (web_service.File, error)
	GetPage(ctx context.Context, requestPath string) (web_service.File, error)
}

type Handler struct {
	service Service
}

func NewHandler(service Service) *Handler {
	return &Handler{service: service}
}

// Serve is the router's NoRoute handler. API paths and non-GET requests get a
// JSON 404. Browser navigation may fall back to index.html; API-like clients
// only get existing files.
func (h *Handler) Serve(c *gin.Context) {
	r := c.Request
	if (r.Method != http.MethodGet && r.Method != http.MethodHead) || isAPIPath(r.URL.Path) {
		core_http_server.WriteError(c, http.StatusNotFound, "not_found")
		return
	}

	get := h.service.GetPage
	if wantsJSON(r) {
		get = h.service.GetFile
	}
	file, err := get(r.Context(), r.URL.Path)
	if errors.Is(err, web_service.ErrNotFound) {
		core_http_server.WriteError(c, http.StatusNotFound, "not_found")
		return
	}
	if err != nil {
		core_http_server.WriteInternalError(c, err)
		return
	}
	defer file.Content.Close()

	header := c.Writer.Header()
	if file.ContentType != "" {
		header.Set("Content-Type", file.ContentType)
	}
	header.Set("Cache-Control", file.CacheControl)
	if file.ETag != "" {
		header.Set("ETag", file.ETag)
	}
	// ServeContent handles If-None-Match, Last-Modified, 304, HEAD and Range.
	http.ServeContent(c.Writer, r, file.Name, file.ModTime, file.Content)
}

func isAPIPath(requestPath string) bool {
	cleanPath := path.Clean("/" + requestPath)
	return cleanPath == "/api" || strings.HasPrefix(cleanPath, "/api/")
}

// wantsJSON reports an API-like client rather than browser navigation.
func wantsJSON(r *http.Request) bool {
	accept := r.Header.Get("Accept")
	return strings.Contains(accept, "application/json") && !strings.Contains(accept, "text/html")
}
