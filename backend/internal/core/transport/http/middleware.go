package core_http_server

import (
	"crypto/rand"
	"log/slog"
	"net/http"
	"runtime/debug"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
)

const requestIDKey = "request_id"

func requestID() gin.HandlerFunc {
	return func(c *gin.Context) {
		id := rand.Text()
		c.Set(requestIDKey, id)
		c.Header("X-Request-ID", id)
		c.Next()
	}
}

func accessLog(log *slog.Logger) gin.HandlerFunc {
	return func(c *gin.Context) {
		started := time.Now()
		c.Next()
		status := c.Writer.Status()
		attrs := []any{
			"request_id", c.GetString(requestIDKey),
			"method", c.Request.Method,
			"path", c.Request.URL.Path,
			"status", status,
			"duration", time.Since(started),
		}
		// Handlers attach internal errors with c.Error; clients only see a code.
		if len(c.Errors) > 0 {
			attrs = append(attrs, "error", strings.Join(c.Errors.Errors(), "; "))
		}
		level := slog.LevelInfo
		if status >= http.StatusInternalServerError {
			level = slog.LevelError
		}
		log.Log(c.Request.Context(), level, "HTTP request", attrs...)
	}
}

func recovery(log *slog.Logger) gin.HandlerFunc {
	return func(c *gin.Context) {
		defer func() {
			if recover() != nil {
				log.ErrorContext(c.Request.Context(), "HTTP handler panic",
					"request_id", c.GetString(requestIDKey),
					"stack", string(debug.Stack()),
				)
				c.Abort()
				// Headers already sent cannot be replaced with a different status.
				if !c.Writer.Written() {
					WriteError(c, http.StatusInternalServerError, "internal_server_error")
				}
			}
		}()
		c.Next()
	}
}
