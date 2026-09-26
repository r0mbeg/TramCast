package core_http_server

import (
	"net/http"

	"github.com/gin-gonic/gin"
)

// ErrorResponse is the JSON body of every error response. Error holds a stable
// machine-readable code; internal details stay in the logs.
type ErrorResponse struct {
	Error string `json:"error"`
}

// WriteError aborts the request with a JSON error code.
func WriteError(c *gin.Context, status int, code string) {
	c.AbortWithStatusJSON(status, ErrorResponse{Error: code})
}

// WriteInternalError attaches err for the access log and responds with a
// generic 500 that does not expose the error text.
func WriteInternalError(c *gin.Context, err error) {
	_ = c.Error(err)
	WriteError(c, http.StatusInternalServerError, "internal_server_error")
}
