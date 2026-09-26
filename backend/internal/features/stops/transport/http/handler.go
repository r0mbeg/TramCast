// Package stops_transport_http exposes the stop catalog over HTTP.
package stops_transport_http

import (
	"context"
	"net/http"

	"github.com/gin-gonic/gin"

	core_http_server "github.com/r0mbeg/TramCast/backend/internal/core/transport/http"
	stops_sqlc "github.com/r0mbeg/TramCast/backend/internal/features/stops/repository/postgres/sqlc"
)

type Service interface {
	ListStops(ctx context.Context) ([]stops_sqlc.Stop, error)
}

type Handler struct {
	service Service
}

func NewHandler(service Service) *Handler {
	return &Handler{service: service}
}

// Register adds the handlers to the API group.
func (h *Handler) Register(router gin.IRouter) {
	router.GET("/stops", h.listStops)
}

type listStopsResponse struct {
	Stops []stopDTO `json:"stops"`
}

type stopDTO struct {
	ID        int64   `json:"id"`
	Name      string  `json:"name"`
	Latitude  float64 `json:"latitude"`
	Longitude float64 `json:"longitude"`
}

func (h *Handler) listStops(c *gin.Context) {
	stops, err := h.service.ListStops(c.Request.Context())
	if err != nil {
		core_http_server.WriteInternalError(c, err)
		return
	}
	response := listStopsResponse{Stops: make([]stopDTO, len(stops))}
	for i, stop := range stops {
		response.Stops[i] = stopDTO{ID: stop.ID, Name: stop.Name, Latitude: stop.Latitude, Longitude: stop.Longitude}
	}
	c.JSON(http.StatusOK, response)
}
