// Package routes_transport_http exposes the route catalog over HTTP.
package routes_transport_http

import (
	"context"
	"errors"
	"net/http"
	"strconv"

	"github.com/gin-gonic/gin"

	core_http_server "github.com/r0mbeg/TramCast/backend/internal/core/transport/http"
	routes_sqlc "github.com/r0mbeg/TramCast/backend/internal/features/routes/repository/postgres/sqlc"
	routes_service "github.com/r0mbeg/TramCast/backend/internal/features/routes/service"
)

type Service interface {
	ListRoutes(ctx context.Context) ([]routes_sqlc.Route, error)
	ListRouteStops(ctx context.Context, routeID int64) ([]routes_service.Pattern, error)
	ListRouteGeometry(ctx context.Context) ([]routes_service.Line, error)
}

type Handler struct {
	service Service
}

func NewHandler(service Service) *Handler {
	return &Handler{service: service}
}

// Register adds the handlers to the API group.
func (h *Handler) Register(router gin.IRouter) {
	router.GET("/routes", h.listRoutes)
	router.GET("/routes/geometry", h.routeGeometry)
	router.GET("/routes/:route_id/stops", h.listRouteStops)
}

// routeGeometry returns a GeoJSON FeatureCollection with one LineString per
// movement variant; GeoJSON is formed by Go as the contract requires.
func (h *Handler) routeGeometry(c *gin.Context) {
	lines, err := h.service.ListRouteGeometry(c.Request.Context())
	if err != nil {
		core_http_server.WriteInternalError(c, err)
		return
	}
	c.JSON(http.StatusOK, newGeometryResponse(lines))
}

func (h *Handler) listRoutes(c *gin.Context) {
	catalog, err := h.service.ListRoutes(c.Request.Context())
	if err != nil {
		core_http_server.WriteInternalError(c, err)
		return
	}
	c.JSON(http.StatusOK, newListRoutesResponse(catalog))
}

func (h *Handler) listRouteStops(c *gin.Context) {
	routeID, err := strconv.ParseInt(c.Param("route_id"), 10, 64)
	if err != nil || routeID <= 0 {
		core_http_server.WriteError(c, http.StatusBadRequest, "invalid_route_id")
		return
	}
	patterns, err := h.service.ListRouteStops(c.Request.Context(), routeID)
	if errors.Is(err, routes_service.ErrRouteNotFound) {
		core_http_server.WriteError(c, http.StatusNotFound, "route_not_found")
		return
	}
	if err != nil {
		core_http_server.WriteInternalError(c, err)
		return
	}
	c.JSON(http.StatusOK, newListRouteStopsResponse(routeID, patterns))
}
