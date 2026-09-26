package core_http_server

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"net/http"

	"github.com/gin-gonic/gin"
	core_config "github.com/r0mbeg/TramCast/backend/internal/core/config"
)

type Server struct {
	cfg    core_config.HTTPConfig
	log    *slog.Logger
	router *gin.Engine
}

func New(cfg core_config.HTTPConfig, log *slog.Logger, ping func(context.Context) error) *Server {
	gin.SetMode(gin.ReleaseMode)
	router := gin.New()
	_ = router.SetTrustedProxies(nil)
	router.Use(requestID(), accessLog(log), recovery(log))

	// Gin does not route HEAD to GET handlers; without HEAD here, a probe would
	// fall through to NoRoute and could get the frontend page instead.
	probeMethods := []string{http.MethodGet, http.MethodHead}
	router.Match(probeMethods, "/healthz", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"status": "ok"})
	})
	router.Match(probeMethods, "/readyz", func(c *gin.Context) {
		ctx, cancel := context.WithTimeout(c.Request.Context(), cfg.ProbeTimeout)
		defer cancel()
		if err := ping(ctx); err != nil {
			c.JSON(http.StatusServiceUnavailable, gin.H{"status": "not_ready"})
			return
		}
		c.JSON(http.StatusOK, gin.H{"status": "ready"})
	})
	// Features may replace this fallback, as the web feature does for the frontend.
	router.NoRoute(func(c *gin.Context) {
		WriteError(c, http.StatusNotFound, "not_found")
	})

	return &Server{cfg: cfg, log: log, router: router}
}

// Router allows features to register routes before Run starts serving requests.
func (s *Server) Router() *gin.Engine {
	return s.router
}

func (s *Server) Run(ctx context.Context) error {
	if ctx.Err() != nil {
		return nil
	}
	listener, err := net.Listen("tcp", s.cfg.Addr)
	if err != nil {
		return fmt.Errorf("listen HTTP: %w", err)
	}
	return s.serve(ctx, listener)
}

func (s *Server) serve(ctx context.Context, listener net.Listener) error {
	// A shutdown signal stops accepting connections without cancelling active work.
	requestCtx, cancelRequests := context.WithCancel(context.Background())
	defer cancelRequests()
	server := &http.Server{
		Handler:           s.router,
		ErrorLog:          slog.NewLogLogger(s.log.Handler(), slog.LevelError),
		ReadHeaderTimeout: s.cfg.ReadHeaderTimeout,
		ReadTimeout:       s.cfg.ReadTimeout,
		WriteTimeout:      s.cfg.WriteTimeout,
		IdleTimeout:       s.cfg.IdleTimeout,
		BaseContext:       func(net.Listener) context.Context { return requestCtx },
	}
	defer server.Close()

	serveErr := make(chan error, 1)
	go func() { serveErr <- server.Serve(listener) }()
	s.log.InfoContext(ctx, "HTTP server listening", "addr", listener.Addr().String())

	select {
	case err := <-serveErr:
		return servingError(err)
	case <-ctx.Done():
		shutdownCtx, cancel := context.WithTimeout(context.Background(), s.cfg.ShutdownTimeout)
		defer cancel()
		if err := server.Shutdown(shutdownCtx); err != nil {
			cancelRequests()
			closeErr := server.Close()
			return errors.Join(fmt.Errorf("shutdown HTTP: %w", err), closeErr, servingError(<-serveErr))
		}
		return servingError(<-serveErr)
	}
}

func servingError(err error) error {
	if err == nil || errors.Is(err, http.ErrServerClosed) {
		return nil
	}
	return fmt.Errorf("serve HTTP: %w", err)
}
