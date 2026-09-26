// Package web_service contains the rules for serving the frontend build.
package web_service

import (
	"bytes"
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"path"
	"strings"
	"time"
)

// IndexFile is the frontend entry page, also returned for client-side routes.
const IndexFile = "index.html"

// ErrNotFound means that the build has no such regular file, or no build at all.
var ErrNotFound = errors.New("web file not found")

// contentTypes pins common web types: on Windows the mime package reads the
// registry, which may map these extensions to other values.
var contentTypes = map[string]string{
	".html":        "text/html; charset=utf-8",
	".css":         "text/css; charset=utf-8",
	".js":          "text/javascript; charset=utf-8",
	".mjs":         "text/javascript; charset=utf-8",
	".json":        "application/json",
	".map":         "application/json",
	".webmanifest": "application/manifest+json",
	".svg":         "image/svg+xml",
	".png":         "image/png",
	".jpg":         "image/jpeg",
	".jpeg":        "image/jpeg",
	".gif":         "image/gif",
	".webp":        "image/webp",
	".ico":         "image/x-icon",
	".woff":        "font/woff",
	".woff2":       "font/woff2",
	".txt":         "text/plain; charset=utf-8",
}

// File is an opened build file. The caller must close Content.
type File struct {
	Name    string
	ModTime time.Time
	Content io.ReadSeekCloser
	// ContentType is empty for unknown extensions; the transport then detects it.
	ContentType  string
	CacheControl string
	// ETag is a strong validator of the content; empty for immutable assets.
	ETag string
}

type Service struct {
	files fs.FS
}

// NewService serves files from the frontend build, for example os.DirFS(dir).
func NewService(files fs.FS) *Service {
	return &Service{files: files}
}

// GetMainPage returns index.html.
func (s *Service) GetMainPage(_ context.Context) (File, error) {
	return s.open(IndexFile)
}

// GetFile returns a regular file of the build. Missing files and directories
// return ErrNotFound; the cleaned path cannot leave the build directory.
func (s *Service) GetFile(_ context.Context, requestPath string) (File, error) {
	name := strings.TrimPrefix(path.Clean("/"+requestPath), "/")
	if name == "" {
		return File{}, fmt.Errorf("build root is a directory: %w", ErrNotFound)
	}
	return s.open(name)
}

// GetPage is GetFile with a fallback to the main page for paths without an
// extension: "/", directories and client-side routes. A missing file with an
// extension, such as an old asset, stays ErrNotFound.
func (s *Service) GetPage(ctx context.Context, requestPath string) (File, error) {
	file, err := s.GetFile(ctx, requestPath)
	if errors.Is(err, ErrNotFound) && path.Ext(path.Clean("/"+requestPath)) == "" {
		return s.GetMainPage(ctx)
	}
	return file, err
}

func (s *Service) open(name string) (File, error) {
	f, err := s.files.Open(name)
	if err != nil {
		return File{}, openError(name, err)
	}
	info, err := f.Stat()
	if err != nil {
		_ = f.Close()
		return File{}, openError(name, err)
	}
	if !info.Mode().IsRegular() {
		_ = f.Close()
		return File{}, fmt.Errorf("web file %q is not a regular file: %w", name, ErrNotFound)
	}

	content, ok := f.(io.ReadSeekCloser)
	if !ok {
		// fs.File need not support Seek, which Range requests rely on.
		data, err := io.ReadAll(f)
		_ = f.Close()
		if err != nil {
			return File{}, fmt.Errorf("read web file %q: %w", name, err)
		}
		content = readSeekNopCloser{bytes.NewReader(data)}
	}
	file := File{
		Name:         name,
		ModTime:      info.ModTime(),
		Content:      content,
		ContentType:  contentTypes[strings.ToLower(path.Ext(name))],
		CacheControl: "public, max-age=31536000, immutable",
	}
	// Vite puts content-hashed files into assets/, so browsers may keep them.
	// Other files, including index.html, are revalidated by a content ETag:
	// the file date alone misses a rollback to an older build.
	if !strings.HasPrefix(name, "assets/") {
		etag, err := contentETag(content)
		if err != nil {
			_ = content.Close()
			return File{}, fmt.Errorf("hash web file %q: %w", name, err)
		}
		file.CacheControl = "no-cache"
		file.ETag = etag
	}
	return file, nil
}

// contentETag hashes the content and rewinds it for serving.
func contentETag(content io.ReadSeeker) (string, error) {
	hash := sha256.New()
	if _, err := io.Copy(hash, content); err != nil {
		return "", err
	}
	if _, err := content.Seek(0, io.SeekStart); err != nil {
		return "", err
	}
	return fmt.Sprintf(`"%x"`, hash.Sum(nil)), nil
}

// openError keeps permission errors internal. Other errors, such as an invalid
// name or a path through a regular file, mean that the file is missing.
func openError(name string, err error) error {
	if errors.Is(err, fs.ErrPermission) {
		return fmt.Errorf("open web file %q: %w", name, err)
	}
	return fmt.Errorf("web file %q: %w", name, ErrNotFound)
}

type readSeekNopCloser struct {
	io.ReadSeeker
}

func (readSeekNopCloser) Close() error { return nil }
