package web_service

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"testing"
	"testing/fstest"
)

const indexBody = "<!doctype html><title>TramCast</title>"

// newBuild creates root/dist with a small build and a secret file outside it.
func newBuild(t *testing.T) fs.FS {
	t.Helper()
	root := t.TempDir()
	dist := filepath.Join(root, "dist")
	writeFile(t, filepath.Join(root, "secret.txt"), "outside-secret")
	writeFile(t, filepath.Join(dist, IndexFile), indexBody)
	writeFile(t, filepath.Join(dist, "assets", "app-3f2a.js"), "console.log('app')")
	writeFile(t, filepath.Join(dist, "favicon.ico"), "\x00\x00\x01\x00")
	writeFile(t, filepath.Join(dist, "report.unknownext"), "data")
	return os.DirFS(dist)
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

func readAndClose(t *testing.T, file File) string {
	t.Helper()
	defer file.Content.Close()
	data, err := io.ReadAll(file.Content)
	if err != nil {
		t.Fatal(err)
	}
	return string(data)
}

func TestGetFile(t *testing.T) {
	service := NewService(newBuild(t))
	for _, tt := range []struct {
		requestPath, name, body, contentType, cacheControl string
	}{
		{"/assets/app-3f2a.js", "assets/app-3f2a.js", "console.log('app')", "text/javascript; charset=utf-8", "public, max-age=31536000, immutable"},
		{"/favicon.ico", "favicon.ico", "\x00\x00\x01\x00", "image/x-icon", "no-cache"},
		{"/index.html", IndexFile, indexBody, "text/html; charset=utf-8", "no-cache"},
		{"/report.unknownext", "report.unknownext", "data", "", "no-cache"},
		{"//assets/./app-3f2a.js", "assets/app-3f2a.js", "console.log('app')", "text/javascript; charset=utf-8", "public, max-age=31536000, immutable"},
	} {
		t.Run(tt.requestPath, func(t *testing.T) {
			file, err := service.GetFile(context.Background(), tt.requestPath)
			if err != nil {
				t.Fatal(err)
			}
			if file.Name != tt.name || file.ContentType != tt.contentType || file.CacheControl != tt.cacheControl || file.ModTime.IsZero() {
				t.Errorf("file = %+v", file)
			}
			// Revalidated files carry a content ETag; immutable assets need none.
			wantETag := ""
			if tt.cacheControl == "no-cache" {
				wantETag = fmt.Sprintf(`"%x"`, sha256.Sum256([]byte(tt.body)))
			}
			if file.ETag != wantETag {
				t.Errorf("ETag = %q, want %q", file.ETag, wantETag)
			}
			// The full body also proves the content was rewound after hashing.
			if body := readAndClose(t, file); body != tt.body {
				t.Errorf("body = %q, want %q", body, tt.body)
			}
		})
	}
}

func TestGetFileNotFound(t *testing.T) {
	service := NewService(newBuild(t))
	for _, requestPath := range []string{
		"/", "/assets", "/assets/", "/assets/missing.js", "/routes/12",
		"/../secret.txt", "/assets/../../secret.txt", "/favicon.ico/extra.png", "/bad:name",
	} {
		t.Run(requestPath, func(t *testing.T) {
			file, err := service.GetFile(context.Background(), requestPath)
			if !errors.Is(err, ErrNotFound) {
				t.Fatalf("error = %v, want ErrNotFound", err)
			}
			if file.Content != nil {
				t.Fatal("no content must be returned with an error")
			}
		})
	}
}

func TestGetPage(t *testing.T) {
	service := NewService(newBuild(t))
	for _, requestPath := range []string{"/", "/routes/12", "/assets", "/assets/", "/../routes"} {
		t.Run("main page for "+requestPath, func(t *testing.T) {
			file, err := service.GetPage(context.Background(), requestPath)
			if err != nil {
				t.Fatal(err)
			}
			if body := readAndClose(t, file); file.Name != IndexFile || body != indexBody || file.CacheControl != "no-cache" {
				t.Fatalf("file %q, body %q, Cache-Control %q", file.Name, body, file.CacheControl)
			}
		})
	}
	t.Run("existing file", func(t *testing.T) {
		file, err := service.GetPage(context.Background(), "/favicon.ico")
		if err != nil {
			t.Fatal(err)
		}
		if file.Name != "favicon.ico" || readAndClose(t, file) != "\x00\x00\x01\x00" {
			t.Fatalf("file = %+v", file)
		}
	})
	for _, requestPath := range []string{"/assets/missing.js", "/../secret.txt"} {
		t.Run("missing file with extension "+requestPath, func(t *testing.T) {
			if _, err := service.GetPage(context.Background(), requestPath); !errors.Is(err, ErrNotFound) {
				t.Fatalf("error = %v, want ErrNotFound", err)
			}
		})
	}
}

func TestMissingBuild(t *testing.T) {
	service := NewService(os.DirFS(filepath.Join(t.TempDir(), "missing")))
	for name, get := range map[string]func() (File, error){
		"main page": func() (File, error) { return service.GetMainPage(context.Background()) },
		"page":      func() (File, error) { return service.GetPage(context.Background(), "/routes/12") },
		"file":      func() (File, error) { return service.GetFile(context.Background(), "/assets/app.js") },
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := get(); !errors.Is(err, ErrNotFound) {
				t.Fatalf("error = %v, want ErrNotFound", err)
			}
		})
	}
}

// noSeekFS hides Seek, which fs.File implementations are not required to have.
type noSeekFS struct{ fs.FS }

func (f noSeekFS) Open(name string) (fs.File, error) {
	file, err := f.FS.Open(name)
	if err != nil {
		return nil, err
	}
	return struct{ fs.File }{file}, nil
}

func TestFileWithoutSeekIsBuffered(t *testing.T) {
	files := noSeekFS{fstest.MapFS{IndexFile: {Data: []byte(indexBody)}}}
	file, err := NewService(files).GetMainPage(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := file.Content.Seek(5, io.SeekStart); err != nil {
		t.Fatalf("content must support Seek: %v", err)
	}
	if body := readAndClose(t, file); body != indexBody[5:] {
		t.Fatalf("body after seek = %q", body)
	}
}

type permissionFS struct{}

func (permissionFS) Open(string) (fs.File, error) { return nil, fs.ErrPermission }

func TestPermissionErrorIsInternal(t *testing.T) {
	_, err := NewService(permissionFS{}).GetPage(context.Background(), "/routes/12")
	if err == nil || errors.Is(err, ErrNotFound) || !errors.Is(err, fs.ErrPermission) {
		t.Fatalf("error = %v, want a permission error distinct from ErrNotFound", err)
	}
}
