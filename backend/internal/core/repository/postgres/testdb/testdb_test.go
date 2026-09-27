package core_postgres_testdb

import (
	"fmt"
	"runtime"
	"slices"
	"strings"
	"testing"
)

func TestUpSection(t *testing.T) {
	up, err := upSection("-- +goose Up\r\nCREATE TABLE a (id int);\r\n\r\n-- +goose Down\r\nDROP TABLE a;\r\n")
	if err != nil || strings.TrimSpace(up) != "CREATE TABLE a (id int);" {
		t.Fatalf("upSection = %q, %v", up, err)
	}
	for name, text := range map[string]string{
		"no annotation":  "CREATE TABLE a (id int);",
		"empty":          "-- +goose Up\n\n-- +goose Down\nDROP TABLE a;",
		"statement":      "-- +goose Up\n-- +goose StatementBegin\nSELECT 1;\n-- +goose StatementEnd\n",
		"no transaction": "-- +goose NO TRANSACTION\n-- +goose Up\nCREATE INDEX CONCURRENTLY a_idx ON a (id);",
	} {
		if _, err := upSection(text); err == nil {
			t.Errorf("%s: no error", name)
		}
	}
}

func TestReadMigrationsInNameOrder(t *testing.T) {
	migrations, err := readMigrations()
	if err != nil {
		t.Fatal(err)
	}
	names := make([]string, len(migrations))
	for i, m := range migrations {
		names[i] = m.name
		if strings.Contains(m.up, "DROP TABLE") {
			t.Errorf("%s: the Up section holds the Down statements", m.name)
		}
	}
	if len(names) < 2 || !slices.IsSorted(names) || !strings.HasSuffix(names[0], "_init_schema.sql") {
		t.Errorf("migrations = %v", names)
	}
}

// fatalRecorder stops New at its first Fatal or Skip and keeps the message.
type fatalRecorder struct {
	testing.TB
	message string
}

func (r *fatalRecorder) Helper() {}

func (r *fatalRecorder) Fatal(args ...any) {
	r.message = fmt.Sprint(args...)
	runtime.Goexit()
}

func (r *fatalRecorder) Fatalf(format string, args ...any) {
	r.message = fmt.Sprintf(format, args...)
	runtime.Goexit()
}

func (r *fatalRecorder) Skip(args ...any) {
	r.message = "skip: " + fmt.Sprint(args...)
	runtime.Goexit()
}

func TestNewHidesAnInvalidURL(t *testing.T) {
	t.Setenv(EnvDatabaseURL, "postgres://it:s3cret-pass@127.0.0.1:not-a-port/it")
	r := &fatalRecorder{TB: t}
	done := make(chan struct{})
	go func() {
		defer close(done)
		New(r)
	}()
	<-done
	if r.message != "invalid "+EnvDatabaseURL {
		t.Errorf("New failed with %q", r.message)
	}
}

func TestNewAppliesEveryMigration(t *testing.T) {
	pool := New(t)
	ctx := t.Context()
	var schema string
	if err := pool.QueryRow(ctx, "SELECT current_schema()").Scan(&schema); err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(schema, "tramcast_test_") {
		t.Fatalf("current schema = %q", schema)
	}
	rows, err := pool.Query(ctx, `SELECT table_name FROM information_schema.tables
		WHERE table_schema = current_schema() ORDER BY table_name`)
	if err != nil {
		t.Fatal(err)
	}
	var tables []string
	for rows.Next() {
		var name string
		if err := rows.Scan(&name); err != nil {
			t.Fatal(err)
		}
		tables = append(tables, name)
	}
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
	want := []string{"forecast_versions", "prediction_jobs", "routes", "routes_stops", "stops", "validation_predictions"}
	if !slices.Equal(tables, want) {
		t.Errorf("tables = %v, want %v", tables, want)
	}
	// The constraint of the second migration.
	var constraints int
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM pg_constraint
		WHERE conname = 'forecast_versions_artifacts_key' AND connamespace = current_schema()::regnamespace`).Scan(&constraints); err != nil {
		t.Fatal(err)
	}
	if constraints != 1 {
		t.Errorf("forecast_versions_artifacts_key constraints = %d, want 1", constraints)
	}

	var inner string
	t.Run("cleanup", func(t *testing.T) {
		if err := New(t).QueryRow(t.Context(), "SELECT current_schema()").Scan(&inner); err != nil {
			t.Fatal(err)
		}
	})
	var exists bool
	if err := pool.QueryRow(ctx, "SELECT EXISTS (SELECT 1 FROM pg_namespace WHERE nspname = $1)", inner).Scan(&exists); err != nil {
		t.Fatal(err)
	}
	if inner == schema || exists {
		t.Errorf("schema %q of the subtest (outer %q) exists after its cleanup: %t", inner, schema, exists)
	}
}
