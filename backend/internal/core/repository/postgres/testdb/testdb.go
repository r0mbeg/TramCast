// Package core_postgres_testdb gives integration tests a PostgreSQL schema of
// their own with every migration applied. The tests skip unless
// TRAMCAST_TEST_DATABASE_URL names a throwaway database.
package core_postgres_testdb

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// EnvDatabaseURL names the variable with the connection string of the test
// database. Never point it at a database whose data matters.
const EnvDatabaseURL = "TRAMCAST_TEST_DATABASE_URL"

// setupTimeout bounds connecting, creating the schema and migrating it, and
// separately dropping it.
const setupTimeout = 30 * time.Second

// New returns a pool whose sessions use a new schema tramcast_test_<hex> with
// every migration of backend/migrations applied; the test cleanup closes the
// pool and drops the schema. The test is skipped when TRAMCAST_TEST_DATABASE_URL
// is unset. Tables and table locks belong to the schema, so tests and packages
// can run in parallel against one database.
func New(t testing.TB) *pgxpool.Pool {
	t.Helper()
	url := os.Getenv(EnvDatabaseURL)
	if url == "" {
		t.Skip(EnvDatabaseURL + " is not set")
	}
	config, err := pgxpool.ParseConfig(url)
	if err != nil {
		// The parse error may quote the connection string with its password.
		t.Fatal("invalid " + EnvDatabaseURL)
	}
	migrations, err := readMigrations()
	if err != nil {
		t.Fatal(err)
	}

	var suffix [8]byte
	_, _ = rand.Read(suffix[:])
	schema := "tramcast_test_" + hex.EncodeToString(suffix[:])
	admin := config.ConnConfig.Copy()
	ctx, cancel := context.WithTimeout(context.Background(), setupTimeout)
	defer cancel()
	if err := execOnce(ctx, admin, "CREATE SCHEMA "+schema); err != nil {
		t.Fatalf("create schema %s: %v", schema, err)
	}
	// Registered first, so it runs after the pool is closed and no session
	// holds a lock in the schema.
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), setupTimeout)
		defer cancel()
		if err := execOnce(ctx, admin, "DROP SCHEMA "+schema+" CASCADE"); err != nil {
			t.Errorf("drop schema %s: %v", schema, err)
		}
	})

	// The schema name is lowercase hex and underscores, safe unquoted.
	config.ConnConfig.RuntimeParams["search_path"] = schema
	config.ConnConfig.RuntimeParams["application_name"] = "tramcast_test"
	pool, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		t.Fatalf("create the test pool: %v", err)
	}
	t.Cleanup(pool.Close)
	for _, m := range migrations {
		// Without arguments pgx uses the simple protocol, which runs the whole
		// section as one implicit transaction.
		if _, err := pool.Exec(ctx, m.up); err != nil {
			t.Fatalf("apply migration %s: %v", m.name, err)
		}
	}
	return pool
}

// execOnce runs sql on a connection of its own.
func execOnce(ctx context.Context, config *pgx.ConnConfig, sql string) error {
	conn, err := pgx.ConnectConfig(ctx, config)
	if err != nil {
		return err
	}
	defer func() { _ = conn.Close(context.WithoutCancel(ctx)) }()
	_, err = conn.Exec(ctx, sql)
	return err
}

type migration struct {
	name, up string
}

// readMigrations returns the Up sections of backend/migrations in name order,
// the order goose applies them.
func readMigrations() ([]migration, error) {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		return nil, errors.New("locate backend/migrations: no caller information")
	}
	dir := filepath.Join(filepath.Dir(file), "..", "..", "..", "..", "..", "migrations")
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var migrations []migration
	// ReadDir sorts by name.
	for _, entry := range entries {
		if entry.IsDir() || filepath.Ext(entry.Name()) != ".sql" {
			continue
		}
		text, err := os.ReadFile(filepath.Join(dir, entry.Name()))
		if err != nil {
			return nil, err
		}
		up, err := upSection(string(text))
		if err != nil {
			return nil, errors.New(entry.Name() + ": " + err.Error())
		}
		migrations = append(migrations, migration{name: entry.Name(), up: up})
	}
	if len(migrations) == 0 {
		return nil, errors.New("no migrations in " + dir)
	}
	return migrations, nil
}

// upSection returns the text between -- +goose Up and -- +goose Down.
// ponytail: other goose annotations (StatementBegin, NO TRANSACTION) are
// refused rather than emulated; support them once a migration needs one.
func upSection(text string) (string, error) {
	before, rest, ok := strings.Cut(text, "-- +goose Up")
	if !ok {
		return "", errors.New("no -- +goose Up annotation")
	}
	up, _, _ := strings.Cut(rest, "-- +goose Down")
	if strings.Contains(before, "-- +goose") || strings.Contains(up, "-- +goose") {
		return "", errors.New("unsupported goose annotation")
	}
	if strings.TrimSpace(up) == "" {
		return "", errors.New("empty Up section")
	}
	return up, nil
}
