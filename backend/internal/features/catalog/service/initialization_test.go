package catalog_service

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"

	catalog_osm_repository "github.com/r0mbeg/TramCast/backend/internal/features/catalog/repository/osm"
	catalog_xlsx_repository "github.com/r0mbeg/TramCast/backend/internal/features/catalog/repository/xlsx"
)

// initializationTx exercises the real sqlc calls used before source parsing.
// Unexpected writes fail the test; database concurrency and successful writes
// require PostgreSQL, rather than a fake implementation of its transactions.
type initializationTx struct {
	pgx.Tx
	t        *testing.T
	hasData  bool
	lockErr  error
	checkErr error
	calls    []string
}

func (tx *initializationTx) Exec(_ context.Context, sql string, _ ...any) (pgconn.CommandTag, error) {
	if !strings.HasPrefix(sql, "-- name: LockCatalogImport ") {
		tx.t.Fatalf("unexpected write before catalog validation: %s", sql)
	}
	tx.calls = append(tx.calls, "lock")
	return pgconn.CommandTag{}, tx.lockErr
}

func (tx *initializationTx) QueryRow(_ context.Context, sql string, _ ...any) pgx.Row {
	if !strings.HasPrefix(sql, "-- name: CatalogHasData ") {
		tx.t.Fatalf("unexpected query before catalog validation: %s", sql)
	}
	tx.calls = append(tx.calls, "check")
	return initializationRow{tx.hasData, tx.checkErr}
}

func (tx *initializationTx) Commit(context.Context) error {
	tx.calls = append(tx.calls, "commit")
	return nil
}

func (tx *initializationTx) Rollback(ctx context.Context) error {
	if ctx.Err() != nil {
		tx.t.Error("rollback must remain possible after caller cancellation")
	}
	tx.calls = append(tx.calls, "rollback")
	return nil
}

type initializationRow struct {
	hasData bool
	err     error
}

func (row initializationRow) Scan(dest ...any) error {
	if row.err != nil {
		return row.err
	}
	*dest[0].(*bool) = row.hasData
	return nil
}

type initializationDB struct {
	t   *testing.T
	tx  *initializationTx
	err error
}

func (db initializationDB) BeginTx(_ context.Context, options pgx.TxOptions) (pgx.Tx, error) {
	if options.IsoLevel != pgx.ReadCommitted {
		db.t.Errorf("isolation = %q, want READ COMMITTED", options.IsoLevel)
	}
	return db.tx, db.err
}

type unreadWorkbook struct{ t *testing.T }

func (r unreadWorkbook) ReadSheets(context.Context, string, []catalog_xlsx_repository.SheetSpec) (map[string]catalog_xlsx_repository.Sheet, error) {
	r.t.Fatal("existing catalog must not read the workbook")
	return nil, nil
}

type unreadOSM struct{ t *testing.T }

func (r unreadOSM) ReadSnapshot(context.Context, string) (catalog_osm_repository.Snapshot, error) {
	r.t.Fatal("existing catalog must not read the OSM snapshot")
	return catalog_osm_repository.Snapshot{}, nil
}

func TestImportIfEmptySkipsWithoutReadingSources(t *testing.T) {
	for _, dryRun := range []bool{false, true} {
		t.Run(map[bool]string{false: "normal", true: "dry run"}[dryRun], func(t *testing.T) {
			tx := &initializationTx{t: t, hasData: true}
			service := NewService(unreadWorkbook{t}, unreadOSM{t}, initializationDB{t: t, tx: tx})
			report, err := service.ImportIfEmpty(context.Background(), Sources{
				Workbook: "missing.xlsx", OSM: "missing.json",
			}, dryRun)
			if err != nil || !reflect.DeepEqual(report, Report{DryRun: dryRun, Skipped: true}) {
				t.Fatalf("report = %+v, error = %v; want skipped", report, err)
			}
			if want := []string{"lock", "check", "rollback"}; !reflect.DeepEqual(tx.calls, want) {
				t.Fatalf("transaction calls = %v, want %v", tx.calls, want)
			}
		})
	}
}

func TestImportIfEmptyRollsBackOnFailureBeforeWriting(t *testing.T) {
	failure := errors.New("source or database unavailable")
	validWorkbook := fakeReader{sheets: toSheets(validSheets())}
	invalid := validSheets()
	invalid[StopsSheet][0]["is_deleted"] = "1"
	for _, tt := range []struct {
		name       string
		workbook   WorkbookReader
		osm        OSMReader
		lockErr    error
		checkErr   error
		validation bool
		wantCalls  []string
	}{
		{"lock", nil, nil, failure, nil, false, []string{"lock", "rollback"}},
		{"check", nil, nil, nil, failure, false, []string{"lock", "check", "rollback"}},
		{"workbook", fakeReader{err: failure}, nil, nil, nil, false, []string{"lock", "check", "rollback"}},
		{"OSM", validWorkbook, fakeOSMReader{err: failure}, nil, nil, false, []string{"lock", "check", "rollback"}},
		{"validation", fakeReader{sheets: toSheets(invalid)}, catalog_osm_repository.New(), nil, nil, true, []string{"lock", "check", "rollback"}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if tt.workbook == nil {
				tt.workbook = unreadWorkbook{t}
			}
			if tt.osm == nil {
				tt.osm = unreadOSM{t}
			}
			tx := &initializationTx{t: t, lockErr: tt.lockErr, checkErr: tt.checkErr}
			service := NewService(tt.workbook, tt.osm, initializationDB{t: t, tx: tx})
			_, err := service.ImportIfEmpty(context.Background(), Sources{Workbook: "catalog.xlsx", OSM: committedSnapshot}, false)
			var validation *ValidationError
			if tt.validation {
				if !errors.As(err, &validation) {
					t.Fatalf("error = %v, want validation error", err)
				}
			} else if !errors.Is(err, failure) {
				t.Fatalf("error = %v, want wrapped failure", err)
			}
			if !reflect.DeepEqual(tx.calls, tt.wantCalls) {
				t.Fatalf("transaction calls = %v, want %v", tx.calls, tt.wantCalls)
			}
		})
	}
}

func TestImportIfEmptyBeginFailureDoesNotReadSources(t *testing.T) {
	failure := errors.New("database unavailable")
	service := NewService(unreadWorkbook{t}, unreadOSM{t}, initializationDB{t: t, err: failure})
	if _, err := service.ImportIfEmpty(context.Background(), Sources{}, false); !errors.Is(err, failure) {
		t.Fatalf("error = %v, want begin failure", err)
	}
}

func TestImportIfEmptyCancellationStillRollsBack(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	tx := &initializationTx{t: t, lockErr: context.Canceled}
	service := NewService(unreadWorkbook{t}, unreadOSM{t}, initializationDB{t: t, tx: tx})
	if _, err := service.ImportIfEmpty(ctx, Sources{}, false); !errors.Is(err, context.Canceled) {
		t.Fatalf("error = %v, want cancellation", err)
	}
	if want := []string{"lock", "rollback"}; !reflect.DeepEqual(tx.calls, want) {
		t.Fatalf("transaction calls = %v, want %v", tx.calls, want)
	}
}
