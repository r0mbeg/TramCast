package catalog_xlsx_repository

import (
	"context"
	"path/filepath"
	"strings"
	"testing"

	"github.com/xuri/excelize/v2"
)

// writeWorkbook saves sheets (name → rows from A1) to a temporary file.
func writeWorkbook(t *testing.T, sheets map[string][][]string, edit func(*excelize.File)) string {
	t.Helper()
	file := excelize.NewFile()
	defer file.Close()
	first := true
	for name, rows := range sheets {
		if first {
			if err := file.SetSheetName("Sheet1", name); err != nil {
				t.Fatal(err)
			}
			first = false
		} else if _, err := file.NewSheet(name); err != nil {
			t.Fatal(err)
		}
		for i, row := range rows {
			cell, err := excelize.CoordinatesToCellName(1, i+1)
			if err != nil {
				t.Fatal(err)
			}
			if err := file.SetSheetRow(name, cell, &row); err != nil {
				t.Fatal(err)
			}
		}
	}
	if edit != nil {
		edit(file)
	}
	path := filepath.Join(t.TempDir(), "catalog.xlsx")
	if err := file.SaveAs(path); err != nil {
		t.Fatal(err)
	}
	return path
}

var stopsSpec = SheetSpec{Name: "Остановки GTFS_STOPS", Columns: []string{"stop_id", "stop_name", "stop_lat"}}

func TestReadSheetsWithDescriptionRow(t *testing.T) {
	path := writeWorkbook(t, map[string][][]string{
		"Остановки GTFS_STOPS": {
			{"идентификатор по остановке", "", "широта"},
			{"stop_id", "stop_name", "stop_lat", "is_deleted"},
			{"3038", " 1-й Нагатинский проезд ", "55.6754278100000022", "0"},
			{},
			{"7609", "Short row"},
		},
	}, func(f *excelize.File) {
		// A numeric cell must come back as its stored text, not reformatted.
		if err := f.SetCellDefault("Остановки GTFS_STOPS", "C5", "55.6754391999999996"); err != nil {
			t.Fatal(err)
		}
	})

	sheets, err := NewRepository().ReadSheets(context.Background(), path, []SheetSpec{stopsSpec})
	if err != nil {
		t.Fatal(err)
	}
	rows := sheets["Остановки GTFS_STOPS"].Rows
	if len(rows) != 2 {
		t.Fatalf("rows = %+v, want 2 data rows", rows)
	}
	first, second := rows[0], rows[1]
	if first.Number != 3 || first.Get("stop_id") != "3038" || first.Get("stop_name") != "1-й Нагатинский проезд" ||
		first.Get("stop_lat") != "55.6754278100000022" || first.Get("is_deleted") != "0" {
		t.Errorf("first row = %+v", first)
	}
	// Row 4 is empty and skipped; numbering follows Excel rows.
	if second.Number != 5 || second.Get("stop_lat") != "55.6754391999999996" || second.Get("is_deleted") != "" {
		t.Errorf("second row = %+v", second)
	}
}

func TestReadSheetsWithHeaderInFirstRow(t *testing.T) {
	path := writeWorkbook(t, map[string][][]string{
		"Остановки GTFS_STOPS": {{"stop_id", "stop_name", "stop_lat"}, {"3038", "Stop", "55.1"}},
	}, nil)
	sheets, err := NewRepository().ReadSheets(context.Background(), path, []SheetSpec{stopsSpec})
	if err != nil {
		t.Fatal(err)
	}
	if rows := sheets["Остановки GTFS_STOPS"].Rows; len(rows) != 1 || rows[0].Number != 2 || rows[0].Get("stop_id") != "3038" {
		t.Fatalf("rows = %+v", rows)
	}
}

func TestReadSheetsErrors(t *testing.T) {
	for _, tt := range []struct {
		name   string
		sheets map[string][][]string
		want   []string
	}{
		{
			name:   "missing sheet",
			sheets: map[string][][]string{"Другой лист": {{"a"}}},
			want:   []string{`"Остановки GTFS_STOPS" not found`, "Другой лист"},
		},
		{
			name:   "missing column",
			sheets: map[string][][]string{"Остановки GTFS_STOPS": {{"описание"}, {"stop_id", "stop_name"}, {"1", "A"}}},
			want:   []string{"no header", "stop_lat", `["stop_id" "stop_name"]`},
		},
		{
			name:   "repeated column",
			sheets: map[string][][]string{"Остановки GTFS_STOPS": {{"stop_id", "stop_name", "stop_lat", "stop_id"}}},
			want:   []string{`column "stop_id" appears more than once`},
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			path := writeWorkbook(t, tt.sheets, nil)
			_, err := NewRepository().ReadSheets(context.Background(), path, []SheetSpec{stopsSpec})
			if err == nil {
				t.Fatal("expected an error")
			}
			for _, want := range tt.want {
				if !strings.Contains(err.Error(), want) {
					t.Errorf("error %q does not contain %q", err, want)
				}
			}
		})
	}
}

func TestReadSheetsMissingFile(t *testing.T) {
	_, err := NewRepository().ReadSheets(context.Background(), filepath.Join(t.TempDir(), "missing.xlsx"), []SheetSpec{stopsSpec})
	if err == nil || !strings.Contains(err.Error(), "open workbook") {
		t.Fatalf("error = %v, want an open error", err)
	}
}
