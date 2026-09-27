// Package catalog_xlsx_repository reads sheets of the reference workbook as text.
package catalog_xlsx_repository

import (
	"context"
	"fmt"
	"slices"
	"strings"

	"github.com/xuri/excelize/v2"
)

// headerSearchRows is how many leading rows may hold the column names: the
// workbook puts Russian descriptions in row 1 and the names in row 2.
const headerSearchRows = 2

// SheetSpec names a sheet and the columns it must contain.
type SheetSpec struct {
	Name    string
	Columns []string
}

// Row is one data row. Values are keyed by column name and trimmed.
type Row struct {
	// Number is the 1-based Excel row number, for error messages.
	Number int
	Values map[string]string
}

// Get returns the value of the named column, or "" if the cell is empty.
func (r Row) Get(column string) string {
	return r.Values[column]
}

type Sheet struct {
	Name string
	Rows []Row
}

type Repository struct{}

func NewRepository() *Repository {
	return &Repository{}
}

// ReadSheets opens the workbook at path and reads the requested sheets. Cells
// are read as their raw stored text, so numbers are never reformatted.
func (r *Repository) ReadSheets(_ context.Context, path string, specs []SheetSpec) (map[string]Sheet, error) {
	file, err := excelize.OpenFile(path, excelize.Options{RawCellValue: true})
	if err != nil {
		return nil, fmt.Errorf("open workbook %q: %w", path, err)
	}
	defer file.Close()

	available := file.GetSheetList()
	sheets := make(map[string]Sheet, len(specs))
	for _, spec := range specs {
		if !slices.Contains(available, spec.Name) {
			return nil, fmt.Errorf("sheet %q not found; workbook has %q", spec.Name, available)
		}
		rows, err := file.GetRows(spec.Name)
		if err != nil {
			return nil, fmt.Errorf("read sheet %q: %w", spec.Name, err)
		}
		sheet, err := parseSheet(spec, rows)
		if err != nil {
			return nil, err
		}
		sheets[spec.Name] = sheet
	}
	return sheets, nil
}

// parseSheet finds the header row among the first rows by the required column
// names, then keys every following non-empty row by those names.
func parseSheet(spec SheetSpec, rows [][]string) (Sheet, error) {
	headerIndex := -1
	var columns map[string]int
	for i := 0; i < min(headerSearchRows, len(rows)); i++ {
		candidate, err := columnIndexes(spec, rows[i], i+1)
		if err != nil {
			return Sheet{}, err
		}
		if hasColumns(candidate, spec.Columns) {
			headerIndex, columns = i, candidate
			break
		}
	}
	if headerIndex < 0 {
		var found []string
		if len(rows) >= headerSearchRows {
			found = trimmed(rows[headerSearchRows-1])
		}
		return Sheet{}, fmt.Errorf("sheet %q: no header with columns %q in the first %d rows; row %d has %q",
			spec.Name, spec.Columns, headerSearchRows, headerSearchRows, found)
	}

	sheet := Sheet{Name: spec.Name}
	for i := headerIndex + 1; i < len(rows); i++ {
		values := make(map[string]string, len(columns))
		empty := true
		for name, column := range columns {
			if column < len(rows[i]) {
				value := strings.TrimSpace(rows[i][column])
				values[name] = value
				empty = empty && value == ""
			}
		}
		if !empty {
			sheet.Rows = append(sheet.Rows, Row{Number: i + 1, Values: values})
		}
	}
	return sheet, nil
}

// columnIndexes maps names to positions; a repeated name would make values ambiguous.
func columnIndexes(spec SheetSpec, row []string, rowNumber int) (map[string]int, error) {
	indexes := make(map[string]int, len(row))
	for i, cell := range row {
		name := strings.TrimSpace(cell)
		if name == "" {
			continue
		}
		if _, seen := indexes[name]; seen && slices.Contains(spec.Columns, name) {
			return nil, fmt.Errorf("sheet %q row %d: column %q appears more than once", spec.Name, rowNumber, name)
		}
		indexes[name] = i
	}
	return indexes, nil
}

func hasColumns(indexes map[string]int, required []string) bool {
	for _, name := range required {
		if _, ok := indexes[name]; !ok {
			return false
		}
	}
	return true
}

func trimmed(row []string) []string {
	result := make([]string, 0, len(row))
	for _, cell := range row {
		if value := strings.TrimSpace(cell); value != "" {
			result = append(result, value)
		}
	}
	return result
}
