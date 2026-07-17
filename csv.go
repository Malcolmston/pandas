package pandas

import (
	"encoding/csv"
	"io"
	"os"
	"strconv"
	"strings"
)

// ReadCSVOptions controls CSV parsing.
type ReadCSVOptions struct {
	// Delimiter is the field separator (defaults to ',').
	Delimiter rune
	// NoHeader treats the first line as data and generates column names
	// "col0", "col1", ... instead of using it as a header.
	NoHeader bool
	// NAValues lists raw strings that should be interpreted as missing. An
	// empty field is always missing.
	NAValues []string
}

// ReadCSV parses CSV data from r into a DataFrame. Column dtypes are inferred:
// a column is Int64 if every present value parses as an integer, Float64 if
// every present value parses as a float, Bool if every value is a boolean
// literal, and String otherwise.
func ReadCSV(r io.Reader, opts *ReadCSVOptions) (*DataFrame, error) {
	o := ReadCSVOptions{Delimiter: ','}
	if opts != nil {
		o = *opts
		if o.Delimiter == 0 {
			o.Delimiter = ','
		}
	}
	cr := csv.NewReader(r)
	cr.Comma = o.Delimiter
	cr.FieldsPerRecord = -1

	records, err := cr.ReadAll()
	if err != nil {
		return nil, err
	}
	if len(records) == 0 {
		return NewDataFrame()
	}

	var header []string
	var body [][]string
	if o.NoHeader {
		for i := range records[0] {
			header = append(header, "col"+strconv.Itoa(i))
		}
		body = records
	} else {
		header = records[0]
		body = records[1:]
	}

	na := make(map[string]struct{}, len(o.NAValues))
	for _, v := range o.NAValues {
		na[v] = struct{}{}
	}
	isNA := func(s string) bool {
		if s == "" {
			return true
		}
		_, ok := na[s]
		return ok
	}

	cols := make([][]any, len(header))
	for _, rec := range body {
		for c := range header {
			var raw string
			if c < len(rec) {
				raw = rec[c]
			}
			if isNA(raw) {
				cols[c] = append(cols[c], nil)
			} else {
				cols[c] = append(cols[c], raw)
			}
		}
	}

	series := make([]*Series, len(header))
	for c, name := range header {
		dt := inferColumnDType(cols[c])
		series[c] = NewSeriesTyped(name, dt, cols[c])
	}
	return NewDataFrame(series...)
}

// ReadCSVFile reads and parses a CSV file at path.
func ReadCSVFile(path string, opts *ReadCSVOptions) (*DataFrame, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()
	return ReadCSV(f, opts)
}

// inferColumnDType chooses the narrowest dtype that fits every present value.
func inferColumnDType(vals []any) DType {
	allInt, allFloat, allBool, hasVal := true, true, true, false
	for _, v := range vals {
		if v == nil {
			continue
		}
		hasVal = true
		s, ok := v.(string)
		if !ok {
			return Object
		}
		if _, err := strconv.ParseInt(s, 10, 64); err != nil {
			allInt = false
		}
		if _, err := strconv.ParseFloat(s, 64); err != nil {
			allFloat = false
		}
		if !isBoolLiteral(s) {
			allBool = false
		}
	}
	if !hasVal {
		return String
	}
	switch {
	case allInt:
		return Int64
	case allFloat:
		return Float64
	case allBool:
		return Bool
	default:
		return String
	}
}

func isBoolLiteral(s string) bool {
	switch strings.ToLower(s) {
	case "true", "false":
		return true
	default:
		return false
	}
}

// WriteCSV writes the DataFrame to w as CSV, including a header row. Missing
// values are written as empty fields.
func (df *DataFrame) WriteCSV(w io.Writer, opts *ReadCSVOptions) error {
	cw := csv.NewWriter(w)
	if opts != nil && opts.Delimiter != 0 {
		cw.Comma = opts.Delimiter
	}
	if err := cw.Write(df.names); err != nil {
		return err
	}
	for i := 0; i < df.NumRows(); i++ {
		rec := make([]string, len(df.columns))
		for c, col := range df.columns {
			if col.valid[i] {
				rec[c] = formatValue(col.data[i])
			}
		}
		if err := cw.Write(rec); err != nil {
			return err
		}
	}
	cw.Flush()
	return cw.Error()
}

// WriteCSVFile writes the DataFrame to a CSV file at path.
func (df *DataFrame) WriteCSVFile(path string, opts *ReadCSVOptions) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	if err := df.WriteCSV(f, opts); err != nil {
		_ = f.Close()
		return err
	}
	return f.Close()
}
