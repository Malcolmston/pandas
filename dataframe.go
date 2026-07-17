package pandas

import (
	"fmt"
	"reflect"
	"sort"
	"strings"
)

// DataFrame is an ordered collection of equal-length named columns (each a
// Series) sharing a single row index.
type DataFrame struct {
	columns []*Series
	names   []string
	index   []any
}

// NewDataFrame builds a DataFrame from an ordered list of columns. Every column
// must have the same length; the first column's length defines the number of
// rows. Column names must be unique. The shared index is taken from the first
// column, or the default positional index when there are no columns.
func NewDataFrame(columns ...*Series) (*DataFrame, error) {
	df := &DataFrame{}
	if len(columns) == 0 {
		return df, nil
	}
	n := columns[0].Len()
	seen := make(map[string]struct{}, len(columns))
	for _, c := range columns {
		if c.Len() != n {
			return nil, fmt.Errorf("pandas: column %q has length %d, want %d", c.Name(), c.Len(), n)
		}
		if _, ok := seen[c.Name()]; ok {
			return nil, fmt.Errorf("pandas: duplicate column name %q", c.Name())
		}
		seen[c.Name()] = struct{}{}
	}
	if len(columns[0].index) == n {
		df.index = append([]any(nil), columns[0].index...)
	} else {
		df.index = defaultIndex(n)
	}
	for _, c := range columns {
		cc := c.Copy()
		cc.index = df.index
		df.columns = append(df.columns, cc)
		df.names = append(df.names, cc.name)
	}
	return df, nil
}

// FromMap builds a DataFrame from a map of column name to values. Because Go
// map iteration is unordered, the resulting columns are ordered by the supplied
// order slice; any names present in data but absent from order are appended in
// sorted order for determinism.
func FromMap(data map[string][]any, order []string) (*DataFrame, error) {
	var names []string
	seen := make(map[string]struct{})
	for _, name := range order {
		if _, ok := data[name]; ok {
			if _, dup := seen[name]; !dup {
				names = append(names, name)
				seen[name] = struct{}{}
			}
		}
	}
	var rest []string
	for name := range data {
		if _, ok := seen[name]; !ok {
			rest = append(rest, name)
		}
	}
	sort.Strings(rest)
	names = append(names, rest...)

	cols := make([]*Series, 0, len(names))
	for _, name := range names {
		cols = append(cols, NewSeries(name, data[name]))
	}
	return NewDataFrame(cols...)
}

// FromRecords builds a DataFrame from a slice of structs. Exported fields become
// columns, in declaration order, using the field name (or the `pandas` struct
// tag when present). A tag of "-" skips the field.
func FromRecords(records any) (*DataFrame, error) {
	rv := reflect.ValueOf(records)
	if rv.Kind() != reflect.Slice {
		return nil, fmt.Errorf("pandas: FromRecords requires a slice, got %T", records)
	}
	elem := rv.Type().Elem()
	if elem.Kind() == reflect.Pointer {
		elem = elem.Elem()
	}
	if elem.Kind() != reflect.Struct {
		return nil, fmt.Errorf("pandas: FromRecords requires a slice of structs, got %T", records)
	}

	type fieldInfo struct {
		idx  int
		name string
	}
	var fields []fieldInfo
	for i := 0; i < elem.NumField(); i++ {
		f := elem.Field(i)
		if f.PkgPath != "" { // unexported
			continue
		}
		name := f.Name
		if tag, ok := f.Tag.Lookup("pandas"); ok {
			if tag == "-" {
				continue
			}
			if tag != "" {
				name = tag
			}
		}
		fields = append(fields, fieldInfo{idx: i, name: name})
	}

	cols := make([][]any, len(fields))
	for r := 0; r < rv.Len(); r++ {
		row := rv.Index(r)
		if row.Kind() == reflect.Pointer {
			row = row.Elem()
		}
		for c, f := range fields {
			cols[c] = append(cols[c], row.Field(f.idx).Interface())
		}
	}

	series := make([]*Series, len(fields))
	for c, f := range fields {
		series[c] = NewSeries(f.name, cols[c])
	}
	return NewDataFrame(series...)
}

// Names returns the column names in order.
func (df *DataFrame) Names() []string {
	return append([]string(nil), df.names...)
}

// Shape returns the number of rows and columns.
func (df *DataFrame) Shape() (rows, cols int) {
	return df.NumRows(), len(df.columns)
}

// NumRows returns the number of rows.
func (df *DataFrame) NumRows() int {
	if len(df.columns) == 0 {
		return 0
	}
	return df.columns[0].Len()
}

// NumCols returns the number of columns.
func (df *DataFrame) NumCols() int { return len(df.columns) }

// Index returns a copy of the row labels.
func (df *DataFrame) Index() []any {
	return append([]any(nil), df.index...)
}

// colIndex returns the position of the named column, or -1.
func (df *DataFrame) colIndex(name string) int {
	for i, n := range df.names {
		if n == name {
			return i
		}
	}
	return -1
}

// HasColumn reports whether a column with the given name exists.
func (df *DataFrame) HasColumn(name string) bool { return df.colIndex(name) >= 0 }

// Col returns the named column and whether it exists.
func (df *DataFrame) Col(name string) (*Series, bool) {
	i := df.colIndex(name)
	if i < 0 {
		return nil, false
	}
	return df.columns[i], true
}

// MustCol returns the named column and panics if it is absent.
func (df *DataFrame) MustCol(name string) *Series {
	c, ok := df.Col(name)
	if !ok {
		panic(fmt.Sprintf("pandas: no column %q", name))
	}
	return c
}

// Copy returns a deep copy of the DataFrame.
func (df *DataFrame) Copy() *DataFrame {
	c := &DataFrame{
		names: append([]string(nil), df.names...),
		index: append([]any(nil), df.index...),
	}
	for _, col := range df.columns {
		cc := col.Copy()
		cc.index = c.index
		c.columns = append(c.columns, cc)
	}
	return c
}

// Select returns a new DataFrame containing only the named columns, in the
// order given. An error is returned if any name is missing.
func (df *DataFrame) Select(names ...string) (*DataFrame, error) {
	cols := make([]*Series, 0, len(names))
	for _, name := range names {
		c, ok := df.Col(name)
		if !ok {
			return nil, fmt.Errorf("pandas: no column %q", name)
		}
		cols = append(cols, c.Copy())
	}
	out := &DataFrame{index: append([]any(nil), df.index...)}
	for _, c := range cols {
		c.index = out.index
		out.columns = append(out.columns, c)
		out.names = append(out.names, c.name)
	}
	return out, nil
}

// Drop returns a new DataFrame without the named columns. Missing names are
// ignored.
func (df *DataFrame) Drop(names ...string) *DataFrame {
	remove := make(map[string]struct{}, len(names))
	for _, n := range names {
		remove[n] = struct{}{}
	}
	var keep []string
	for _, n := range df.names {
		if _, ok := remove[n]; !ok {
			keep = append(keep, n)
		}
	}
	out, _ := df.Select(keep...)
	return out
}

// Rename returns a new DataFrame with columns renamed according to mapping
// (old name -> new name). Names absent from the mapping are unchanged.
func (df *DataFrame) Rename(mapping map[string]string) *DataFrame {
	c := df.Copy()
	for i, n := range c.names {
		if nn, ok := mapping[n]; ok {
			c.names[i] = nn
			c.columns[i].name = nn
		}
	}
	return c
}

// WithColumn returns a new DataFrame with col added (or replaced if a column of
// the same name already exists). The column length must match the row count of
// a non-empty frame.
func (df *DataFrame) WithColumn(col *Series) (*DataFrame, error) {
	if df.NumCols() > 0 && col.Len() != df.NumRows() {
		return nil, fmt.Errorf("pandas: column %q has length %d, want %d", col.Name(), col.Len(), df.NumRows())
	}
	c := df.Copy()
	if len(c.index) == 0 {
		c.index = append([]any(nil), col.index...)
	}
	nc := col.Copy()
	nc.index = c.index
	if i := c.colIndex(col.Name()); i >= 0 {
		c.columns[i] = nc
		return c, nil
	}
	c.columns = append(c.columns, nc)
	c.names = append(c.names, nc.name)
	return c, nil
}

// Head returns the first n rows.
func (df *DataFrame) Head(n int) *DataFrame {
	return df.sliceRows(0, minInt(n, df.NumRows()))
}

// Tail returns the last n rows.
func (df *DataFrame) Tail(n int) *DataFrame {
	start := df.NumRows() - n
	if start < 0 {
		start = 0
	}
	return df.sliceRows(start, df.NumRows())
}

// ILoc returns the half-open range of rows [start, end) by position, mirroring
// pandas iloc slicing. Out-of-range bounds are clamped.
func (df *DataFrame) ILoc(start, end int) *DataFrame {
	if start < 0 {
		start = 0
	}
	if end > df.NumRows() {
		end = df.NumRows()
	}
	if start > end {
		start = end
	}
	return df.sliceRows(start, end)
}

// sliceRows returns rows [start, end).
func (df *DataFrame) sliceRows(start, end int) *DataFrame {
	out := &DataFrame{
		names: append([]string(nil), df.names...),
		index: append([]any(nil), df.index[start:end]...),
	}
	for _, col := range df.columns {
		nc := col.slice(start, end)
		nc.index = out.index
		out.columns = append(out.columns, nc)
	}
	return out
}

// Loc returns the rows whose index label is in labels, preserving the order of
// labels. Labels not present in the index are skipped.
func (df *DataFrame) Loc(labels ...any) *DataFrame {
	pos := make(map[any][]int)
	for i, lbl := range df.index {
		pos[lbl] = append(pos[lbl], i)
	}
	var rows []int
	for _, lbl := range labels {
		rows = append(rows, pos[lbl]...)
	}
	return df.Take(rows)
}

// Take returns the rows at the given positions, in order.
func (df *DataFrame) Take(rows []int) *DataFrame {
	out := &DataFrame{names: append([]string(nil), df.names...)}
	for _, r := range rows {
		if r >= 0 && r < df.NumRows() {
			out.index = append(out.index, df.index[r])
		}
	}
	for _, col := range df.columns {
		nc := &Series{name: col.name, dtype: col.dtype, index: out.index}
		for _, r := range rows {
			if r >= 0 && r < col.Len() {
				nc.data = append(nc.data, col.data[r])
				nc.valid = append(nc.valid, col.valid[r])
			}
		}
		out.columns = append(out.columns, nc)
	}
	return out
}

// Filter returns the rows where mask is true. The mask length must equal the
// number of rows.
func (df *DataFrame) Filter(mask []bool) *DataFrame {
	var rows []int
	for i, m := range mask {
		if m {
			rows = append(rows, i)
		}
	}
	return df.Take(rows)
}

// FilterFunc returns the rows for which pred, applied to a Row view, is true.
func (df *DataFrame) FilterFunc(pred func(Row) bool) *DataFrame {
	mask := make([]bool, df.NumRows())
	for i := 0; i < df.NumRows(); i++ {
		mask[i] = pred(df.Row(i))
	}
	return df.Filter(mask)
}

// Row is a read-only view of a single row, keyed by column name.
type Row struct {
	df  *DataFrame
	pos int
}

// Row returns a view of the row at position i.
func (df *DataFrame) Row(i int) Row { return Row{df: df, pos: i} }

// Get returns the value of the named column in this row and whether it is
// present (not NA).
func (r Row) Get(name string) (any, bool) {
	c, ok := r.df.Col(name)
	if !ok {
		return nil, false
	}
	return c.At(r.pos)
}

// Float returns the named column value as a float64.
func (r Row) Float(name string) (float64, bool) {
	v, ok := r.Get(name)
	if !ok {
		return 0, false
	}
	return toFloat64(v)
}

// Label returns the index label of this row.
func (r Row) Label() any { return r.df.index[r.pos] }

// DropNA returns a new DataFrame with rows containing any missing value
// removed.
func (df *DataFrame) DropNA() *DataFrame {
	var rows []int
	for i := 0; i < df.NumRows(); i++ {
		keep := true
		for _, col := range df.columns {
			if !col.valid[i] {
				keep = false
				break
			}
		}
		if keep {
			rows = append(rows, i)
		}
	}
	return df.Take(rows)
}

// FillNA returns a new DataFrame with missing values in the named column filled.
// When column is empty every column is filled with value.
func (df *DataFrame) FillNA(column string, value any) *DataFrame {
	c := df.Copy()
	for i, col := range c.columns {
		if column == "" || col.name == column {
			filled := col.FillNA(value)
			filled.index = c.index
			c.columns[i] = filled
		}
	}
	return c
}

// SortBy returns a new DataFrame sorted by the named columns. The ascending
// slice, when non-nil, gives the direction per key; missing entries default to
// ascending. Sorting is stable and missing values sort last.
func (df *DataFrame) SortBy(keys []string, ascending []bool) (*DataFrame, error) {
	cols := make([]*Series, len(keys))
	for i, k := range keys {
		c, ok := df.Col(k)
		if !ok {
			return nil, fmt.Errorf("pandas: no column %q", k)
		}
		cols[i] = c
	}
	asc := func(i int) bool {
		if i < len(ascending) {
			return ascending[i]
		}
		return true
	}
	rows := make([]int, df.NumRows())
	for i := range rows {
		rows[i] = i
	}
	sort.SliceStable(rows, func(a, b int) bool {
		ra, rb := rows[a], rows[b]
		for i, c := range cols {
			va, oka := c.data[ra], c.valid[ra]
			vb, okb := c.data[rb], c.valid[rb]
			if !oka || !okb {
				if oka == okb {
					continue
				}
				return oka // present value sorts before missing
			}
			if less(va, vb) {
				return asc(i)
			}
			if less(vb, va) {
				return !asc(i)
			}
		}
		return false
	})
	return df.Take(rows), nil
}

// Describe returns summary statistics (count, mean, std, min, max) for every
// numeric column. The result has one row per statistic, labelled by the "stat"
// column, and one column per numeric source column.
func (df *DataFrame) Describe() *DataFrame {
	statNames := []string{"count", "mean", "std", "min", "max"}
	out := &DataFrame{}
	statCol := NewSeries("stat", toAnySlice(statNames))
	out.columns = append(out.columns, statCol)
	out.names = append(out.names, "stat")
	out.index = statCol.index
	statCol.index = out.index

	for _, col := range df.columns {
		if col.dtype != Float64 && col.dtype != Int64 {
			continue
		}
		count := float64(col.Count())
		mean, _ := col.Mean()
		std, _ := col.Std()
		mn, _ := col.Min()
		mx, _ := col.Max()
		vals := []any{count, mean, std, mn, mx}
		s := NewSeriesTyped(col.name, Float64, vals)
		s.index = out.index
		out.columns = append(out.columns, s)
		out.names = append(out.names, col.name)
	}
	return out
}

// String renders the DataFrame as an aligned text table of its named columns.
// The row index is not shown; use Index to inspect the labels. Cells are left
// aligned and the final column carries no trailing padding, so the output is
// stable for use in tests and examples.
func (df *DataFrame) String() string {
	header := append([]string(nil), df.names...)
	rows := make([][]string, df.NumRows())
	for i := 0; i < df.NumRows(); i++ {
		cells := make([]string, 0, len(df.names))
		for _, col := range df.columns {
			if col.valid[i] {
				cells = append(cells, formatValue(col.data[i]))
			} else {
				cells = append(cells, "NA")
			}
		}
		rows[i] = cells
	}
	widths := make([]int, len(header))
	for i, h := range header {
		widths[i] = len(h)
	}
	for _, r := range rows {
		for i, c := range r {
			if len(c) > widths[i] {
				widths[i] = len(c)
			}
		}
	}
	var b strings.Builder
	writeRow := func(cells []string) {
		for i, c := range cells {
			if i > 0 {
				b.WriteString("  ")
			}
			if i == len(cells)-1 {
				b.WriteString(c)
			} else {
				b.WriteString(pad(c, widths[i]))
			}
		}
		b.WriteByte('\n')
	}
	writeRow(header)
	for _, r := range rows {
		writeRow(r)
	}
	return b.String()
}

func pad(s string, w int) string {
	if len(s) >= w {
		return s
	}
	return s + strings.Repeat(" ", w-len(s))
}

func toAnySlice[T any](in []T) []any {
	out := make([]any, len(in))
	for i, v := range in {
		out[i] = v
	}
	return out
}
