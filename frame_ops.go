package pandas

import "fmt"

// xreduceNumeric applies a per-column reduction over every numeric column,
// returning a Float64 Series whose index labels are the source column names.
func (df *DataFrame) xreduceNumeric(name string, fn func(*Series) (float64, bool)) *Series {
	c := &Series{name: name, dtype: Float64}
	for _, col := range df.columns {
		if col.dtype != Float64 && col.dtype != Int64 {
			continue
		}
		v, ok := fn(col)
		c.data = append(c.data, v)
		c.valid = append(c.valid, ok)
		c.index = append(c.index, col.name)
	}
	return c
}

// Sum returns the column-wise sum of every numeric column as a Series indexed by
// column name, mirroring the default (axis=0) pandas DataFrame.sum.
func (df *DataFrame) Sum() *Series { return df.xreduceNumeric("sum", (*Series).Sum) }

// Mean returns the column-wise mean of every numeric column as a Series indexed
// by column name.
func (df *DataFrame) Mean() *Series { return df.xreduceNumeric("mean", (*Series).Mean) }

// Min returns the column-wise minimum of every numeric column as a Series
// indexed by column name.
func (df *DataFrame) Min() *Series { return df.xreduceNumeric("min", (*Series).Min) }

// Max returns the column-wise maximum of every numeric column as a Series
// indexed by column name.
func (df *DataFrame) Max() *Series { return df.xreduceNumeric("max", (*Series).Max) }

// Std returns the column-wise sample standard deviation of every numeric column
// as a Series indexed by column name.
func (df *DataFrame) Std() *Series { return df.xreduceNumeric("std", (*Series).Std) }

// Var returns the column-wise sample variance of every numeric column as a
// Series indexed by column name.
func (df *DataFrame) Var() *Series { return df.xreduceNumeric("var", (*Series).Var) }

// Median returns the column-wise median of every numeric column as a Series
// indexed by column name.
func (df *DataFrame) Median() *Series { return df.xreduceNumeric("median", (*Series).Median) }

// Nunique returns the number of distinct present values in every column as an
// Int64 Series indexed by column name. Unlike the numeric reductions it covers
// columns of all dtypes.
func (df *DataFrame) Nunique() *Series {
	c := &Series{name: "nunique", dtype: Int64}
	for _, col := range df.columns {
		c.data = append(c.data, int64(len(col.Unique())))
		c.valid = append(c.valid, true)
		c.index = append(c.index, col.name)
	}
	return c
}

// Abs returns a copy of the DataFrame with every numeric column replaced by its
// element-wise absolute value. Non-numeric columns are copied unchanged.
func (df *DataFrame) Abs() *DataFrame {
	out := df.Copy()
	for i, col := range out.columns {
		if col.dtype != Float64 && col.dtype != Int64 {
			continue
		}
		nc := col.Abs()
		nc.index = out.index
		out.columns[i] = nc
	}
	return out
}

// Round returns a copy of the DataFrame with every numeric column rounded to the
// given number of decimal places. Non-numeric columns are copied unchanged.
func (df *DataFrame) Round(decimals int) *DataFrame {
	out := df.Copy()
	for i, col := range out.columns {
		if col.dtype != Float64 && col.dtype != Int64 {
			continue
		}
		nc := col.Round(decimals)
		nc.index = out.index
		out.columns[i] = nc
	}
	return out
}

// DropDuplicates returns a new DataFrame with duplicate rows removed, keeping the
// first occurrence of each distinct row and preserving order. Two rows are equal
// when all their columns are equal, with missing values matching missing values.
func (df *DataFrame) DropDuplicates() *DataFrame {
	seen := make(map[string]struct{})
	var rows []int
	for r := 0; r < df.NumRows(); r++ {
		key := df.xrowKey(r)
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		rows = append(rows, r)
	}
	return df.Take(rows)
}

// xrowKey builds a deterministic string identity for row r across all columns.
func (df *DataFrame) xrowKey(r int) string {
	parts := make([]string, len(df.columns))
	for i, col := range df.columns {
		if col.valid[r] {
			parts[i] = "v" + formatValue(col.data[r])
		} else {
			parts[i] = "\x00NA"
		}
	}
	key := ""
	for _, p := range parts {
		key += p + "\x1f"
	}
	return key
}

// SetIndex returns a new DataFrame using the values of the named column as the
// row index; that column is removed from the result. An error is returned if
// the column does not exist.
func (df *DataFrame) SetIndex(column string) (*DataFrame, error) {
	col, ok := df.Col(column)
	if !ok {
		return nil, fmt.Errorf("pandas: no column %q", column)
	}
	newIndex := make([]any, df.NumRows())
	for i := 0; i < df.NumRows(); i++ {
		if col.valid[i] {
			newIndex[i] = col.data[i]
		} else {
			newIndex[i] = nil
		}
	}
	out := df.Drop(column)
	out.index = newIndex
	for _, c := range out.columns {
		c.index = out.index
	}
	return out, nil
}

// ResetIndex returns a new DataFrame whose current index labels are inserted as
// a leading column named "index" and whose row index is replaced by the default
// positional index 0..n-1.
func (df *DataFrame) ResetIndex() *DataFrame {
	idxCol := NewSeries("index", append([]any(nil), df.index...))
	out := &DataFrame{index: defaultIndex(df.NumRows())}
	idxCol.index = out.index
	out.columns = append(out.columns, idxCol)
	out.names = append(out.names, "index")
	for _, col := range df.columns {
		nc := col.Copy()
		nc.index = out.index
		out.columns = append(out.columns, nc)
		out.names = append(out.names, nc.name)
	}
	return out
}

// Corr returns the Pearson correlation matrix of the numeric columns. The result
// is a square DataFrame with one row per numeric column (row index labelled by
// column name) and one column per numeric column. Pairs with fewer than two
// jointly present observations, or a column with zero variance, yield a missing
// cell; the diagonal is 1 for columns with non-zero variance.
func (df *DataFrame) Corr() *DataFrame {
	var numeric []*Series
	for _, col := range df.columns {
		if col.dtype == Float64 || col.dtype == Int64 {
			numeric = append(numeric, col)
		}
	}
	index := make([]any, len(numeric))
	for i, col := range numeric {
		index[i] = col.name
	}
	out := &DataFrame{index: index}
	for _, cj := range numeric {
		c := &Series{name: cj.name, dtype: Float64, index: out.index}
		for _, ci := range numeric {
			v, ok := ci.Corr(cj)
			c.data = append(c.data, v)
			c.valid = append(c.valid, ok)
		}
		out.columns = append(out.columns, c)
		out.names = append(out.names, cj.name)
	}
	return out
}

// Concat vertically stacks DataFrames into a single frame. The output columns
// are the union of the inputs' columns, ordered by first appearance; a frame
// lacking a column contributes missing values for it. The output index is the
// concatenation of the inputs' index labels. Concat of no frames returns an
// empty DataFrame.
func Concat(dfs ...*DataFrame) (*DataFrame, error) {
	var order []string
	seen := make(map[string]struct{})
	total := 0
	var index []any
	for _, d := range dfs {
		if d == nil {
			continue
		}
		total += d.NumRows()
		index = append(index, d.index...)
		for _, n := range d.names {
			if _, ok := seen[n]; !ok {
				seen[n] = struct{}{}
				order = append(order, n)
			}
		}
	}
	out := &DataFrame{index: index}
	if len(order) == 0 {
		return out, nil
	}
	for _, name := range order {
		data := make([]any, 0, total)
		for _, d := range dfs {
			if d == nil {
				continue
			}
			if col, ok := d.Col(name); ok {
				data = append(data, col.Values()...)
			} else {
				for i := 0; i < d.NumRows(); i++ {
					data = append(data, nil)
				}
			}
		}
		col := NewSeries(name, data)
		col.index = out.index
		out.columns = append(out.columns, col)
		out.names = append(out.names, name)
	}
	return out, nil
}

// Transpose returns a new DataFrame with rows and columns swapped: each original
// column name becomes a row index label and each original row (identified by its
// index label, formatted as text) becomes a column. All resulting columns have
// Object dtype because a row may mix source dtypes. An error is returned if the
// original index labels are not unique, since they must form distinct column
// names.
func (df *DataFrame) Transpose() (*DataFrame, error) {
	newIndex := make([]any, len(df.names))
	for i, n := range df.names {
		newIndex[i] = n
	}
	out := &DataFrame{index: newIndex}
	seen := make(map[string]struct{}, df.NumRows())
	for r := 0; r < df.NumRows(); r++ {
		name := formatValue(df.index[r])
		if _, ok := seen[name]; ok {
			return nil, fmt.Errorf("pandas: cannot transpose, duplicate index label %q", name)
		}
		seen[name] = struct{}{}
		data := make([]any, len(df.columns))
		valid := make([]bool, len(df.columns))
		for c, col := range df.columns {
			data[c] = col.data[r]
			valid[c] = col.valid[r]
		}
		col := xbuildSeries(name, Object, data, valid, out.index)
		out.columns = append(out.columns, col)
		out.names = append(out.names, name)
	}
	return out, nil
}
