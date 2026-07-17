package pandas

import (
	"fmt"
	"sort"
	"strings"
)

// AggFunc names a supported aggregation.
type AggFunc int

const (
	// AggSum sums the present numeric values.
	AggSum AggFunc = iota
	// AggMean averages the present numeric values.
	AggMean
	// AggMin is the minimum present numeric value.
	AggMin
	// AggMax is the maximum present numeric value.
	AggMax
	// AggCount counts the present values.
	AggCount
	// AggStd is the sample standard deviation of present numeric values.
	AggStd
)

func (a AggFunc) suffix() string {
	switch a {
	case AggSum:
		return "sum"
	case AggMean:
		return "mean"
	case AggMin:
		return "min"
	case AggMax:
		return "max"
	case AggCount:
		return "count"
	case AggStd:
		return "std"
	default:
		return "agg"
	}
}

// GroupBy is a lazy grouping of a DataFrame by one or more key columns.
type GroupBy struct {
	df       *DataFrame
	keys     []string
	groupKey []string // deterministic ordering of group identities
	rows     map[string][]int
	keyVals  map[string][]any
}

// GroupBy groups the DataFrame by the given key columns. Groups are ordered
// deterministically by their key values.
func (df *DataFrame) GroupBy(keys ...string) (*GroupBy, error) {
	for _, k := range keys {
		if !df.HasColumn(k) {
			return nil, fmt.Errorf("pandas: no column %q", k)
		}
	}
	gb := &GroupBy{
		df:      df,
		keys:    keys,
		rows:    make(map[string][]int),
		keyVals: make(map[string][]any),
	}
	keyCols := make([]*Series, len(keys))
	for i, k := range keys {
		keyCols[i], _ = df.Col(k)
	}
	for r := 0; r < df.NumRows(); r++ {
		parts := make([]string, len(keys))
		vals := make([]any, len(keys))
		for i, c := range keyCols {
			if c.valid[r] {
				vals[i] = c.data[r]
				parts[i] = formatValue(c.data[r])
			} else {
				vals[i] = nil
				parts[i] = "\x00NA"
			}
		}
		id := strings.Join(parts, "\x1f")
		if _, ok := gb.rows[id]; !ok {
			gb.groupKey = append(gb.groupKey, id)
			gb.keyVals[id] = vals
		}
		gb.rows[id] = append(gb.rows[id], r)
	}
	sort.SliceStable(gb.groupKey, func(a, b int) bool {
		va := gb.keyVals[gb.groupKey[a]]
		vb := gb.keyVals[gb.groupKey[b]]
		for i := range va {
			if lessNA(va[i], vb[i]) {
				return true
			}
			if lessNA(vb[i], va[i]) {
				return false
			}
		}
		return false
	})
	return gb, nil
}

// lessNA orders values with nil (missing) sorting last.
func lessNA(a, b any) bool {
	if a == nil || b == nil {
		return a != nil && b == nil
	}
	return less(a, b)
}

// Groups returns the number of groups.
func (gb *GroupBy) Groups() int { return len(gb.groupKey) }

// Agg applies aggregations to columns. The spec maps a source column name to
// the aggregations to compute for it. The result DataFrame has the key columns
// followed by one column per (source column, aggregation) named
// "<column>_<agg>", one row per group in deterministic order.
func (gb *GroupBy) Agg(spec map[string][]AggFunc, order []string) (*DataFrame, error) {
	// Determine deterministic ordering of the (column, agg) outputs.
	var cols []string
	if len(order) > 0 {
		cols = append(cols, order...)
	} else {
		for c := range spec {
			cols = append(cols, c)
		}
		sort.Strings(cols)
	}
	for _, c := range cols {
		if !gb.df.HasColumn(c) {
			return nil, fmt.Errorf("pandas: no column %q", c)
		}
	}

	// Build key columns.
	keySeries := make([]*Series, len(gb.keys))
	for i, k := range gb.keys {
		src, _ := gb.df.Col(k)
		s := &Series{name: k, dtype: src.dtype}
		for _, id := range gb.groupKey {
			v := gb.keyVals[id][i]
			s.data = append(s.data, v)
			s.valid = append(s.valid, v != nil)
		}
		keySeries[i] = s
	}

	out := make([]*Series, 0, len(gb.keys)+len(cols))
	out = append(out, keySeries...)

	for _, c := range cols {
		src, _ := gb.df.Col(c)
		for _, af := range spec[c] {
			s := &Series{name: c + "_" + af.suffix(), dtype: Float64}
			if af == AggCount {
				s.dtype = Int64
			}
			for _, id := range gb.groupKey {
				sub := src.Take(gb.rows[id])
				val, ok := applyAgg(sub, af)
				s.data = append(s.data, val)
				s.valid = append(s.valid, ok)
			}
			out = append(out, s)
		}
	}
	return NewDataFrame(out...)
}

// Take extracts the given rows from a Series (used by GroupBy aggregation).
func (s *Series) Take(rows []int) *Series {
	c := &Series{name: s.name, dtype: s.dtype}
	for _, r := range rows {
		if r >= 0 && r < s.Len() {
			c.data = append(c.data, s.data[r])
			c.valid = append(c.valid, s.valid[r])
			c.index = append(c.index, s.index[r])
		}
	}
	return c
}

// applyAgg computes a single aggregation over a Series, returning the value and
// whether it is defined.
func applyAgg(s *Series, af AggFunc) (any, bool) {
	switch af {
	case AggSum:
		v, ok := s.Sum()
		return v, ok
	case AggMean:
		v, ok := s.Mean()
		return v, ok
	case AggMin:
		v, ok := s.Min()
		return v, ok
	case AggMax:
		v, ok := s.Max()
		return v, ok
	case AggStd:
		v, ok := s.Std()
		return v, ok
	case AggCount:
		return int64(s.Count()), true
	default:
		return nil, false
	}
}

// Sum aggregates the given columns with AggSum. It is a convenience wrapper
// around Agg.
func (gb *GroupBy) Sum(columns ...string) (*DataFrame, error) {
	return gb.single(AggSum, columns)
}

// Mean aggregates the given columns with AggMean.
func (gb *GroupBy) Mean(columns ...string) (*DataFrame, error) {
	return gb.single(AggMean, columns)
}

// Min aggregates the given columns with AggMin.
func (gb *GroupBy) Min(columns ...string) (*DataFrame, error) {
	return gb.single(AggMin, columns)
}

// Max aggregates the given columns with AggMax.
func (gb *GroupBy) Max(columns ...string) (*DataFrame, error) {
	return gb.single(AggMax, columns)
}

// Std aggregates the given columns with AggStd.
func (gb *GroupBy) Std(columns ...string) (*DataFrame, error) {
	return gb.single(AggStd, columns)
}

// Count returns the per-group count of the given columns (or of every non-key
// column when none are named).
func (gb *GroupBy) Count(columns ...string) (*DataFrame, error) {
	if len(columns) == 0 {
		columns = gb.nonKeyColumns()
	}
	return gb.single(AggCount, columns)
}

func (gb *GroupBy) single(af AggFunc, columns []string) (*DataFrame, error) {
	spec := make(map[string][]AggFunc, len(columns))
	for _, c := range columns {
		spec[c] = []AggFunc{af}
	}
	return gb.Agg(spec, columns)
}

func (gb *GroupBy) nonKeyColumns() []string {
	keySet := make(map[string]struct{}, len(gb.keys))
	for _, k := range gb.keys {
		keySet[k] = struct{}{}
	}
	var out []string
	for _, n := range gb.df.names {
		if _, ok := keySet[n]; !ok {
			out = append(out, n)
		}
	}
	return out
}
