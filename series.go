package pandas

import (
	"math"
	"sort"
	"strings"
)

// Series is a named, typed one-dimensional column of values with an associated
// index. Missing values (NA) are tracked with a validity mask, so a Series of
// any dtype can represent absent data without a sentinel value.
type Series struct {
	name  string
	dtype DType
	data  []any  // canonical typed values; the slot is undefined when !valid[i]
	valid []bool // false marks a missing (NA) value
	index []any  // row labels, defaults to 0..n-1
}

// NewSeries builds a Series named name from values. The dtype is inferred from
// the first non-nil element (defaulting to Object for an all-nil input) and
// every element is coerced to that dtype. A nil element, or one that cannot be
// coerced, becomes a missing value.
func NewSeries(name string, values []any) *Series {
	dt := Object
	for _, v := range values {
		if v != nil {
			dt = inferDType(v)
			break
		}
	}
	return NewSeriesTyped(name, dt, values)
}

// NewSeriesTyped builds a Series with an explicit dtype, coercing each value.
func NewSeriesTyped(name string, dt DType, values []any) *Series {
	s := &Series{
		name:  name,
		dtype: dt,
		data:  make([]any, len(values)),
		valid: make([]bool, len(values)),
		index: defaultIndex(len(values)),
	}
	for i, v := range values {
		cv, ok := coerce(v, dt)
		s.data[i] = cv
		s.valid[i] = ok
	}
	return s
}

// defaultIndex returns the positional index 0..n-1.
func defaultIndex(n int) []any {
	idx := make([]any, n)
	for i := 0; i < n; i++ {
		idx[i] = int64(i)
	}
	return idx
}

// Name returns the series name.
func (s *Series) Name() string { return s.name }

// Rename returns a copy of the series with a new name.
func (s *Series) Rename(name string) *Series {
	c := s.Copy()
	c.name = name
	return c
}

// DType returns the element type of the series.
func (s *Series) DType() DType { return s.dtype }

// Len returns the number of elements.
func (s *Series) Len() int { return len(s.data) }

// Index returns a copy of the row labels.
func (s *Series) Index() []any {
	out := make([]any, len(s.index))
	copy(out, s.index)
	return out
}

// At returns the value at position i and whether it is present (not NA).
func (s *Series) At(i int) (any, bool) {
	if i < 0 || i >= len(s.data) {
		return nil, false
	}
	return s.data[i], s.valid[i]
}

// IsNA reports, per position, whether the value is missing.
func (s *Series) IsNA() []bool {
	out := make([]bool, len(s.valid))
	for i, v := range s.valid {
		out[i] = !v
	}
	return out
}

// Values returns the values as a slice; missing entries are nil.
func (s *Series) Values() []any {
	out := make([]any, len(s.data))
	for i := range s.data {
		if s.valid[i] {
			out[i] = s.data[i]
		}
	}
	return out
}

// Copy returns a deep copy of the series.
func (s *Series) Copy() *Series {
	c := &Series{
		name:  s.name,
		dtype: s.dtype,
		data:  make([]any, len(s.data)),
		valid: make([]bool, len(s.valid)),
		index: make([]any, len(s.index)),
	}
	copy(c.data, s.data)
	copy(c.valid, s.valid)
	copy(c.index, s.index)
	return c
}

// Apply returns a new Series with fn applied to every present value. Missing
// values are passed through untouched. The result dtype is re-inferred.
func (s *Series) Apply(fn func(any) any) *Series {
	out := make([]any, len(s.data))
	for i := range s.data {
		if s.valid[i] {
			out[i] = fn(s.data[i])
		}
	}
	r := NewSeries(s.name, out)
	r.index = append([]any(nil), s.index...)
	return r
}

// Map is an alias for Apply, mirroring the pandas Series.map method.
func (s *Series) Map(fn func(any) any) *Series { return s.Apply(fn) }

// FillNA returns a copy with every missing value replaced by value (coerced to
// the series dtype).
func (s *Series) FillNA(value any) *Series {
	c := s.Copy()
	cv, ok := coerce(value, s.dtype)
	if !ok {
		return c
	}
	for i := range c.data {
		if !c.valid[i] {
			c.data[i] = cv
			c.valid[i] = true
		}
	}
	return c
}

// DropNA returns a copy with missing values removed, preserving index labels.
func (s *Series) DropNA() *Series {
	c := &Series{name: s.name, dtype: s.dtype}
	for i := range s.data {
		if s.valid[i] {
			c.data = append(c.data, s.data[i])
			c.valid = append(c.valid, true)
			c.index = append(c.index, s.index[i])
		}
	}
	return c
}

// Head returns the first n rows (all rows when n exceeds the length).
func (s *Series) Head(n int) *Series { return s.slice(0, minInt(n, s.Len())) }

// Tail returns the last n rows.
func (s *Series) Tail(n int) *Series {
	start := s.Len() - n
	if start < 0 {
		start = 0
	}
	return s.slice(start, s.Len())
}

// slice returns the half-open range [start, end).
func (s *Series) slice(start, end int) *Series {
	c := &Series{name: s.name, dtype: s.dtype}
	c.data = append(c.data, s.data[start:end]...)
	c.valid = append(c.valid, s.valid[start:end]...)
	c.index = append(c.index, s.index[start:end]...)
	return c
}

// Filter returns the rows where mask is true. The mask must match the length.
func (s *Series) Filter(mask []bool) *Series {
	c := &Series{name: s.name, dtype: s.dtype}
	for i := range s.data {
		if i < len(mask) && mask[i] {
			c.data = append(c.data, s.data[i])
			c.valid = append(c.valid, s.valid[i])
			c.index = append(c.index, s.index[i])
		}
	}
	return c
}

// Unique returns the distinct present values in first-seen order.
func (s *Series) Unique() []any {
	seen := make(map[any]struct{})
	var out []any
	for i := range s.data {
		if !s.valid[i] {
			continue
		}
		k := s.data[i]
		if _, ok := seen[k]; ok {
			continue
		}
		seen[k] = struct{}{}
		out = append(out, k)
	}
	return out
}

// ValueCounts returns a Series counting occurrences of each present value,
// sorted by descending count and then by ascending value for determinism. The
// resulting index holds the distinct values.
func (s *Series) ValueCounts() *Series {
	counts := make(map[any]int64)
	var order []any
	for i := range s.data {
		if !s.valid[i] {
			continue
		}
		k := s.data[i]
		if _, ok := counts[k]; !ok {
			order = append(order, k)
		}
		counts[k]++
	}
	sort.SliceStable(order, func(a, b int) bool {
		if counts[order[a]] != counts[order[b]] {
			return counts[order[a]] > counts[order[b]]
		}
		return less(order[a], order[b])
	})
	c := &Series{name: "count", dtype: Int64}
	for _, k := range order {
		c.data = append(c.data, counts[k])
		c.valid = append(c.valid, true)
		c.index = append(c.index, k)
	}
	return c
}

// Count returns the number of present (non-NA) values.
func (s *Series) Count() int {
	n := 0
	for _, v := range s.valid {
		if v {
			n++
		}
	}
	return n
}

// Sum returns the sum of the present numeric values and whether any existed.
func (s *Series) Sum() (float64, bool) {
	var total float64
	found := false
	for i := range s.data {
		if !s.valid[i] {
			continue
		}
		f, ok := toFloat64(s.data[i])
		if !ok {
			continue
		}
		total += f
		found = true
	}
	return total, found
}

// Mean returns the arithmetic mean of the present numeric values.
func (s *Series) Mean() (float64, bool) {
	var total float64
	n := 0
	for i := range s.data {
		if !s.valid[i] {
			continue
		}
		f, ok := toFloat64(s.data[i])
		if !ok {
			continue
		}
		total += f
		n++
	}
	if n == 0 {
		return 0, false
	}
	return total / float64(n), true
}

// Min returns the minimum of the present numeric values.
func (s *Series) Min() (float64, bool) {
	var m float64
	found := false
	for i := range s.data {
		if !s.valid[i] {
			continue
		}
		f, ok := toFloat64(s.data[i])
		if !ok {
			continue
		}
		if !found || f < m {
			m = f
			found = true
		}
	}
	return m, found
}

// Max returns the maximum of the present numeric values.
func (s *Series) Max() (float64, bool) {
	var m float64
	found := false
	for i := range s.data {
		if !s.valid[i] {
			continue
		}
		f, ok := toFloat64(s.data[i])
		if !ok {
			continue
		}
		if !found || f > m {
			m = f
			found = true
		}
	}
	return m, found
}

// Std returns the sample standard deviation (n-1 denominator) of the present
// numeric values. It requires at least two values.
func (s *Series) Std() (float64, bool) {
	mean, ok := s.Mean()
	if !ok {
		return 0, false
	}
	var ss float64
	n := 0
	for i := range s.data {
		if !s.valid[i] {
			continue
		}
		f, ok := toFloat64(s.data[i])
		if !ok {
			continue
		}
		d := f - mean
		ss += d * d
		n++
	}
	if n < 2 {
		return 0, false
	}
	return math.Sqrt(ss / float64(n-1)), true
}

// Sort returns a copy sorted by value. Missing values are placed last. When
// ascending is false the order is reversed (missing values still last).
func (s *Series) Sort(ascending bool) *Series {
	idx := make([]int, len(s.data))
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool {
		ia, ib := idx[a], idx[b]
		if !s.valid[ia] || !s.valid[ib] {
			return s.valid[ia] && !s.valid[ib]
		}
		if less(s.data[ia], s.data[ib]) {
			return ascending
		}
		if less(s.data[ib], s.data[ia]) {
			return !ascending
		}
		return false
	})
	c := &Series{name: s.name, dtype: s.dtype}
	for _, i := range idx {
		c.data = append(c.data, s.data[i])
		c.valid = append(c.valid, s.valid[i])
		c.index = append(c.index, s.index[i])
	}
	return c
}

// String renders the series as a simple aligned table.
func (s *Series) String() string {
	var b strings.Builder
	b.WriteString(s.name)
	b.WriteString(" (")
	b.WriteString(s.dtype.String())
	b.WriteString(")\n")
	for i := range s.data {
		b.WriteString(formatValue(s.index[i]))
		b.WriteByte('\t')
		if s.valid[i] {
			b.WriteString(formatValue(s.data[i]))
		} else {
			b.WriteString("NA")
		}
		b.WriteByte('\n')
	}
	return b.String()
}

// less provides a deterministic ordering across the supported comparable types.
func less(a, b any) bool {
	switch av := a.(type) {
	case float64:
		if bv, ok := toFloat64(b); ok {
			return av < bv
		}
	case int64:
		if bv, ok := toInt64(b); ok {
			return av < bv
		}
	case string:
		if bv, ok := b.(string); ok {
			return av < bv
		}
	case bool:
		if bv, ok := b.(bool); ok {
			return !av && bv
		}
	}
	return formatValue(a) < formatValue(b)
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
