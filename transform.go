package pandas

import "math"

// xbuildSeries constructs a Series directly from parallel data/valid/index
// slices. It is an internal constructor used by the transform and statistics
// helpers; the index slice is copied so callers may reuse it.
func xbuildSeries(name string, dt DType, data []any, valid []bool, index []any) *Series {
	return &Series{
		name:  name,
		dtype: dt,
		data:  data,
		valid: valid,
		index: append([]any(nil), index...),
	}
}

// xfloatAt returns the value at position i as a float64 together with whether
// it is present and numeric.
func (s *Series) xfloatAt(i int) (float64, bool) {
	if !s.valid[i] {
		return 0, false
	}
	return toFloat64(s.data[i])
}

// Abs returns a copy of the series with every present numeric value replaced by
// its absolute value. Missing values are preserved. The result dtype matches
// the source for Int64 columns and is Float64 otherwise.
func (s *Series) Abs() *Series {
	dt := Float64
	if s.dtype == Int64 {
		dt = Int64
	}
	data := make([]any, s.Len())
	valid := make([]bool, s.Len())
	for i := range s.data {
		f, ok := s.xfloatAt(i)
		if !ok {
			continue
		}
		if dt == Int64 {
			data[i] = int64(math.Abs(f))
		} else {
			data[i] = math.Abs(f)
		}
		valid[i] = true
	}
	return xbuildSeries(s.name, dt, data, valid, s.index)
}

// Round returns a copy with every present numeric value rounded to the given
// number of decimal places. Ties (a value exactly halfway between two
// candidates) are resolved to the nearest even digit — the round-half-to-even
// ("banker's rounding") rule that numpy, and therefore pandas Series.round,
// applies. Missing values are preserved and the result dtype is Float64.
func (s *Series) Round(decimals int) *Series {
	mult := math.Pow(10, float64(decimals))
	data := make([]any, s.Len())
	valid := make([]bool, s.Len())
	for i := range s.data {
		f, ok := s.xfloatAt(i)
		if !ok {
			continue
		}
		data[i] = math.RoundToEven(f*mult) / mult
		valid[i] = true
	}
	return xbuildSeries(s.name, Float64, data, valid, s.index)
}

// Clip returns a copy with present numeric values constrained to the closed
// interval [lower, upper]. Missing values are preserved and the result dtype is
// Float64.
func (s *Series) Clip(lower, upper float64) *Series {
	data := make([]any, s.Len())
	valid := make([]bool, s.Len())
	for i := range s.data {
		f, ok := s.xfloatAt(i)
		if !ok {
			continue
		}
		if f < lower {
			f = lower
		}
		if f > upper {
			f = upper
		}
		data[i] = f
		valid[i] = true
	}
	return xbuildSeries(s.name, Float64, data, valid, s.index)
}

// CumSum returns the cumulative sum of the present numeric values. Missing
// values remain missing and are skipped when accumulating, mirroring the
// pandas Series.cumsum semantics. The result dtype is Float64.
func (s *Series) CumSum() *Series {
	data := make([]any, s.Len())
	valid := make([]bool, s.Len())
	acc := 0.0
	for i := range s.data {
		f, ok := s.xfloatAt(i)
		if !ok {
			continue
		}
		acc += f
		data[i] = acc
		valid[i] = true
	}
	return xbuildSeries(s.name, Float64, data, valid, s.index)
}

// CumProd returns the cumulative product of the present numeric values. Missing
// values remain missing and are skipped. The result dtype is Float64.
func (s *Series) CumProd() *Series {
	data := make([]any, s.Len())
	valid := make([]bool, s.Len())
	acc := 1.0
	started := false
	for i := range s.data {
		f, ok := s.xfloatAt(i)
		if !ok {
			continue
		}
		if !started {
			acc = f
			started = true
		} else {
			acc *= f
		}
		data[i] = acc
		valid[i] = true
	}
	return xbuildSeries(s.name, Float64, data, valid, s.index)
}

// CumMax returns the cumulative maximum of the present numeric values. Missing
// values remain missing and are skipped. The result dtype is Float64.
func (s *Series) CumMax() *Series {
	data := make([]any, s.Len())
	valid := make([]bool, s.Len())
	var acc float64
	started := false
	for i := range s.data {
		f, ok := s.xfloatAt(i)
		if !ok {
			continue
		}
		if !started || f > acc {
			acc = f
			started = true
		}
		data[i] = acc
		valid[i] = true
	}
	return xbuildSeries(s.name, Float64, data, valid, s.index)
}

// CumMin returns the cumulative minimum of the present numeric values. Missing
// values remain missing and are skipped. The result dtype is Float64.
func (s *Series) CumMin() *Series {
	data := make([]any, s.Len())
	valid := make([]bool, s.Len())
	var acc float64
	started := false
	for i := range s.data {
		f, ok := s.xfloatAt(i)
		if !ok {
			continue
		}
		if !started || f < acc {
			acc = f
			started = true
		}
		data[i] = acc
		valid[i] = true
	}
	return xbuildSeries(s.name, Float64, data, valid, s.index)
}

// Shift returns a copy with values moved by n positions, introducing missing
// values at the vacated end. A positive n shifts values toward higher
// positions (down); a negative n shifts toward lower positions (up). The index
// labels are unchanged, matching pandas Series.shift.
func (s *Series) Shift(n int) *Series {
	data := make([]any, s.Len())
	valid := make([]bool, s.Len())
	for i := range s.data {
		src := i - n
		if src < 0 || src >= s.Len() {
			continue
		}
		data[i] = s.data[src]
		valid[i] = s.valid[src]
	}
	return xbuildSeries(s.name, s.dtype, data, valid, s.index)
}

// Diff returns the first discrete difference: element i minus element i-n. The
// first n positions, and any position where either operand is missing, are set
// to missing. The result dtype is Float64.
func (s *Series) Diff() *Series {
	return s.diffN(1)
}

func (s *Series) diffN(n int) *Series {
	data := make([]any, s.Len())
	valid := make([]bool, s.Len())
	for i := range s.data {
		src := i - n
		if src < 0 {
			continue
		}
		cur, ok1 := s.xfloatAt(i)
		prev, ok2 := s.xfloatAt(src)
		if !ok1 || !ok2 {
			continue
		}
		data[i] = cur - prev
		valid[i] = true
	}
	return xbuildSeries(s.name, Float64, data, valid, s.index)
}

// PctChange returns the fractional change between the current and prior
// element: (x[i] - x[i-1]) / x[i-1]. The first position, positions with a
// missing operand, and divisions by zero are set to missing. The result dtype
// is Float64.
func (s *Series) PctChange() *Series {
	data := make([]any, s.Len())
	valid := make([]bool, s.Len())
	for i := range s.data {
		if i == 0 {
			continue
		}
		cur, ok1 := s.xfloatAt(i)
		prev, ok2 := s.xfloatAt(i - 1)
		if !ok1 || !ok2 || prev == 0 {
			continue
		}
		data[i] = (cur - prev) / prev
		valid[i] = true
	}
	return xbuildSeries(s.name, Float64, data, valid, s.index)
}

// Astype returns a copy of the series converted to dtype dt. Values that cannot
// be coerced to the target dtype become missing, matching the coercion rules
// used at construction time.
func (s *Series) Astype(dt DType) *Series {
	data := make([]any, s.Len())
	valid := make([]bool, s.Len())
	for i := range s.data {
		if !s.valid[i] {
			continue
		}
		cv, ok := coerce(s.data[i], dt)
		data[i] = cv
		valid[i] = ok
	}
	return xbuildSeries(s.name, dt, data, valid, s.index)
}

// Between returns a boolean mask reporting, per position, whether the present
// numeric value lies in the closed interval [low, high]. Missing or
// non-numeric values yield false.
func (s *Series) Between(low, high float64) []bool {
	out := make([]bool, s.Len())
	for i := range s.data {
		f, ok := s.xfloatAt(i)
		if !ok {
			continue
		}
		out[i] = f >= low && f <= high
	}
	return out
}

// IsIn returns a boolean mask reporting, per position, whether the present
// value equals one of the given values. Comparison uses the canonical stored
// representation, so numeric values are matched after coercion to the column
// dtype.
func (s *Series) IsIn(values []any) []bool {
	set := make(map[any]struct{}, len(values))
	for _, v := range values {
		if cv, ok := coerce(v, s.dtype); ok {
			set[cv] = struct{}{}
		}
	}
	out := make([]bool, s.Len())
	for i := range s.data {
		if !s.valid[i] {
			continue
		}
		if _, ok := set[s.data[i]]; ok {
			out[i] = true
		}
	}
	return out
}

// xelementwise applies a binary float operation position by position between
// two series, up to the shorter length. A position is missing when either
// operand is missing or non-numeric. The result index is taken from s.
func (s *Series) xelementwise(other *Series, op func(a, b float64) float64) *Series {
	n := s.Len()
	if other.Len() < n {
		n = other.Len()
	}
	data := make([]any, n)
	valid := make([]bool, n)
	for i := 0; i < n; i++ {
		a, oka := s.xfloatAt(i)
		b, okb := other.xfloatAt(i)
		if !oka || !okb {
			continue
		}
		data[i] = op(a, b)
		valid[i] = true
	}
	return xbuildSeries(s.name, Float64, data, valid, s.index[:n])
}

// Add returns the element-wise sum of two series, aligned by position up to the
// shorter length. Positions where either operand is missing become missing. The
// result dtype is Float64.
func (s *Series) Add(other *Series) *Series {
	return s.xelementwise(other, func(a, b float64) float64 { return a + b })
}

// Sub returns the element-wise difference (s - other), aligned by position up
// to the shorter length. The result dtype is Float64.
func (s *Series) Sub(other *Series) *Series {
	return s.xelementwise(other, func(a, b float64) float64 { return a - b })
}

// Mul returns the element-wise product of two series, aligned by position up to
// the shorter length. The result dtype is Float64.
func (s *Series) Mul(other *Series) *Series {
	return s.xelementwise(other, func(a, b float64) float64 { return a * b })
}

// Div returns the element-wise quotient (s / other), aligned by position up to
// the shorter length. A zero divisor yields a missing value. The result dtype
// is Float64.
func (s *Series) Div(other *Series) *Series {
	n := s.Len()
	if other.Len() < n {
		n = other.Len()
	}
	data := make([]any, n)
	valid := make([]bool, n)
	for i := 0; i < n; i++ {
		a, oka := s.xfloatAt(i)
		b, okb := other.xfloatAt(i)
		if !oka || !okb || b == 0 {
			continue
		}
		data[i] = a / b
		valid[i] = true
	}
	return xbuildSeries(s.name, Float64, data, valid, s.index[:n])
}
