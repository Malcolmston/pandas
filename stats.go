package pandas

import (
	"math"
	"sort"
)

// xpresentFloats returns the present numeric values in position order together
// with their originating positions.
func (s *Series) xpresentFloats() (vals []float64, pos []int) {
	for i := range s.data {
		f, ok := s.xfloatAt(i)
		if !ok {
			continue
		}
		vals = append(vals, f)
		pos = append(pos, i)
	}
	return vals, pos
}

// Var returns the sample variance (n-1 denominator) of the present numeric
// values and whether at least two existed.
func (s *Series) Var() (float64, bool) {
	mean, ok := s.Mean()
	if !ok {
		return 0, false
	}
	var ss float64
	n := 0
	vals, _ := s.xpresentFloats()
	for _, f := range vals {
		d := f - mean
		ss += d * d
		n++
	}
	if n < 2 {
		return 0, false
	}
	return ss / float64(n-1), true
}

// Prod returns the product of the present numeric values and whether any
// existed.
func (s *Series) Prod() (float64, bool) {
	vals, _ := s.xpresentFloats()
	if len(vals) == 0 {
		return 0, false
	}
	p := 1.0
	for _, f := range vals {
		p *= f
	}
	return p, true
}

// Quantile returns the q-th quantile (0 <= q <= 1) of the present numeric
// values using linear interpolation between the two nearest ranks, matching the
// default pandas Series.quantile behaviour. It reports false when there are no
// present values or q is out of range.
func (s *Series) Quantile(q float64) (float64, bool) {
	if q < 0 || q > 1 {
		return 0, false
	}
	vals, _ := s.xpresentFloats()
	if len(vals) == 0 {
		return 0, false
	}
	sort.Float64s(vals)
	if len(vals) == 1 {
		return vals[0], true
	}
	pos := q * float64(len(vals)-1)
	lo := int(math.Floor(pos))
	hi := int(math.Ceil(pos))
	if lo == hi {
		return vals[lo], true
	}
	frac := pos - float64(lo)
	return vals[lo] + frac*(vals[hi]-vals[lo]), true
}

// Median returns the median (0.5 quantile) of the present numeric values and
// whether any existed.
func (s *Series) Median() (float64, bool) {
	return s.Quantile(0.5)
}

// Mode returns the most frequently occurring present value(s), in ascending
// order. When several values tie for the highest frequency all of them are
// returned. An empty slice is returned when there are no present values.
func (s *Series) Mode() []any {
	counts := make(map[any]int)
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
	best := 0
	for _, c := range counts {
		if c > best {
			best = c
		}
	}
	var out []any
	for _, k := range order {
		if counts[k] == best {
			out = append(out, k)
		}
	}
	sort.SliceStable(out, func(a, b int) bool { return less(out[a], out[b]) })
	return out
}

// ArgMax returns the position of the maximum present numeric value and whether
// one existed. The first occurrence wins on ties.
func (s *Series) ArgMax() (int, bool) {
	var m float64
	arg := -1
	for i := range s.data {
		f, ok := s.xfloatAt(i)
		if !ok {
			continue
		}
		if arg < 0 || f > m {
			m = f
			arg = i
		}
	}
	return arg, arg >= 0
}

// ArgMin returns the position of the minimum present numeric value and whether
// one existed. The first occurrence wins on ties.
func (s *Series) ArgMin() (int, bool) {
	var m float64
	arg := -1
	for i := range s.data {
		f, ok := s.xfloatAt(i)
		if !ok {
			continue
		}
		if arg < 0 || f < m {
			m = f
			arg = i
		}
	}
	return arg, arg >= 0
}

// Cov returns the sample covariance (n-1 denominator) between this series and
// other, computed over positions where both hold a present numeric value. It
// reports false when fewer than two such positions exist.
func (s *Series) Cov(other *Series) (float64, bool) {
	xs, ys := s.xpairedFloats(other)
	n := len(xs)
	if n < 2 {
		return 0, false
	}
	mx, my := xmean(xs), xmean(ys)
	var sum float64
	for i := range xs {
		sum += (xs[i] - mx) * (ys[i] - my)
	}
	return sum / float64(n-1), true
}

// Corr returns the Pearson correlation coefficient between this series and
// other, computed over positions where both hold a present numeric value. It
// reports false when fewer than two such positions exist or either series has
// zero variance.
func (s *Series) Corr(other *Series) (float64, bool) {
	xs, ys := s.xpairedFloats(other)
	n := len(xs)
	if n < 2 {
		return 0, false
	}
	mx, my := xmean(xs), xmean(ys)
	var sxy, sxx, syy float64
	for i := range xs {
		dx := xs[i] - mx
		dy := ys[i] - my
		sxy += dx * dy
		sxx += dx * dx
		syy += dy * dy
	}
	if sxx == 0 || syy == 0 {
		return 0, false
	}
	return sxy / math.Sqrt(sxx*syy), true
}

// xpairedFloats returns the numeric values of s and other at positions where
// both are present, aligned by position up to the shorter length.
func (s *Series) xpairedFloats(other *Series) (xs, ys []float64) {
	n := s.Len()
	if other.Len() < n {
		n = other.Len()
	}
	for i := 0; i < n; i++ {
		a, oka := s.xfloatAt(i)
		b, okb := other.xfloatAt(i)
		if !oka || !okb {
			continue
		}
		xs = append(xs, a)
		ys = append(ys, b)
	}
	return xs, ys
}

func xmean(v []float64) float64 {
	var sum float64
	for _, f := range v {
		sum += f
	}
	return sum / float64(len(v))
}

// Rank returns the ranks of the present numeric values using the average
// method, matching the pandas Series.rank default: values are ranked from 1
// (smallest) and tied values share the mean of the ranks they span. Missing
// values remain missing. The result dtype is Float64.
func (s *Series) Rank() *Series { return s.RankBy("average") }

// RankBy returns the ranks of the present numeric values, resolving ties
// according to method, one of:
//
//   - "average": tied values share the mean of the ranks they span (the default).
//   - "min":     tied values all take the lowest rank in the group.
//   - "max":     tied values all take the highest rank in the group.
//   - "first":   ties are broken by original position, giving distinct ranks.
//   - "dense":   like "min" but group ranks always increase by one, leaving no
//     gaps after ties.
//
// These mirror the pandas Series.rank method= argument. Missing values remain
// missing and the result dtype is Float64. An unrecognised method falls back to
// "average".
func (s *Series) RankBy(method string) *Series {
	vals, pos := s.xpresentFloats()
	order := make([]int, len(vals))
	for i := range order {
		order[i] = i
	}
	// A stable sort by value keeps tied values in original-position order,
	// which is exactly what the "first" method requires.
	sort.SliceStable(order, func(a, b int) bool { return vals[order[a]] < vals[order[b]] })

	ranks := make([]float64, len(vals))
	dense := 0
	i := 0
	for i < len(order) {
		j := i
		for j+1 < len(order) && vals[order[j+1]] == vals[order[i]] {
			j++
		}
		dense++
		for k := i; k <= j; k++ {
			switch method {
			case "min":
				ranks[order[k]] = float64(i + 1)
			case "max":
				ranks[order[k]] = float64(j + 1)
			case "first":
				ranks[order[k]] = float64(k + 1)
			case "dense":
				ranks[order[k]] = float64(dense)
			default: // "average"
				sum := 0.0
				for m := i; m <= j; m++ {
					sum += float64(m + 1)
				}
				ranks[order[k]] = sum / float64(j-i+1)
			}
		}
		i = j + 1
	}

	data := make([]any, s.Len())
	valid := make([]bool, s.Len())
	for k, p := range pos {
		data[p] = ranks[k]
		valid[p] = true
	}
	return xbuildSeries(s.name, Float64, data, valid, s.index)
}

// NLargest returns the n largest present values as a new series, ordered from
// largest to smallest, preserving each value's original index label. Ties are
// broken by original position. When n exceeds the number of present values all
// of them are returned.
func (s *Series) NLargest(n int) *Series {
	return s.xnextreme(n, false)
}

// NSmallest returns the n smallest present values as a new series, ordered from
// smallest to largest, preserving each value's original index label. Ties are
// broken by original position. When n exceeds the number of present values all
// of them are returned.
func (s *Series) NSmallest(n int) *Series {
	return s.xnextreme(n, true)
}

func (s *Series) xnextreme(n int, smallest bool) *Series {
	type pv struct {
		pos int
		f   float64
	}
	var items []pv
	for i := range s.data {
		f, ok := s.xfloatAt(i)
		if !ok {
			continue
		}
		items = append(items, pv{pos: i, f: f})
	}
	sort.SliceStable(items, func(a, b int) bool {
		if items[a].f != items[b].f {
			if smallest {
				return items[a].f < items[b].f
			}
			return items[a].f > items[b].f
		}
		return items[a].pos < items[b].pos
	})
	if n < 0 {
		n = 0
	}
	if n > len(items) {
		n = len(items)
	}
	c := &Series{name: s.name, dtype: s.dtype}
	for _, it := range items[:n] {
		c.data = append(c.data, s.data[it.pos])
		c.valid = append(c.valid, true)
		c.index = append(c.index, s.index[it.pos])
	}
	return c
}
