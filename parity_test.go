package pandas

import "testing"

// The assertions in this file encode concrete known-answer vectors taken
// directly from the upstream pandas test suite (pandas-dev/pandas, tag
// v2.2.3) and from numpy's documented reduction/rounding semantics that
// pandas defers to. Each test names the upstream source it mirrors so the
// Go port can be checked for behavioural parity with the original library.

// TestParityRankMethods mirrors pandas/tests/series/methods/test_rank.py,
// whose fixtures assert the exact ranks of
// Series([1, 3, 4, 2, nan, 2, 1, 5, nan, 3]) for each tie-breaking method.
func TestParityRankMethods(t *testing.T) {
	s := NewSeries("x", []any{1.0, 3, 4, 2, nil, 2, 1, 5, nil, 3})

	cases := []struct {
		method string
		want   []string
	}{
		// upstream "average"
		{"average", []string{"1.5", "5.5", "7", "3.5", "NA", "3.5", "1.5", "8", "NA", "5.5"}},
		// upstream "min"
		{"min", []string{"1", "5", "7", "3", "NA", "3", "1", "8", "NA", "5"}},
		// upstream "max"
		{"max", []string{"2", "6", "7", "4", "NA", "4", "2", "8", "NA", "6"}},
		// upstream "first"
		{"first", []string{"1", "5", "7", "3", "NA", "4", "2", "8", "NA", "6"}},
		// upstream "dense"
		{"dense", []string{"1", "3", "4", "2", "NA", "2", "1", "5", "NA", "3"}},
	}
	for _, c := range cases {
		xeqStr(t, xdump(s.RankBy(c.method)), c.want)
	}
	// The default Rank() must equal the "average" method.
	xeqStr(t, xdump(s.Rank()), cases[0].want)
}

// TestParityDiff mirrors pandas/tests/series/methods/test_diff.py::test_diff_np,
// where Series(np.arange(5)).diff() yields [NaN, 1, 1, 1, 1].
func TestParityDiff(t *testing.T) {
	s := NewSeries("x", []any{0, 1, 2, 3, 4})
	xeqStr(t, xdump(s.Diff()), []string{"NA", "1", "1", "1", "1"})
}

// TestParityPctChange mirrors pandas/tests/series/methods/test_pct_change.py.
// The modern pandas semantics (fill_method=None) compute x/x.shift(1) - 1
// without forward-filling, which is exactly what Series.PctChange does.
func TestParityPctChange(t *testing.T) {
	// test_pct_change (fill_method=None): a mid-series NA propagates.
	s := NewSeries("x", []any{1.0, 1.5, nil, 2.5, 3.0})
	xeqStr(t, xdump(s.PctChange()), []string{"NA", "0.5", "NA", "NA", "0.2"})

	// test_pct_change_no_warning_na_beginning: leading NAs, then a clean run.
	s2 := NewSeries("x", []any{nil, nil, 1.0, 2.0, 3.0})
	xeqStr(t, xdump(s2.PctChange()), []string{"NA", "NA", "NA", "1", "0.5"})
}

// TestParityValueCounts mirrors
// pandas/tests/series/methods/test_value_counts.py, where
// Series(["a", "b", "c", "c", "c", "b"]).value_counts() gives counts
// [3, 2, 1] indexed by ["c", "b", "a"] with the result named "count".
func TestParityValueCounts(t *testing.T) {
	s := NewSeries("xxx", []any{"a", "b", "c", "c", "c", "b"})
	vc := s.ValueCounts()
	if vc.Name() != "count" {
		t.Fatalf("value_counts name: got %q want %q", vc.Name(), "count")
	}
	xeqStr(t, xdump(vc), []string{"3", "2", "1"})
	xeqStr(t, xidx(vc), []string{"c", "b", "a"})
}

// TestParityCumulative mirrors the numpy accumulate semantics asserted in
// pandas/tests/series/test_cumulative.py (cumsum/cumprod/cummin/cummax).
func TestParityCumulative(t *testing.T) {
	s := NewSeries("x", []any{2.0, 1, 3, 5, 4})
	xeqStr(t, xdump(s.CumSum()), []string{"2", "3", "6", "11", "15"})
	xeqStr(t, xdump(s.CumProd()), []string{"2", "2", "6", "30", "120"})
	xeqStr(t, xdump(s.CumMax()), []string{"2", "2", "3", "5", "5"})
	xeqStr(t, xdump(s.CumMin()), []string{"2", "1", "1", "1", "1"})
}

// TestParityQuantile mirrors pandas/tests/series/methods/test_quantile.py,
// which asserts Series.quantile equals numpy.percentile with the default
// linear interpolation.
func TestParityQuantile(t *testing.T) {
	s := NewSeries("x", []any{1.0, 2, 3, 4})
	for _, c := range []struct {
		q    float64
		want float64
	}{
		{0.0, 1.0},
		{0.25, 1.75},
		{0.5, 2.5},
		{0.75, 3.25},
		{1.0, 4.0},
	} {
		got, ok := s.Quantile(c.q)
		if !ok || !xclose(got, c.want) {
			t.Fatalf("quantile(%v): got %v (ok=%v) want %v", c.q, got, ok, c.want)
		}
	}
	// Median is the 0.5 quantile.
	m, ok := s.Median()
	if !ok || !xclose(m, 2.5) {
		t.Fatalf("median: got %v want 2.5", m)
	}
}

// TestParityStdVar mirrors the ddof=1 (sample) default used by
// pandas Series.std / Series.var, checked against the closed-form values for
// [1, 2, 3, 4]: variance = 5/3, std = sqrt(5/3).
func TestParityStdVar(t *testing.T) {
	s := NewSeries("x", []any{1.0, 2, 3, 4})
	v, ok := s.Var()
	if !ok || !xclose(v, 5.0/3.0) {
		t.Fatalf("var: got %v want %v", v, 5.0/3.0)
	}
	sd, ok := s.Std()
	if !ok || !xclose(sd*sd, 5.0/3.0) {
		t.Fatalf("std: got %v want sqrt(5/3)", sd)
	}
}

// TestParityRoundHalfEven mirrors numpy/pandas round-half-to-even, the rule
// pandas Series.round and DataFrame.round apply. numpy: round(2.5) == 2,
// round(-1.25, 1) == -1.2, round(2.75, 1) == 2.8.
func TestParityRoundHalfEven(t *testing.T) {
	s := NewSeries("x", []any{1.234, 2.5, 2.678})
	xeqStr(t, xdump(s.Round(0)), []string{"1", "2", "3"})

	s2 := NewSeries("x", []any{-1.25, 2.75, 0.5, 1.5})
	xeqStr(t, xdump(s2.Round(1)), []string{"-1.2", "2.8", "0.5", "1.5"})
	xeqStr(t, xdump(s2.Round(0)), []string{"-1", "3", "0", "2"})
}

// TestParityMode mirrors pandas Series.mode, which returns every value tying
// for the highest frequency in ascending order.
func TestParityMode(t *testing.T) {
	s := NewSeries("x", []any{1, 2, 2, 3, 3})
	got := s.Mode()
	if len(got) != 2 {
		t.Fatalf("mode length: got %d want 2 (%v)", len(got), got)
	}
	if a, _ := toInt64(got[0]); a != 2 {
		t.Fatalf("mode[0]: got %v want 2", got[0])
	}
	if b, _ := toInt64(got[1]); b != 3 {
		t.Fatalf("mode[1]: got %v want 3", got[1])
	}
}

// TestParityNLargestNSmallest mirrors
// pandas/tests/series/methods/test_nlargest.py, whose nlargest/nsmallest
// return the n extreme values in sorted order.
func TestParityNLargestNSmallest(t *testing.T) {
	s := NewSeries("x", []any{5.0, 1, 4, 2, 3})
	xeqStr(t, xdump(s.NLargest(3)), []string{"5", "4", "3"})
	xeqStr(t, xdump(s.NSmallest(3)), []string{"1", "2", "3"})
}
