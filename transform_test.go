package pandas

import "testing"

func TestAbs(t *testing.T) {
	s := NewSeries("x", []any{-1, 2, -3, nil})
	got := s.Abs()
	if got.DType() != Int64 {
		t.Fatalf("dtype: got %v want int64", got.DType())
	}
	xeqStr(t, xdump(got), []string{"1", "2", "3", "NA"})

	f := NewSeries("x", []any{-1.5, 2.5, nil})
	xeqStr(t, xdump(f.Abs()), []string{"1.5", "2.5", "NA"})
}

func TestRound(t *testing.T) {
	s := NewSeries("x", []any{1.234, 2.5, 2.678, nil})
	xeqStr(t, xdump(s.Round(1)), []string{"1.2", "2.5", "2.7", "NA"})
	// numpy/pandas use round-half-to-even, so 2.5 rounds to the even 2.
	xeqStr(t, xdump(s.Round(0)), []string{"1", "2", "3", "NA"})
}

func TestClip(t *testing.T) {
	s := NewSeries("x", []any{1.0, 5.0, 10.0, nil})
	xeqStr(t, xdump(s.Clip(2, 8)), []string{"2", "5", "8", "NA"})
}

func TestCumSum(t *testing.T) {
	s := NewSeries("x", []any{1.0, 2.0, nil, 4.0})
	xeqStr(t, xdump(s.CumSum()), []string{"1", "3", "NA", "7"})
}

func TestCumProd(t *testing.T) {
	s := NewSeries("x", []any{1.0, 2.0, 3.0, nil, 2.0})
	xeqStr(t, xdump(s.CumProd()), []string{"1", "2", "6", "NA", "12"})
}

func TestCumMaxMin(t *testing.T) {
	s := NewSeries("x", []any{1.0, 3.0, 2.0, 5.0, 4.0})
	xeqStr(t, xdump(s.CumMax()), []string{"1", "3", "3", "5", "5"})
	s2 := NewSeries("x", []any{3.0, 1.0, 2.0, 0.0})
	xeqStr(t, xdump(s2.CumMin()), []string{"3", "1", "1", "0"})
}

func TestShift(t *testing.T) {
	s := NewSeries("x", []any{1, 2, 3})
	xeqStr(t, xdump(s.Shift(1)), []string{"NA", "1", "2"})
	xeqStr(t, xdump(s.Shift(-1)), []string{"2", "3", "NA"})
	xeqStr(t, xdump(s.Shift(0)), []string{"1", "2", "3"})
}

func TestDiff(t *testing.T) {
	s := NewSeries("x", []any{1.0, 3.0, 6.0, 10.0})
	xeqStr(t, xdump(s.Diff()), []string{"NA", "2", "3", "4"})
}

func TestPctChange(t *testing.T) {
	s := NewSeries("x", []any{100.0, 110.0, 99.0})
	got := s.PctChange()
	vals := got.Values()
	if vals[0] != nil {
		t.Fatalf("first should be NA, got %v", vals[0])
	}
	if !xclose(vals[1].(float64), 0.1) || !xclose(vals[2].(float64), -0.1) {
		t.Fatalf("pct change: got %v", vals)
	}
}

func TestAstype(t *testing.T) {
	s := NewSeries("x", []any{"1", "2", "x"})
	got := s.Astype(Int64)
	if got.DType() != Int64 {
		t.Fatalf("dtype: %v", got.DType())
	}
	xeqStr(t, xdump(got), []string{"1", "2", "NA"})
}

func TestBetween(t *testing.T) {
	s := NewSeries("x", []any{1.0, 2.0, 3.0, 4.0, nil})
	xeqBool(t, s.Between(2, 3), []bool{false, true, true, false, false})
}

func TestIsIn(t *testing.T) {
	s := NewSeries("x", []any{1, 2, 3, 4})
	xeqBool(t, s.IsIn([]any{1, 3}), []bool{true, false, true, false})
}

func TestArithmetic(t *testing.T) {
	a := NewSeries("a", []any{1.0, 2.0, 3.0, nil})
	b := NewSeries("b", []any{10.0, 20.0, 0.0, 5.0})
	xeqStr(t, xdump(a.Add(b)), []string{"11", "22", "3", "NA"})
	xeqStr(t, xdump(a.Sub(b)), []string{"-9", "-18", "3", "NA"})
	xeqStr(t, xdump(a.Mul(b)), []string{"10", "40", "0", "NA"})
	// Div by zero -> NA at position 2.
	xeqStr(t, xdump(a.Div(b)), []string{"0.1", "0.1", "NA", "NA"})
}

func TestArithmeticUnequalLength(t *testing.T) {
	a := NewSeries("a", []any{1.0, 2.0, 3.0})
	b := NewSeries("b", []any{10.0, 20.0})
	got := a.Add(b)
	if got.Len() != 2 {
		t.Fatalf("len: got %d want 2", got.Len())
	}
	xeqStr(t, xdump(got), []string{"11", "22"})
}

func BenchmarkCumSum(b *testing.B) {
	vals := make([]any, 1000)
	for i := range vals {
		vals[i] = float64(i)
	}
	s := NewSeries("x", vals)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = s.CumSum()
	}
}
