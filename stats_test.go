package pandas

import "testing"

func TestVar(t *testing.T) {
	s := NewSeries("x", []any{1.0, 2.0, 3.0, 4.0})
	v, ok := s.Var()
	if !ok || !xclose(v, 5.0/3.0) {
		t.Fatalf("var: got %v %v want 1.6667", v, ok)
	}
	if _, ok := NewSeries("x", []any{1.0}).Var(); ok {
		t.Fatal("var of single value should be false")
	}
}

func TestProd(t *testing.T) {
	s := NewSeries("x", []any{1.0, 2.0, 3.0, 4.0})
	p, ok := s.Prod()
	if !ok || !xclose(p, 24) {
		t.Fatalf("prod: got %v", p)
	}
}

func TestQuantileMedian(t *testing.T) {
	s := NewSeries("x", []any{1.0, 2.0, 3.0, 4.0})
	q, _ := s.Quantile(0.5)
	if !xclose(q, 2.5) {
		t.Fatalf("q0.5: %v", q)
	}
	q, _ = s.Quantile(0.25)
	if !xclose(q, 1.75) {
		t.Fatalf("q0.25: %v", q)
	}
	m, _ := s.Median()
	if !xclose(m, 2.5) {
		t.Fatalf("median: %v", m)
	}
	if _, ok := s.Quantile(1.5); ok {
		t.Fatal("out of range quantile should fail")
	}
}

func TestMode(t *testing.T) {
	s := NewSeries("x", []any{1, 2, 2, 3, 3})
	got := s.Mode()
	if len(got) != 2 || got[0] != int64(2) || got[1] != int64(3) {
		t.Fatalf("mode: got %v", got)
	}
	single := NewSeries("x", []any{5, 5, 1})
	if m := single.Mode(); len(m) != 1 || m[0] != int64(5) {
		t.Fatalf("mode single: %v", m)
	}
}

func TestArgMaxMin(t *testing.T) {
	s := NewSeries("x", []any{1.0, 5.0, 3.0, 5.0})
	if i, ok := s.ArgMax(); !ok || i != 1 {
		t.Fatalf("argmax: %d", i)
	}
	if i, ok := s.ArgMin(); !ok || i != 0 {
		t.Fatalf("argmin: %d", i)
	}
}

func TestCovCorr(t *testing.T) {
	x := NewSeries("x", []any{1.0, 2.0, 3.0})
	y := NewSeries("y", []any{2.0, 4.0, 6.0})
	c, ok := x.Corr(y)
	if !ok || !xclose(c, 1.0) {
		t.Fatalf("corr: %v", c)
	}
	cv, ok := x.Cov(y)
	if !ok || !xclose(cv, 2.0) {
		t.Fatalf("cov: %v", cv)
	}
	// Anti-correlated.
	z := NewSeries("z", []any{6.0, 4.0, 2.0})
	c, _ = x.Corr(z)
	if !xclose(c, -1.0) {
		t.Fatalf("neg corr: %v", c)
	}
	// Zero variance -> false.
	flat := NewSeries("f", []any{1.0, 1.0, 1.0})
	if _, ok := x.Corr(flat); ok {
		t.Fatal("corr with zero-variance should be false")
	}
}

func TestRank(t *testing.T) {
	s := NewSeries("x", []any{10.0, 20.0, 20.0, 40.0})
	got := s.Rank()
	xeqStr(t, xdump(got), []string{"1", "2.5", "2.5", "4"})
}

func TestNLargestNSmallest(t *testing.T) {
	s := NewSeries("x", []any{3.0, 1.0, 4.0, 1.0, 5.0})
	xeqStr(t, xdump(s.NLargest(2)), []string{"5", "4"})
	xeqStr(t, xdump(s.NSmallest(2)), []string{"1", "1"})
	// NLargest preserves index labels.
	big := s.NLargest(1)
	if got := xidx(big); got[0] != "4" {
		t.Fatalf("nlargest index: got %v want position 4", got)
	}
	// n exceeding length returns all.
	if s.NLargest(99).Len() != 5 {
		t.Fatal("nlargest over-length should return all")
	}
}

func BenchmarkRank(b *testing.B) {
	vals := make([]any, 1000)
	for i := range vals {
		vals[i] = float64((i * 7) % 100)
	}
	s := NewSeries("x", vals)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = s.Rank()
	}
}

func BenchmarkCorr(b *testing.B) {
	xv := make([]any, 1000)
	yv := make([]any, 1000)
	for i := range xv {
		xv[i] = float64(i)
		yv[i] = float64(i * 2)
	}
	x := NewSeries("x", xv)
	y := NewSeries("y", yv)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = x.Corr(y)
	}
}
