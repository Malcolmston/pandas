package pandas

import (
	"math"
	"reflect"
	"testing"
)

func floatEq(a, b float64) bool { return math.Abs(a-b) < 1e-9 }

func TestNewSeriesInferAndNA(t *testing.T) {
	s := NewSeries("x", []any{1.5, nil, 2.5, "bad"})
	if s.DType() != Float64 {
		t.Fatalf("dtype = %v, want Float64", s.DType())
	}
	if s.Len() != 4 {
		t.Fatalf("len = %d", s.Len())
	}
	if v, ok := s.At(0); !ok || v.(float64) != 1.5 {
		t.Fatalf("At(0) = %v,%v", v, ok)
	}
	if _, ok := s.At(1); ok {
		t.Fatalf("At(1) should be NA")
	}
	if _, ok := s.At(3); ok {
		t.Fatalf("At(3) 'bad' should coerce to NA")
	}
	if _, ok := s.At(99); ok {
		t.Fatalf("out of range should be false")
	}
	na := s.IsNA()
	if !reflect.DeepEqual(na, []bool{false, true, false, true}) {
		t.Fatalf("IsNA = %v", na)
	}
}

func TestSeriesAllNilObject(t *testing.T) {
	s := NewSeries("e", []any{nil, nil})
	if s.DType() != Object {
		t.Fatalf("dtype = %v, want Object", s.DType())
	}
}

func TestSeriesReductions(t *testing.T) {
	s := NewSeries("v", []any{2.0, 4.0, nil, 6.0})
	if sum, ok := s.Sum(); !ok || !floatEq(sum, 12) {
		t.Fatalf("sum = %v,%v", sum, ok)
	}
	if m, ok := s.Mean(); !ok || !floatEq(m, 4) {
		t.Fatalf("mean = %v,%v", m, ok)
	}
	if mn, ok := s.Min(); !ok || !floatEq(mn, 2) {
		t.Fatalf("min = %v", mn)
	}
	if mx, ok := s.Max(); !ok || !floatEq(mx, 6) {
		t.Fatalf("max = %v", mx)
	}
	if c := s.Count(); c != 3 {
		t.Fatalf("count = %d", c)
	}
	std, ok := s.Std()
	if !ok || !floatEq(std, 2) {
		t.Fatalf("std = %v,%v", std, ok)
	}
}

func TestSeriesReductionsEmpty(t *testing.T) {
	s := NewSeriesTyped("v", Float64, []any{nil})
	if _, ok := s.Sum(); ok {
		t.Fatal("sum of all-NA should be false")
	}
	if _, ok := s.Mean(); ok {
		t.Fatal("mean should be false")
	}
	if _, ok := s.Min(); ok {
		t.Fatal("min should be false")
	}
	if _, ok := s.Max(); ok {
		t.Fatal("max should be false")
	}
	if _, ok := s.Std(); ok {
		t.Fatal("std should be false")
	}
}

func TestSeriesStdNeedsTwo(t *testing.T) {
	s := NewSeries("v", []any{5.0})
	if _, ok := s.Std(); ok {
		t.Fatal("std of single value should be false")
	}
}

func TestSeriesApplyMap(t *testing.T) {
	s := NewSeries("v", []any{1.0, nil, 3.0})
	doubled := s.Apply(func(v any) any { return v.(float64) * 2 })
	if v, _ := doubled.At(0); v.(float64) != 2 {
		t.Fatalf("apply = %v", v)
	}
	if _, ok := doubled.At(1); ok {
		t.Fatal("NA should pass through apply")
	}
	mapped := s.Map(func(v any) any { return v.(float64) + 1 })
	if v, _ := mapped.At(2); v.(float64) != 4 {
		t.Fatalf("map = %v", v)
	}
}

func TestSeriesFillDropNA(t *testing.T) {
	s := NewSeries("v", []any{1.0, nil, 3.0})
	filled := s.FillNA(0.0)
	if v, ok := filled.At(1); !ok || v.(float64) != 0 {
		t.Fatalf("fillna = %v,%v", v, ok)
	}
	// Original unchanged.
	if _, ok := s.At(1); ok {
		t.Fatal("original mutated")
	}
	dropped := s.DropNA()
	if dropped.Len() != 2 {
		t.Fatalf("dropna len = %d", dropped.Len())
	}
	// FillNA with an uncoercible value returns a copy unchanged.
	bad := s.FillNA("nope")
	if _, ok := bad.At(1); ok {
		t.Fatal("fillna with bad value should leave NA")
	}
}

func TestSeriesHeadTailFilter(t *testing.T) {
	s := NewSeries("v", []any{1.0, 2.0, 3.0, 4.0})
	if s.Head(2).Len() != 2 {
		t.Fatal("head")
	}
	if s.Head(99).Len() != 4 {
		t.Fatal("head clamp")
	}
	tail := s.Tail(2)
	if v, _ := tail.At(0); v.(float64) != 3 {
		t.Fatalf("tail = %v", v)
	}
	if s.Tail(99).Len() != 4 {
		t.Fatal("tail clamp")
	}
	f := s.Filter([]bool{true, false, true, false})
	if f.Len() != 2 {
		t.Fatalf("filter len = %d", f.Len())
	}
	if v, _ := f.At(1); v.(float64) != 3 {
		t.Fatalf("filter = %v", v)
	}
}

func TestSeriesUniqueValueCounts(t *testing.T) {
	s := NewSeries("v", []any{"b", "a", "b", "a", "a", nil})
	u := s.Unique()
	if !reflect.DeepEqual(u, []any{"b", "a"}) {
		t.Fatalf("unique = %v", u)
	}
	vc := s.ValueCounts()
	// a:3, b:2 -> sorted by count desc.
	if v, _ := vc.At(0); v.(int64) != 3 {
		t.Fatalf("valuecounts[0] = %v", v)
	}
	if vc.index[0] != "a" {
		t.Fatalf("valuecounts idx[0] = %v", vc.index[0])
	}
	if vc.index[1] != "b" {
		t.Fatalf("valuecounts idx[1] = %v", vc.index[1])
	}
}

func TestSeriesValueCountsTieBreak(t *testing.T) {
	s := NewSeries("v", []any{"y", "x", "y", "x"})
	vc := s.ValueCounts()
	// Equal counts -> ascending value order: x before y.
	if vc.index[0] != "x" || vc.index[1] != "y" {
		t.Fatalf("tie order = %v", vc.index)
	}
}

func TestSeriesSort(t *testing.T) {
	s := NewSeries("v", []any{3.0, 1.0, nil, 2.0})
	asc := s.Sort(true)
	got := []float64{}
	for i := 0; i < asc.Len(); i++ {
		if v, ok := asc.At(i); ok {
			got = append(got, v.(float64))
		}
	}
	if !reflect.DeepEqual(got, []float64{1, 2, 3}) {
		t.Fatalf("sort asc = %v", got)
	}
	// NA should be last.
	if _, ok := asc.At(3); ok {
		t.Fatal("NA should sort last")
	}
	desc := s.Sort(false)
	if v, _ := desc.At(0); v.(float64) != 3 {
		t.Fatalf("sort desc[0] = %v", v)
	}
}

func TestSeriesRenameCopyValues(t *testing.T) {
	s := NewSeries("v", []any{1.0, nil})
	r := s.Rename("w")
	if r.Name() != "w" || s.Name() != "v" {
		t.Fatal("rename")
	}
	vals := s.Values()
	if vals[0].(float64) != 1 || vals[1] != nil {
		t.Fatalf("values = %v", vals)
	}
	idx := s.Index()
	if idx[0].(int64) != 0 {
		t.Fatalf("index = %v", idx)
	}
}

func TestSeriesStringRender(t *testing.T) {
	s := NewSeries("v", []any{1.0, nil})
	out := s.String()
	if out == "" {
		t.Fatal("empty string render")
	}
}

func TestIntStringBoolSeries(t *testing.T) {
	i := NewSeries("i", []any{int64(1), 2, nil})
	if i.DType() != Int64 {
		t.Fatalf("int dtype = %v", i.DType())
	}
	if v, _ := i.At(1); v.(int64) != 2 {
		t.Fatalf("int coerce = %v", v)
	}
	b := NewSeries("b", []any{true, false, nil})
	if b.DType() != Bool {
		t.Fatalf("bool dtype = %v", b.DType())
	}
	str := NewSeries("s", []any{"a", 3, nil})
	if str.DType() != String {
		t.Fatalf("string dtype = %v", str.DType())
	}
	if v, _ := str.At(1); v.(string) != "3" {
		t.Fatalf("string coerce = %v", v)
	}
}

func TestLessFallback(t *testing.T) {
	if !less(false, true) {
		t.Fatal("bool less")
	}
	if less(true, false) {
		t.Fatal("bool less reverse")
	}
	// Cross-type fallback uses string formatting.
	if !less(int64(2), "3") {
		t.Fatal("cross-type less fallback")
	}
}
