package pandas

import "testing"

// xseriesMap maps a reduction series' string index labels to their float values.
func xseriesMap(s *Series) map[string]float64 {
	m := make(map[string]float64)
	for i := range s.index {
		if s.valid[i] {
			f, _ := toFloat64(s.data[i])
			m[formatValue(s.index[i])] = f
		}
	}
	return m
}

func newTestFrame(t *testing.T) *DataFrame {
	t.Helper()
	df, err := FromMap(map[string][]any{
		"a": {1.0, 2.0, 3.0, 4.0},
		"b": {10.0, 20.0, 30.0, 40.0},
		"g": {"x", "y", "x", "y"},
	}, []string{"a", "b", "g"})
	if err != nil {
		t.Fatal(err)
	}
	return df
}

func TestFrameReductions(t *testing.T) {
	df := newTestFrame(t)

	sum := xseriesMap(df.Sum())
	if !xclose(sum["a"], 10) || !xclose(sum["b"], 100) {
		t.Fatalf("sum: %v", sum)
	}
	if _, ok := sum["g"]; ok {
		t.Fatal("string column should be excluded from Sum")
	}
	mean := xseriesMap(df.Mean())
	if !xclose(mean["a"], 2.5) || !xclose(mean["b"], 25) {
		t.Fatalf("mean: %v", mean)
	}
	mn := xseriesMap(df.Min())
	mx := xseriesMap(df.Max())
	if !xclose(mn["a"], 1) || !xclose(mx["b"], 40) {
		t.Fatalf("min/max: %v %v", mn, mx)
	}
	md := xseriesMap(df.Median())
	if !xclose(md["a"], 2.5) {
		t.Fatalf("median: %v", md)
	}
	vr := xseriesMap(df.Var())
	if !xclose(vr["a"], 5.0/3.0) {
		t.Fatalf("var: %v", vr)
	}
	sd := xseriesMap(df.Std())
	if !xclose(sd["a"]*sd["a"], 5.0/3.0) {
		t.Fatalf("std: %v", sd)
	}
}

func TestNunique(t *testing.T) {
	df := newTestFrame(t)
	nu := xseriesMap(df.Nunique())
	if nu["a"] != 4 || nu["g"] != 2 {
		t.Fatalf("nunique: %v", nu)
	}
}

func TestFrameAbsRound(t *testing.T) {
	df, _ := FromMap(map[string][]any{
		"a": {-1.25, 2.75},
		"g": {"p", "q"},
	}, []string{"a", "g"})
	ab := df.Abs()
	col, _ := ab.Col("a")
	xeqStr(t, xdump(col), []string{"1.25", "2.75"})
	rd := df.Round(1)
	col, _ = rd.Col("a")
	xeqStr(t, xdump(col), []string{"-1.3", "2.8"})
	// String column untouched.
	gcol, _ := rd.Col("g")
	xeqStr(t, xdump(gcol), []string{"p", "q"})
}

func TestDropDuplicates(t *testing.T) {
	df, _ := FromMap(map[string][]any{
		"a": {1, 1, 2, 1},
		"b": {"x", "x", "y", "x"},
	}, []string{"a", "b"})
	got := df.DropDuplicates()
	if got.NumRows() != 2 {
		t.Fatalf("rows: got %d want 2", got.NumRows())
	}
	ca, _ := got.Col("a")
	xeqStr(t, xdump(ca), []string{"1", "2"})
}

func TestSetResetIndex(t *testing.T) {
	df, _ := FromMap(map[string][]any{
		"id":  {"a", "b", "c"},
		"val": {1.0, 2.0, 3.0},
	}, []string{"id", "val"})
	si, err := df.SetIndex("id")
	if err != nil {
		t.Fatal(err)
	}
	if si.HasColumn("id") {
		t.Fatal("id column should be removed after SetIndex")
	}
	if formatValue(si.Index()[1]) != "b" {
		t.Fatalf("index: %v", si.Index())
	}
	ri := si.ResetIndex()
	if !ri.HasColumn("index") {
		t.Fatal("ResetIndex should add 'index' column")
	}
	ic, _ := ri.Col("index")
	xeqStr(t, xdump(ic), []string{"a", "b", "c"})
	if formatValue(ri.Index()[0]) != "0" {
		t.Fatalf("reset index labels: %v", ri.Index())
	}
}

func TestFrameCorr(t *testing.T) {
	df, _ := FromMap(map[string][]any{
		"a": {1.0, 2.0, 3.0},
		"b": {2.0, 4.0, 6.0},
		"c": {3.0, 2.0, 1.0},
	}, []string{"a", "b", "c"})
	cm := df.Corr()
	rows, cols := cm.Shape()
	if rows != 3 || cols != 3 {
		t.Fatalf("shape: %d x %d", rows, cols)
	}
	acol, _ := cm.Col("a")
	// a vs a = 1, a vs b = 1, a vs c = -1
	if !xclose(acol.data[0].(float64), 1) || !xclose(acol.data[1].(float64), 1) || !xclose(acol.data[2].(float64), -1) {
		t.Fatalf("corr a column: %v", xdump(acol))
	}
}

func TestConcat(t *testing.T) {
	df1, _ := FromMap(map[string][]any{
		"a": {1.0, 2.0},
		"b": {"x", "y"},
	}, []string{"a", "b"})
	df2, _ := FromMap(map[string][]any{
		"a": {3.0},
		"c": {true},
	}, []string{"a", "c"})
	got, err := Concat(df1, df2)
	if err != nil {
		t.Fatal(err)
	}
	if got.NumRows() != 3 {
		t.Fatalf("rows: %d", got.NumRows())
	}
	xeqStr(t, got.Names(), []string{"a", "b", "c"})
	ca, _ := got.Col("a")
	xeqStr(t, xdump(ca), []string{"1", "2", "3"})
	cb, _ := got.Col("b")
	xeqStr(t, xdump(cb), []string{"x", "y", "NA"})
	cc, _ := got.Col("c")
	xeqStr(t, xdump(cc), []string{"NA", "NA", "true"})
}

func TestTranspose(t *testing.T) {
	df, _ := FromMap(map[string][]any{
		"a": {1.0, 2.0},
		"b": {3.0, 4.0},
	}, []string{"a", "b"})
	tr, err := df.Transpose()
	if err != nil {
		t.Fatal(err)
	}
	// Original index 0,1 -> columns "0","1"; original names -> index.
	xeqStr(t, tr.Names(), []string{"0", "1"})
	if formatValue(tr.Index()[0]) != "a" || formatValue(tr.Index()[1]) != "b" {
		t.Fatalf("transpose index: %v", tr.Index())
	}
	c0, _ := tr.Col("0")
	xeqStr(t, xdump(c0), []string{"1", "3"})
	c1, _ := tr.Col("1")
	xeqStr(t, xdump(c1), []string{"2", "4"})
}

func BenchmarkConcat(b *testing.B) {
	mk := func() *DataFrame {
		vals := make([]any, 500)
		for i := range vals {
			vals[i] = float64(i)
		}
		df, _ := FromMap(map[string][]any{"a": vals, "b": vals}, []string{"a", "b"})
		return df
	}
	d1, d2 := mk(), mk()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = Concat(d1, d2)
	}
}
