package pandas

import "testing"

func gbDF(t *testing.T) *DataFrame {
	t.Helper()
	df, err := FromMap(map[string][]any{
		"team":  {"A", "B", "A", "B", "A"},
		"score": {10.0, 5.0, 20.0, 15.0, nil},
		"wins":  {int64(1), int64(0), int64(1), int64(1), int64(0)},
	}, []string{"team", "score", "wins"})
	if err != nil {
		t.Fatal(err)
	}
	return df
}

func TestGroupBySum(t *testing.T) {
	df := gbDF(t)
	gb, err := df.GroupBy("team")
	if err != nil {
		t.Fatal(err)
	}
	if gb.Groups() != 2 {
		t.Fatalf("groups = %d", gb.Groups())
	}
	res, err := gb.Sum("score")
	if err != nil {
		t.Fatal(err)
	}
	// Deterministic order: A then B.
	if v, _ := res.MustCol("team").At(0); v.(string) != "A" {
		t.Fatalf("group order = %v", v)
	}
	if v, _ := res.MustCol("score_sum").At(0); v.(float64) != 30 {
		t.Fatalf("A sum = %v", v)
	}
	if v, _ := res.MustCol("score_sum").At(1); v.(float64) != 20 {
		t.Fatalf("B sum = %v", v)
	}
}

func TestGroupByAllAggs(t *testing.T) {
	df := gbDF(t)
	gb, _ := df.GroupBy("team")

	mean, _ := gb.Mean("score")
	if v, _ := mean.MustCol("score_mean").At(0); v.(float64) != 15 {
		t.Fatalf("A mean = %v", v)
	}
	mn, _ := gb.Min("score")
	if v, _ := mn.MustCol("score_min").At(0); v.(float64) != 10 {
		t.Fatalf("A min = %v", v)
	}
	mx, _ := gb.Max("score")
	if v, _ := mx.MustCol("score_max").At(0); v.(float64) != 20 {
		t.Fatalf("A max = %v", v)
	}
	std, _ := gb.Std("score")
	if _, ok := std.MustCol("score_std").At(0); !ok {
		t.Fatal("A std should exist")
	}
	cnt, _ := gb.Count("score")
	// A has score present twice (one NA excluded).
	if v, _ := cnt.MustCol("score_count").At(0); v.(int64) != 2 {
		t.Fatalf("A count = %v", v)
	}
}

func TestGroupByCountDefault(t *testing.T) {
	df := gbDF(t)
	gb, _ := df.GroupBy("team")
	cnt, err := gb.Count()
	if err != nil {
		t.Fatal(err)
	}
	// Should count all non-key columns.
	if !cnt.HasColumn("score_count") || !cnt.HasColumn("wins_count") {
		t.Fatalf("count cols = %v", cnt.Names())
	}
}

func TestGroupByMultiKeyAndAgg(t *testing.T) {
	df, _ := FromMap(map[string][]any{
		"a": {"x", "x", "y"},
		"b": {int64(1), int64(2), int64(1)},
		"v": {10.0, 20.0, 30.0},
	}, []string{"a", "b", "v"})
	gb, _ := df.GroupBy("a", "b")
	if gb.Groups() != 3 {
		t.Fatalf("groups = %d", gb.Groups())
	}
	res, err := gb.Agg(map[string][]AggFunc{
		"v": {AggSum, AggMean},
	}, []string{"v"})
	if err != nil {
		t.Fatal(err)
	}
	if !res.HasColumn("v_sum") || !res.HasColumn("v_mean") {
		t.Fatalf("agg cols = %v", res.Names())
	}
}

func TestGroupByErrors(t *testing.T) {
	df := gbDF(t)
	if _, err := df.GroupBy("nope"); err == nil {
		t.Fatal("groupby missing key")
	}
	gb, _ := df.GroupBy("team")
	if _, err := gb.Sum("nope"); err == nil {
		t.Fatal("agg missing col")
	}
	if _, err := gb.Agg(map[string][]AggFunc{"nope": {AggSum}}, nil); err == nil {
		t.Fatal("agg missing col via spec")
	}
}

func TestGroupByNAKey(t *testing.T) {
	df, _ := FromMap(map[string][]any{
		"k": {"a", nil, "a", nil},
		"v": {1.0, 2.0, 3.0, 4.0},
	}, []string{"k", "v"})
	gb, _ := df.GroupBy("k")
	if gb.Groups() != 2 {
		t.Fatalf("groups with NA key = %d", gb.Groups())
	}
	res, _ := gb.Sum("v")
	// NA group sorts last.
	if _, ok := res.MustCol("k").At(1); ok {
		t.Fatal("NA key group should sort last")
	}
}

func TestAggFuncSuffix(t *testing.T) {
	cases := map[AggFunc]string{
		AggSum: "sum", AggMean: "mean", AggMin: "min",
		AggMax: "max", AggCount: "count", AggStd: "std",
	}
	for f, want := range cases {
		if f.suffix() != want {
			t.Fatalf("suffix(%d) = %q, want %q", f, f.suffix(), want)
		}
	}
}
