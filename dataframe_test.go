package pandas

import (
	"reflect"
	"testing"
)

func sampleDF(t *testing.T) *DataFrame {
	t.Helper()
	df, err := FromMap(map[string][]any{
		"city":  {"NYC", "LA", "NYC", "SF"},
		"temp":  {31.0, 28.0, 33.0, 20.0},
		"sunny": {true, true, false, nil},
	}, []string{"city", "temp", "sunny"})
	if err != nil {
		t.Fatal(err)
	}
	return df
}

func TestNewDataFrameErrors(t *testing.T) {
	a := NewSeries("a", []any{1.0, 2.0})
	b := NewSeries("b", []any{1.0})
	if _, err := NewDataFrame(a, b); err == nil {
		t.Fatal("expected length mismatch error")
	}
	a2 := NewSeries("a", []any{3.0, 4.0})
	if _, err := NewDataFrame(a, a2); err == nil {
		t.Fatal("expected duplicate name error")
	}
	if df, err := NewDataFrame(); err != nil || df.NumRows() != 0 {
		t.Fatal("empty frame")
	}
}

func TestFromMapOrdering(t *testing.T) {
	df, _ := FromMap(map[string][]any{
		"z": {1.0}, "a": {2.0}, "m": {3.0},
	}, []string{"m"})
	// m first (explicit), then a, z sorted.
	if !reflect.DeepEqual(df.Names(), []string{"m", "a", "z"}) {
		t.Fatalf("names = %v", df.Names())
	}
}

type rec struct {
	Name   string `pandas:"name"`
	Age    int64
	Secret string `pandas:"-"`
	hidden int
}

func TestFromRecords(t *testing.T) {
	df, err := FromRecords([]rec{{"a", 30, "x", 1}, {"b", 25, "y", 2}})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(df.Names(), []string{"name", "Age"}) {
		t.Fatalf("names = %v", df.Names())
	}
	if df.NumRows() != 2 {
		t.Fatalf("rows = %d", df.NumRows())
	}
	// Pointer slice too.
	df2, err := FromRecords([]*rec{{Name: "c", Age: 40}})
	if err != nil || df2.NumRows() != 1 {
		t.Fatalf("ptr records: %v", err)
	}
	if _, err := FromRecords(42); err == nil {
		t.Fatal("expected non-slice error")
	}
	if _, err := FromRecords([]int{1, 2}); err == nil {
		t.Fatal("expected non-struct error")
	}
}

func TestShapeAndAccess(t *testing.T) {
	df := sampleDF(t)
	r, c := df.Shape()
	if r != 4 || c != 3 {
		t.Fatalf("shape = %d,%d", r, c)
	}
	if !df.HasColumn("temp") || df.HasColumn("nope") {
		t.Fatal("HasColumn")
	}
	if _, ok := df.Col("nope"); ok {
		t.Fatal("Col missing")
	}
	if df.MustCol("temp").Name() != "temp" {
		t.Fatal("MustCol")
	}
	func() {
		defer func() { _ = recover() }()
		df.MustCol("nope")
		t.Fatal("MustCol should panic")
	}()
	if len(df.Index()) != 4 {
		t.Fatal("Index")
	}
}

func TestSelectDropRename(t *testing.T) {
	df := sampleDF(t)
	sel, err := df.Select("temp", "city")
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(sel.Names(), []string{"temp", "city"}) {
		t.Fatalf("select = %v", sel.Names())
	}
	if _, err := df.Select("nope"); err == nil {
		t.Fatal("select missing")
	}
	dropped := df.Drop("sunny", "ghost")
	if dropped.HasColumn("sunny") || dropped.NumCols() != 2 {
		t.Fatalf("drop = %v", dropped.Names())
	}
	renamed := df.Rename(map[string]string{"temp": "temperature"})
	if !renamed.HasColumn("temperature") || renamed.HasColumn("temp") {
		t.Fatal("rename")
	}
	if df.HasColumn("temperature") {
		t.Fatal("rename mutated original")
	}
}

func TestWithColumn(t *testing.T) {
	df := sampleDF(t)
	nc := NewSeries("flag", []any{1.0, 2.0, 3.0, 4.0})
	out, err := df.WithColumn(nc)
	if err != nil {
		t.Fatal(err)
	}
	if !out.HasColumn("flag") {
		t.Fatal("with column")
	}
	// Replace existing.
	repl := NewSeries("temp", []any{0.0, 0.0, 0.0, 0.0})
	out2, _ := out.WithColumn(repl)
	if v, _ := out2.MustCol("temp").At(0); v.(float64) != 0 {
		t.Fatal("replace column")
	}
	if out2.NumCols() != out.NumCols() {
		t.Fatal("replace changed col count")
	}
	// Wrong length.
	bad := NewSeries("bad", []any{1.0})
	if _, err := df.WithColumn(bad); err == nil {
		t.Fatal("expected length error")
	}
	// Add to empty frame.
	empty, _ := NewDataFrame()
	e2, err := empty.WithColumn(NewSeries("x", []any{1.0, 2.0}))
	if err != nil || e2.NumRows() != 2 {
		t.Fatalf("with column on empty: %v", err)
	}
}

func TestHeadTailILocLocTake(t *testing.T) {
	df := sampleDF(t)
	if df.Head(2).NumRows() != 2 {
		t.Fatal("head")
	}
	if df.Tail(1).NumRows() != 1 {
		t.Fatal("tail")
	}
	if df.Tail(99).NumRows() != 4 {
		t.Fatal("tail clamp")
	}
	sl := df.ILoc(1, 3)
	if sl.NumRows() != 2 {
		t.Fatalf("iloc = %d", sl.NumRows())
	}
	if v, _ := sl.MustCol("city").At(0); v.(string) != "LA" {
		t.Fatalf("iloc content = %v", v)
	}
	// Out of range clamps.
	if df.ILoc(-5, 99).NumRows() != 4 {
		t.Fatal("iloc clamp")
	}
	if df.ILoc(3, 1).NumRows() != 0 {
		t.Fatal("iloc reversed")
	}
	loc := df.Loc(int64(0), int64(2))
	if loc.NumRows() != 2 {
		t.Fatalf("loc = %d", loc.NumRows())
	}
	tk := df.Take([]int{2, 99, 0})
	if tk.NumRows() != 2 {
		t.Fatalf("take = %d", tk.NumRows())
	}
}

func TestFilterFuncAndRow(t *testing.T) {
	df := sampleDF(t)
	hot := df.FilterFunc(func(r Row) bool {
		f, ok := r.Float("temp")
		return ok && f > 30
	})
	if hot.NumRows() != 2 {
		t.Fatalf("filterfunc = %d", hot.NumRows())
	}
	r := df.Row(0)
	if v, ok := r.Get("city"); !ok || v.(string) != "NYC" {
		t.Fatalf("row get = %v", v)
	}
	if _, ok := r.Get("nope"); ok {
		t.Fatal("row get missing")
	}
	if _, ok := r.Float("city"); ok {
		t.Fatal("row float on string")
	}
	if r.Label().(int64) != 0 {
		t.Fatalf("label = %v", r.Label())
	}
	// Float on missing row value.
	r3 := df.Row(3)
	if _, ok := r3.Float("sunny"); ok {
		t.Fatal("float on NA")
	}
}

func TestFilterMask(t *testing.T) {
	df := sampleDF(t)
	f := df.Filter([]bool{true, false, false, true})
	if f.NumRows() != 2 {
		t.Fatalf("filter = %d", f.NumRows())
	}
}

func TestDropFillNA(t *testing.T) {
	df := sampleDF(t)
	d := df.DropNA()
	if d.NumRows() != 3 {
		t.Fatalf("dropna = %d", d.NumRows())
	}
	f := df.FillNA("sunny", true)
	if v, ok := f.MustCol("sunny").At(3); !ok || v.(bool) != true {
		t.Fatalf("fillna = %v,%v", v, ok)
	}
	// Fill all columns (empty column name) with a value; only matching dtype fills.
	fAll := df.FillNA("", true)
	if _, ok := fAll.MustCol("sunny").At(3); !ok {
		t.Fatal("fillna all")
	}
}

func TestSortBy(t *testing.T) {
	df := sampleDF(t)
	sorted, err := df.SortBy([]string{"temp"}, []bool{true})
	if err != nil {
		t.Fatal(err)
	}
	if v, _ := sorted.MustCol("temp").At(0); v.(float64) != 20 {
		t.Fatalf("sortby = %v", v)
	}
	// Descending.
	desc, _ := df.SortBy([]string{"temp"}, []bool{false})
	if v, _ := desc.MustCol("temp").At(0); v.(float64) != 33 {
		t.Fatalf("sortby desc = %v", v)
	}
	// Multi-key with default ascending (nil slice).
	multi, err := df.SortBy([]string{"city", "temp"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if v, _ := multi.MustCol("city").At(0); v.(string) != "LA" {
		t.Fatalf("multi sort = %v", v)
	}
	if _, err := df.SortBy([]string{"nope"}, nil); err == nil {
		t.Fatal("sortby missing")
	}
}

func TestSortByNAOrdering(t *testing.T) {
	df, _ := FromMap(map[string][]any{
		"k": {3.0, nil, 1.0},
	}, []string{"k"})
	sorted, _ := df.SortBy([]string{"k"}, []bool{true})
	// present values first, NA last.
	if _, ok := sorted.MustCol("k").At(2); ok {
		t.Fatal("NA should sort last")
	}
}

func TestDescribe(t *testing.T) {
	df := sampleDF(t)
	desc := df.Describe()
	if !desc.HasColumn("temp") || desc.HasColumn("city") {
		t.Fatalf("describe cols = %v", desc.Names())
	}
	if desc.NumRows() != 5 {
		t.Fatalf("describe rows = %d", desc.NumRows())
	}
	// count of temp = 4.
	if v, _ := desc.MustCol("temp").At(0); v.(float64) != 4 {
		t.Fatalf("describe count = %v", v)
	}
}

func TestDataFrameStringAndCopy(t *testing.T) {
	df := sampleDF(t)
	if df.String() == "" {
		t.Fatal("string")
	}
	c := df.Copy()
	c.columns[0].data[0] = "MUT"
	if v, _ := df.MustCol("city").At(0); v.(string) != "NYC" {
		t.Fatal("copy not deep")
	}
}
