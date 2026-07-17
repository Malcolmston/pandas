package pandas

import "testing"

func joinFrames(t *testing.T) (*DataFrame, *DataFrame) {
	t.Helper()
	left, _ := FromMap(map[string][]any{
		"id":  {int64(1), int64(2), int64(3)},
		"val": {"a", "b", "c"},
	}, []string{"id", "val"})
	right, _ := FromMap(map[string][]any{
		"id":  {int64(2), int64(3), int64(4)},
		"val": {"x", "y", "z"},
	}, []string{"id", "val"})
	return left, right
}

func TestMergeInner(t *testing.T) {
	left, right := joinFrames(t)
	out, err := left.Merge(right, "id", InnerJoin)
	if err != nil {
		t.Fatal(err)
	}
	// ids 2 and 3 match.
	if out.NumRows() != 2 {
		t.Fatalf("inner rows = %d", out.NumRows())
	}
	// Colliding "val" becomes val_left / val_right.
	if !out.HasColumn("val_left") || !out.HasColumn("val_right") {
		t.Fatalf("cols = %v", out.Names())
	}
	if v, _ := out.MustCol("val_left").At(0); v.(string) != "b" {
		t.Fatalf("val_left = %v", v)
	}
	if v, _ := out.MustCol("val_right").At(0); v.(string) != "x" {
		t.Fatalf("val_right = %v", v)
	}
}

func TestMergeLeft(t *testing.T) {
	left, right := joinFrames(t)
	out, err := left.Merge(right, "id", LeftJoin)
	if err != nil {
		t.Fatal(err)
	}
	// All 3 left rows kept.
	if out.NumRows() != 3 {
		t.Fatalf("left rows = %d", out.NumRows())
	}
	// id=1 has no match -> right NA.
	if v, _ := out.MustCol("id").At(0); v.(int64) != 1 {
		t.Fatalf("first id = %v", v)
	}
	if _, ok := out.MustCol("val_right").At(0); ok {
		t.Fatal("unmatched right should be NA")
	}
}

func TestMergeNoCollision(t *testing.T) {
	left, _ := FromMap(map[string][]any{
		"id":   {int64(1), int64(2)},
		"name": {"a", "b"},
	}, []string{"id", "name"})
	right, _ := FromMap(map[string][]any{
		"id":  {int64(1), int64(2)},
		"qty": {int64(10), int64(20)},
	}, []string{"id", "qty"})
	out, err := left.Merge(right, "id", InnerJoin)
	if err != nil {
		t.Fatal(err)
	}
	// No suffixing needed.
	if !out.HasColumn("name") || !out.HasColumn("qty") {
		t.Fatalf("cols = %v", out.Names())
	}
}

func TestMergeErrors(t *testing.T) {
	left, right := joinFrames(t)
	if _, err := left.Merge(right, "nope", InnerJoin); err == nil {
		t.Fatal("missing left key")
	}
	l2, _ := FromMap(map[string][]any{"other": {int64(1)}}, []string{"other"})
	if _, err := left.Merge(l2, "id", InnerJoin); err == nil {
		t.Fatal("missing right key")
	}
}

func TestMergeNAKey(t *testing.T) {
	left, _ := FromMap(map[string][]any{
		"id": {int64(1), nil},
		"a":  {"p", "q"},
	}, []string{"id", "a"})
	right, _ := FromMap(map[string][]any{
		"id": {int64(1)},
		"b":  {"r"},
	}, []string{"id", "b"})
	// Inner: NA key on left never matches.
	out, _ := left.Merge(right, "id", InnerJoin)
	if out.NumRows() != 1 {
		t.Fatalf("inner NA rows = %d", out.NumRows())
	}
	// Left: NA row kept with NA right side.
	outL, _ := left.Merge(right, "id", LeftJoin)
	if outL.NumRows() != 2 {
		t.Fatalf("left NA rows = %d", outL.NumRows())
	}
}
