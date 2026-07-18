package pandas

import (
	"math"
	"testing"
)

// xdump renders a series' values as strings ("NA" for missing) for structural
// comparison in tests.
func xdump(s *Series) []string {
	out := make([]string, s.Len())
	for i := range s.data {
		if s.valid[i] {
			out[i] = formatValue(s.data[i])
		} else {
			out[i] = "NA"
		}
	}
	return out
}

// xidx renders a series' index labels as strings.
func xidx(s *Series) []string {
	out := make([]string, len(s.index))
	for i := range s.index {
		out[i] = formatValue(s.index[i])
	}
	return out
}

// xeqStr fails the test when two string slices differ.
func xeqStr(t *testing.T, got, want []string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length mismatch: got %v, want %v", got, want)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("at %d: got %q, want %q (full got=%v want=%v)", i, got[i], want[i], got, want)
		}
	}
}

// xeqBool fails the test when two bool slices differ.
func xeqBool(t *testing.T, got, want []bool) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length mismatch: got %v, want %v", got, want)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("at %d: got %v, want %v", i, got, want)
		}
	}
}

// xclose reports approximate float equality.
func xclose(a, b float64) bool { return math.Abs(a-b) < 1e-9 }
