package pandas

import "testing"

func TestStrCase(t *testing.T) {
	s := NewSeries("x", []any{"Ab", "cD", nil})
	xeqStr(t, xdump(s.Str().Lower()), []string{"ab", "cd", "NA"})
	xeqStr(t, xdump(s.Str().Upper()), []string{"AB", "CD", "NA"})
}

func TestStrTitle(t *testing.T) {
	s := NewSeries("x", []any{"hello world", "GO lang"})
	xeqStr(t, xdump(s.Str().Title()), []string{"Hello World", "Go Lang"})
}

func TestStrLen(t *testing.T) {
	s := NewSeries("x", []any{"ab", "abc", nil, "héllo"})
	got := s.Str().Len()
	if got.DType() != Int64 {
		t.Fatalf("dtype: %v", got.DType())
	}
	xeqStr(t, xdump(got), []string{"2", "3", "NA", "5"})
}

func TestStrStripReplace(t *testing.T) {
	s := NewSeries("x", []any{"  a  ", "b"})
	xeqStr(t, xdump(s.Str().Strip()), []string{"a", "b"})
	r := NewSeries("x", []any{"foo", "bar"})
	xeqStr(t, xdump(r.Str().Replace("o", "0")), []string{"f00", "bar"})
}

func TestStrPredicates(t *testing.T) {
	s := NewSeries("x", []any{"cat", "dog", nil})
	xeqBool(t, s.Str().Contains("a"), []bool{true, false, false})
	xeqBool(t, s.Str().StartsWith("ca"), []bool{true, false, false})
	xeqBool(t, s.Str().EndsWith("og"), []bool{false, true, false})
}
