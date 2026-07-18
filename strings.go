package pandas

import (
	"strings"
	"unicode/utf8"
)

// StrAccessor provides vectorised string operations over a Series, mirroring
// the pandas Series.str accessor. Operations that return a Series preserve the
// index and leave missing values missing; operations that return a boolean mask
// yield false for missing or non-string values.
type StrAccessor struct {
	s *Series
}

// Str returns a string accessor for the series. Values are interpreted using
// their display form, so a String-dtype series works directly while other
// dtypes are formatted first.
func (s *Series) Str() StrAccessor { return StrAccessor{s: s} }

// xmapString applies fn to the string form of every present value, producing a
// new String series with the same index.
func (a StrAccessor) xmapString(fn func(string) string) *Series {
	s := a.s
	data := make([]any, s.Len())
	valid := make([]bool, s.Len())
	for i := range s.data {
		if !s.valid[i] {
			continue
		}
		data[i] = fn(formatValue(s.data[i]))
		valid[i] = true
	}
	return xbuildSeries(s.name, String, data, valid, s.index)
}

// xmapBool applies pred to the string form of every present value, producing a
// boolean mask; missing values map to false.
func (a StrAccessor) xmapBool(pred func(string) bool) []bool {
	s := a.s
	out := make([]bool, s.Len())
	for i := range s.data {
		if !s.valid[i] {
			continue
		}
		out[i] = pred(formatValue(s.data[i]))
	}
	return out
}

// Lower returns a new series with every present value lower-cased.
func (a StrAccessor) Lower() *Series {
	return a.xmapString(strings.ToLower)
}

// Upper returns a new series with every present value upper-cased.
func (a StrAccessor) Upper() *Series {
	return a.xmapString(strings.ToUpper)
}

// Title returns a new series with the first letter of each whitespace-separated
// word capitalised and the remaining letters lower-cased.
func (a StrAccessor) Title() *Series {
	return a.xmapString(xtitleCase)
}

// xtitleCase capitalises the first rune of every whitespace-delimited word.
func xtitleCase(s string) string {
	var b strings.Builder
	prevSpace := true
	for _, r := range strings.ToLower(s) {
		if prevSpace {
			b.WriteString(strings.ToUpper(string(r)))
		} else {
			b.WriteRune(r)
		}
		prevSpace = r == ' ' || r == '\t' || r == '\n'
	}
	return b.String()
}

// Strip returns a new series with leading and trailing white space removed from
// every present value.
func (a StrAccessor) Strip() *Series {
	return a.xmapString(strings.TrimSpace)
}

// Replace returns a new series with every non-overlapping occurrence of old
// replaced by new in each present value.
func (a StrAccessor) Replace(old, new string) *Series {
	return a.xmapString(func(s string) string { return strings.ReplaceAll(s, old, new) })
}

// Len returns an Int64 series giving the number of Unicode code points in each
// present value; missing values remain missing.
func (a StrAccessor) Len() *Series {
	s := a.s
	data := make([]any, s.Len())
	valid := make([]bool, s.Len())
	for i := range s.data {
		if !s.valid[i] {
			continue
		}
		data[i] = int64(utf8.RuneCountInString(formatValue(s.data[i])))
		valid[i] = true
	}
	return xbuildSeries(s.name, Int64, data, valid, s.index)
}

// Contains returns a boolean mask reporting, per position, whether the present
// value contains the substring sub.
func (a StrAccessor) Contains(sub string) []bool {
	return a.xmapBool(func(s string) bool { return strings.Contains(s, sub) })
}

// StartsWith returns a boolean mask reporting, per position, whether the present
// value begins with prefix.
func (a StrAccessor) StartsWith(prefix string) []bool {
	return a.xmapBool(func(s string) bool { return strings.HasPrefix(s, prefix) })
}

// EndsWith returns a boolean mask reporting, per position, whether the present
// value ends with suffix.
func (a StrAccessor) EndsWith(suffix string) []bool {
	return a.xmapBool(func(s string) bool { return strings.HasSuffix(s, suffix) })
}
