package pandas

import "testing"

func TestDTypeString(t *testing.T) {
	cases := map[DType]string{
		Float64: "float64", Int64: "int64", String: "string",
		Bool: "bool", Object: "object",
	}
	for d, want := range cases {
		if d.String() != want {
			t.Fatalf("DType(%d).String() = %q, want %q", d, d.String(), want)
		}
	}
}

func TestInferDType(t *testing.T) {
	cases := []struct {
		v    any
		want DType
	}{
		{1.0, Float64},
		{float32(1), Float64},
		{int(1), Int64},
		{int64(1), Int64},
		{uint8(1), Int64},
		{"s", String},
		{true, Bool},
		{[]int{1}, Object},
		{nil, Object},
	}
	for _, c := range cases {
		if got := inferDType(c.v); got != c.want {
			t.Fatalf("inferDType(%v) = %v, want %v", c.v, got, c.want)
		}
	}
}

func TestToFloat64(t *testing.T) {
	if f, ok := toFloat64("3.5"); !ok || f != 3.5 {
		t.Fatalf("string->float = %v,%v", f, ok)
	}
	if f, ok := toFloat64(true); !ok || f != 1 {
		t.Fatalf("bool->float = %v,%v", f, ok)
	}
	if f, ok := toFloat64(false); !ok || f != 0 {
		t.Fatalf("false->float = %v,%v", f, ok)
	}
	if _, ok := toFloat64("nope"); ok {
		t.Fatal("bad string should fail")
	}
	if _, ok := toFloat64([]int{1}); ok {
		t.Fatal("unsupported type should fail")
	}
	for _, v := range []any{int8(1), int16(1), int32(1), int64(1), uint(1), uint16(1), uint32(1), uint64(1), float32(1)} {
		if f, ok := toFloat64(v); !ok || f != 1 {
			t.Fatalf("toFloat64(%T) = %v,%v", v, f, ok)
		}
	}
}

func TestToInt64(t *testing.T) {
	if i, ok := toInt64("42"); !ok || i != 42 {
		t.Fatalf("string->int = %v,%v", i, ok)
	}
	if i, ok := toInt64(3.9); !ok || i != 3 {
		t.Fatalf("float->int = %v,%v", i, ok)
	}
	if _, ok := toInt64("nope"); ok {
		t.Fatal("bad string should fail")
	}
	if _, ok := toInt64(true); ok {
		t.Fatal("bool->int unsupported")
	}
	for _, v := range []any{int8(1), int16(1), int32(1), uint(1), uint8(1), uint16(1), uint32(1), uint64(1), float32(1)} {
		if i, ok := toInt64(v); !ok || i != 1 {
			t.Fatalf("toInt64(%T) = %v,%v", v, i, ok)
		}
	}
}

func TestCoerce(t *testing.T) {
	if _, ok := coerce(nil, Float64); ok {
		t.Fatal("nil should be missing")
	}
	if v, ok := coerce(5, String); !ok || v.(string) != "5" {
		t.Fatalf("int->string = %v,%v", v, ok)
	}
	if v, ok := coerce("true", Bool); !ok || v.(bool) != true {
		t.Fatalf("string->bool = %v,%v", v, ok)
	}
	if _, ok := coerce("notbool", Bool); ok {
		t.Fatal("bad bool should fail")
	}
	if _, ok := coerce(5, Bool); ok {
		t.Fatal("int->bool should fail")
	}
	if v, ok := coerce([]int{1}, Object); !ok {
		t.Fatalf("object passthrough = %v,%v", v, ok)
	}
}

func TestFormatValue(t *testing.T) {
	cases := []struct {
		v    any
		want string
	}{
		{nil, ""},
		{1.5, "1.5"},
		{int64(7), "7"},
		{"hi", "hi"},
		{true, "true"},
		{[]int{1}, "[1]"},
	}
	for _, c := range cases {
		if got := formatValue(c.v); got != c.want {
			t.Fatalf("formatValue(%v) = %q, want %q", c.v, got, c.want)
		}
	}
}
