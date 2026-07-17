package pandas

import (
	"fmt"
	"strconv"
)

// DType enumerates the element types a Series can hold.
type DType int

const (
	// Float64 is a column of float64 values.
	Float64 DType = iota
	// Int64 is a column of int64 values.
	Int64
	// String is a column of string values.
	String
	// Bool is a column of bool values.
	Bool
	// Object is a column of arbitrary values whose concrete type is not one
	// of the specialised kinds above.
	Object
)

// String returns the human readable name of the dtype.
func (d DType) String() string {
	switch d {
	case Float64:
		return "float64"
	case Int64:
		return "int64"
	case String:
		return "string"
	case Bool:
		return "bool"
	default:
		return "object"
	}
}

// inferDType returns the DType that best describes v. A nil value is treated as
// Object because it carries no type information.
func inferDType(v any) DType {
	switch v.(type) {
	case float64, float32:
		return Float64
	case int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64:
		return Int64
	case string:
		return String
	case bool:
		return Bool
	default:
		return Object
	}
}

// coerce converts v to the canonical Go representation for dtype d. It reports
// whether the conversion succeeded; a false result means the value should be
// treated as missing for that column.
func coerce(v any, d DType) (any, bool) {
	if v == nil {
		return nil, false
	}
	switch d {
	case Float64:
		return toFloat64(v)
	case Int64:
		return toInt64(v)
	case String:
		if s, ok := v.(string); ok {
			return s, true
		}
		return fmt.Sprint(v), true
	case Bool:
		if b, ok := v.(bool); ok {
			return b, true
		}
		if s, ok := v.(string); ok {
			b, err := strconv.ParseBool(s)
			if err != nil {
				return nil, false
			}
			return b, true
		}
		return nil, false
	default:
		return v, true
	}
}

// toFloat64 converts numeric and string values to float64.
func toFloat64(v any) (float64, bool) {
	switch t := v.(type) {
	case float64:
		return t, true
	case float32:
		return float64(t), true
	case int:
		return float64(t), true
	case int8:
		return float64(t), true
	case int16:
		return float64(t), true
	case int32:
		return float64(t), true
	case int64:
		return float64(t), true
	case uint:
		return float64(t), true
	case uint8:
		return float64(t), true
	case uint16:
		return float64(t), true
	case uint32:
		return float64(t), true
	case uint64:
		return float64(t), true
	case bool:
		if t {
			return 1, true
		}
		return 0, true
	case string:
		f, err := strconv.ParseFloat(t, 64)
		if err != nil {
			return 0, false
		}
		return f, true
	default:
		return 0, false
	}
}

// toInt64 converts numeric and string values to int64.
func toInt64(v any) (int64, bool) {
	switch t := v.(type) {
	case int:
		return int64(t), true
	case int8:
		return int64(t), true
	case int16:
		return int64(t), true
	case int32:
		return int64(t), true
	case int64:
		return t, true
	case uint:
		return int64(t), true
	case uint8:
		return int64(t), true
	case uint16:
		return int64(t), true
	case uint32:
		return int64(t), true
	case uint64:
		return int64(t), true
	case float64:
		return int64(t), true
	case float32:
		return int64(t), true
	case string:
		i, err := strconv.ParseInt(t, 10, 64)
		if err != nil {
			return 0, false
		}
		return i, true
	default:
		return 0, false
	}
}

// formatValue renders a stored value for display and CSV output.
func formatValue(v any) string {
	switch t := v.(type) {
	case nil:
		return ""
	case float64:
		return strconv.FormatFloat(t, 'g', -1, 64)
	case int64:
		return strconv.FormatInt(t, 10)
	case string:
		return t
	case bool:
		return strconv.FormatBool(t)
	default:
		return fmt.Sprint(t)
	}
}
