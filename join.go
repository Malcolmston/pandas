package pandas

import (
	"fmt"
)

// JoinType selects the join semantics.
type JoinType int

const (
	// InnerJoin keeps only rows whose key exists in both frames.
	InnerJoin JoinType = iota
	// LeftJoin keeps every left row, filling unmatched right columns with NA.
	LeftJoin
)

// Merge joins df (left) with right on the given key column, which must exist in
// both frames. Non-key columns that collide are suffixed with "_left" and
// "_right". The output row order is deterministic: left rows in order, and for
// each left row its matching right rows in their original order.
func (df *DataFrame) Merge(right *DataFrame, on string, how JoinType) (*DataFrame, error) {
	leftKey, ok := df.Col(on)
	if !ok {
		return nil, fmt.Errorf("pandas: left frame has no column %q", on)
	}
	rightKey, ok := right.Col(on)
	if !ok {
		return nil, fmt.Errorf("pandas: right frame has no column %q", on)
	}

	// Index right rows by key value.
	rIndex := make(map[any][]int)
	for r := 0; r < right.NumRows(); r++ {
		if !rightKey.valid[r] {
			continue
		}
		k := rightKey.data[r]
		rIndex[k] = append(rIndex[k], r)
	}

	// Determine output columns: key, then left non-key, then right non-key.
	leftOther := otherColumns(df, on)
	rightOther := otherColumns(right, on)
	leftNames, rightNames := resolveNames(on, leftOther, rightOther)

	// Prepare builders.
	keyOut := &Series{name: on, dtype: leftKey.dtype}
	leftOut := make([]*Series, len(leftOther))
	for i, name := range leftOther {
		src, _ := df.Col(name)
		leftOut[i] = &Series{name: leftNames[i], dtype: src.dtype}
	}
	rightOut := make([]*Series, len(rightOther))
	for i, name := range rightOther {
		src, _ := right.Col(name)
		rightOut[i] = &Series{name: rightNames[i], dtype: src.dtype}
	}

	appendLeft := func(lr int) {
		for i, name := range leftOther {
			src, _ := df.Col(name)
			leftOut[i].data = append(leftOut[i].data, src.data[lr])
			leftOut[i].valid = append(leftOut[i].valid, src.valid[lr])
		}
	}
	appendRight := func(rr int) {
		for i, name := range rightOther {
			src, _ := right.Col(name)
			rightOut[i].data = append(rightOut[i].data, src.data[rr])
			rightOut[i].valid = append(rightOut[i].valid, src.valid[rr])
		}
	}
	appendRightNA := func() {
		for i := range rightOther {
			rightOut[i].data = append(rightOut[i].data, nil)
			rightOut[i].valid = append(rightOut[i].valid, false)
		}
	}

	for lr := 0; lr < df.NumRows(); lr++ {
		var matches []int
		if leftKey.valid[lr] {
			matches = rIndex[leftKey.data[lr]]
		}
		if len(matches) == 0 {
			if how == LeftJoin {
				keyOut.data = append(keyOut.data, leftKey.data[lr])
				keyOut.valid = append(keyOut.valid, leftKey.valid[lr])
				appendLeft(lr)
				appendRightNA()
			}
			continue
		}
		for _, rr := range matches {
			keyOut.data = append(keyOut.data, leftKey.data[lr])
			keyOut.valid = append(keyOut.valid, leftKey.valid[lr])
			appendLeft(lr)
			appendRight(rr)
		}
	}

	out := make([]*Series, 0, 1+len(leftOut)+len(rightOut))
	out = append(out, keyOut)
	out = append(out, leftOut...)
	out = append(out, rightOut...)
	return NewDataFrame(out...)
}

// otherColumns returns the column names of df excluding the key.
func otherColumns(df *DataFrame, key string) []string {
	var out []string
	for _, n := range df.names {
		if n != key {
			out = append(out, n)
		}
	}
	return out
}

// resolveNames disambiguates colliding non-key column names with _left/_right
// suffixes.
func resolveNames(key string, left, right []string) (leftNames, rightNames []string) {
	rightSet := make(map[string]struct{}, len(right))
	for _, n := range right {
		rightSet[n] = struct{}{}
	}
	leftSet := make(map[string]struct{}, len(left))
	for _, n := range left {
		leftSet[n] = struct{}{}
	}
	leftNames = make([]string, len(left))
	for i, n := range left {
		if _, clash := rightSet[n]; clash || n == key {
			leftNames[i] = n + "_left"
		} else {
			leftNames[i] = n
		}
	}
	rightNames = make([]string, len(right))
	for i, n := range right {
		if _, clash := leftSet[n]; clash || n == key {
			rightNames[i] = n + "_right"
		} else {
			rightNames[i] = n
		}
	}
	return leftNames, rightNames
}
