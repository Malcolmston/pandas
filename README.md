# pandas

DataFrames and data analysis for Go — a pandas-style `Series` and `DataFrame`,
built entirely on the Go standard library (no cgo, no third-party modules).

## Features

- **Series**: a named, typed 1-D column (`float64`, `int64`, `string`, `bool`,
  plus an `Object` fallback) with an index and first-class missing-value (NA)
  support.
- **DataFrame**: ordered, equal-length named columns. Build from column maps,
  slices of structs, or CSV; write back to CSV.
- **Selection & indexing**: select columns by name, rows by position (`ILoc`,
  `Head`, `Tail`), by index label (`Loc`), or by boolean mask (`Filter`,
  `FilterFunc`).
- **Transformation**: add/drop/rename columns, `Apply`/`Map` over a column,
  `SortBy` one or more keys, `FillNA`/`DropNA`, `Unique`/`ValueCounts`.
- **GroupBy** with `Sum`, `Mean`, `Min`, `Max`, `Count`, `Std` aggregations.
- **Merge**: inner and left joins on a key column.
- **Describe**: count / mean / std / min / max per numeric column.
- **Deterministic**: every operation produces a stable, reproducible ordering.

## Install

```sh
go get github.com/malcolmston/pandas
```

Requires Go 1.24 or newer.

## Quick start

```go
package main

import (
	"fmt"
	"os"

	"github.com/malcolmston/pandas"
)

func main() {
	// Build a DataFrame from column data.
	df, _ := pandas.FromMap(map[string][]any{
		"city":  {"NYC", "LA", "NYC", "LA"},
		"month": {"Jan", "Jan", "Feb", "Feb"},
		"sales": {100.0, 80.0, 120.0, 90.0},
	}, []string{"city", "month", "sales"})

	// Filter rows.
	hot := df.FilterFunc(func(r pandas.Row) bool {
		v, ok := r.Float("sales")
		return ok && v >= 100
	})
	fmt.Print(hot)

	// Group by city and average sales.
	gb, _ := df.GroupBy("city")
	means, _ := gb.Mean("sales")
	fmt.Print(means)

	// Summary statistics for numeric columns.
	fmt.Print(df.Describe())

	// Write to CSV.
	_ = df.WriteCSV(os.Stdout, nil)
}
```

### Reading CSV

```go
f, _ := os.Open("data.csv")
defer f.Close()
df, _ := pandas.ReadCSV(f, nil) // column dtypes are inferred
```

### Joining

```go
inner, _ := left.Merge(right, "id", pandas.InnerJoin)
left2, _ := left.Merge(right, "id", pandas.LeftJoin)
```

## Documentation

Run `go doc github.com/malcolmston/pandas` for the full API, or read the package
overview in [`doc.go`](doc.go).

## License

See repository.
