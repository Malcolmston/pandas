// Package pandas provides pandas-style data structures for Go: a typed,
// one-dimensional Series and a two-dimensional DataFrame, built entirely on the
// Go standard library.
//
// # Series
//
// A Series is a named, typed column of values with an associated index and
// first-class support for missing values (NA). The supported element types
// (see DType) are Float64, Int64, String and Bool, plus a fallback Object type
// for anything else. Values are coerced to the column dtype on construction; a
// nil input or a value that cannot be coerced becomes NA.
//
//	s := pandas.NewSeries("price", []any{10.0, 20.0, nil, 40.0})
//	mean, ok := s.Mean() // 23.333..., true (NA is skipped)
//
// Series offers element access (At, Values, IsNA), transformation (Apply, Map,
// FillNA, DropNA, Sort, Filter, Head, Tail), set operations (Unique,
// ValueCounts) and reductions (Sum, Mean, Min, Max, Std, Count).
//
// # DataFrame
//
// A DataFrame is an ordered set of equal-length named columns sharing one row
// index. Construct one from Series (NewDataFrame), a map of column to values
// (FromMap), a slice of structs (FromRecords) or CSV data (ReadCSV,
// ReadCSVFile). Write it back out with WriteCSV and WriteCSVFile.
//
//	df, _ := pandas.FromMap(map[string][]any{
//	    "city": {"NYC", "LA", "NYC"},
//	    "temp": {31.0, 28.0, 33.0},
//	}, []string{"city", "temp"})
//
// # Selection and indexing
//
// Columns are selected by name (Select, Col, Drop). Rows are selected by
// position range (ILoc, Head, Tail), by index label (Loc), by explicit
// positions (Take) or by a boolean mask (Filter, FilterFunc). The Row type
// gives a keyed, read-only view of a single row.
//
// # Transformation
//
// DataFrames support adding and replacing columns (WithColumn), dropping and
// renaming (Drop, Rename), sorting by one or more keys (SortBy), and missing
// value handling (FillNA, DropNA).
//
// # GroupBy and aggregation
//
// GroupBy partitions a DataFrame by one or more key columns, with deterministic
// group ordering. Aggregate with the AggFunc set — Sum, Mean, Min, Max, Count,
// Std — either through the typed convenience methods or the general Agg method,
// which computes several aggregations at once.
//
//	gb, _ := df.GroupBy("city")
//	means, _ := gb.Mean("temp") // one row per city
//
// # Join and merge
//
// Merge joins two DataFrames on a shared key column with inner or left join
// semantics (InnerJoin, LeftJoin). Colliding non-key columns are disambiguated
// with _left and _right suffixes.
//
// # Describe
//
// Describe returns count, mean, std, min and max for every numeric column.
//
// # Determinism
//
// Every operation produces a stable, reproducible ordering: grouping,
// value_counts and sorting all break ties deterministically, and missing
// values sort last.
package pandas
