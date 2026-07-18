# Changelog

## 0.2.0

Expanded the API toward pandas parity with 56 new exported functions/types
(standard-library only, deterministic), all covered by known-answer table tests
and benchmarks for the performance-sensitive routines.

### Series statistics (stats.go)
- `Var`, `Prod`, `Quantile`, `Median`, `Mode`, `ArgMax`, `ArgMin`
- `Cov`, `Corr` (Pearson) between two series
- `Rank` (average method), `NLargest`, `NSmallest`

### Series transforms (transform.go)
- `Abs`, `Round`, `Clip`, `Astype`
- `CumSum`, `CumProd`, `CumMax`, `CumMin`
- `Shift`, `Diff`, `PctChange`
- `Between`, `IsIn`
- Element-wise arithmetic: `Add`, `Sub`, `Mul`, `Div`

### Vectorised string accessor (strings.go)
- `Series.Str` returning `StrAccessor` with `Lower`, `Upper`, `Title`,
  `Strip`, `Replace`, `Len`, `Contains`, `StartsWith`, `EndsWith`

### DataFrame operations (frame_ops.go)
- Column-wise reductions returning a Series indexed by column name:
  `Sum`, `Mean`, `Min`, `Max`, `Std`, `Var`, `Median`, `Nunique`
- `Abs`, `Round`, `DropDuplicates`
- `SetIndex`, `ResetIndex`
- `Corr` correlation matrix, `Transpose`
- Top-level `Concat` for vertically stacking frames (union of columns)

## 0.1.0

Initial release: Series, DataFrame, selection/indexing, transformation,
GroupBy aggregation, Merge, Describe, and CSV I/O.
