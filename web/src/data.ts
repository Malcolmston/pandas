// Library content for the pandas documentation site. Mirrors the shape used by
// the malcolmston/go landing site's data.ts so the sibling sites stay in sync.
export interface Lib {
  id: string; name: string; icon: string; accent: string; pkg: string; node: string;
  repo: string; docs: string; tagline: string; blurb: string; tags: string[];
  features: string[]; node_code: string; go_code: string; integrate: string;
}

export const NODE_ACCENT = '#8cc84b';

export const PANDAS: Lib = {
  id:"pandas", name:"pandas", icon:'<i class="fa-solid fa-table"></i>', accent:"#e86bb0",
  pkg:"github.com/malcolmston/pandas", node:"pandas-dev/pandas",
  repo:"https://github.com/malcolmston/pandas", docs:"https://malcolmston.github.io/pandas/",
  tagline:"pandas-style DataFrames and data analysis for Go.",
  blurb:"A from-scratch, standard-library-only Go take on pandas: a named, typed one-dimensional "+
    "Series with first-class missing-value (NA) support, and an ordered DataFrame of equal-length "+
    "columns built from column maps, slices of structs, or CSV. On top of those two types sit the "+
    "everyday analysis verbs — column and row selection (Select, Col, ILoc, Loc, FilterFunc), "+
    "transformation (WithColumn, SortBy, FillNA, DropNA, Describe), GroupBy aggregations (Sum, Mean, "+
    "Min, Max, Count, Std) and inner/left Merge. Everything is built on encoding/csv, sort, strconv "+
    "and reflect — no cgo, no third-party modules — and every operation produces a stable, "+
    "reproducible ordering with missing values sorted last.",
  tags:["Series","DataFrame","NA-aware","GroupBy","Merge","CSV I/O","Describe","stdlib-only"],
  features:[
    "<code>Series</code> — a named, typed 1-D column (<code>Float64</code>/<code>Int64</code>/<code>String</code>/<code>Bool</code> + <code>Object</code>) with an index and first-class <code>IsNA</code> missing-value support",
    "<code>DataFrame</code> construction from column maps (<code>FromMap</code>), slices of structs (<code>FromRecords</code>), Series (<code>NewDataFrame</code>) or CSV (<code>ReadCSV</code>/<code>ReadCSVFile</code>)",
    "Selection &amp; indexing — columns via <code>Select</code>/<code>Col</code>/<code>Drop</code>, rows via <code>ILoc</code>/<code>Head</code>/<code>Tail</code>, labels via <code>Loc</code>, masks via <code>Filter</code>/<code>FilterFunc</code>",
    "Transformation — <code>WithColumn</code>, <code>Rename</code>, <code>Apply</code>/<code>Map</code>, <code>SortBy</code> on one or more keys, and NA handling with <code>FillNA</code>/<code>DropNA</code>",
    "<code>GroupBy</code> partitioning with deterministic ordering and the <code>Sum</code>, <code>Mean</code>, <code>Min</code>, <code>Max</code>, <code>Count</code>, <code>Std</code> aggregations (or the general <code>Agg</code>)",
    "<code>Merge</code> — inner and left joins on a shared key (<code>InnerJoin</code>/<code>LeftJoin</code>), with <code>_left</code>/<code>_right</code> suffixes for colliding columns",
    "<code>Describe</code> — count / mean / std / min / max for every numeric column, plus <code>Unique</code> and <code>ValueCounts</code>",
    "Zero dependencies — pure Go standard library (<code>encoding/csv</code>, <code>sort</code>, <code>strconv</code>, <code>reflect</code>), with stable, reproducible ordering throughout"
  ],
  node_code:
`import pandas as pd

df = pd.DataFrame({
    "city":  ["NYC", "LA", "NYC", "LA"],
    "month": ["Jan", "Jan", "Feb", "Feb"],
    "sales": [100.0, 80.0, 120.0, 90.0],
})

hot = df[df["sales"] >= 100]
means = df.groupby("city")["sales"].mean()
print(df.describe())`,
  go_code:
`import "github.com/malcolmston/pandas"

df, _ := pandas.FromMap(map[string][]any{
    "city":  {"NYC", "LA", "NYC", "LA"},
    "month": {"Jan", "Jan", "Feb", "Feb"},
    "sales": {100.0, 80.0, 120.0, 90.0},
}, []string{"city", "month", "sales"})

hot := df.FilterFunc(func(r pandas.Row) bool {
    v, ok := r.Float("sales")
    return ok && v >= 100
})
gb, _ := df.GroupBy("city")
means, _ := gb.Mean("sales")
fmt.Print(hot, means, df.Describe())`,
  integrate:
`<span class="tok-c">// Build a DataFrame from column data; a nil cell becomes NA.</span>
df, _ := pandas.FromMap(map[string][]any{
    "city":  {"NYC", "LA", "NYC", "LA"},
    "units": {10.0, 8.0, 12.0, nil},
    "price": {9.99, 12.50, 9.99, 15.0},
}, []string{"city", "units", "price"})

<span class="tok-c">// Fill the missing unit count, then attach a derived revenue column.</span>
df = df.FillNA("units", 0.0)
rev := pandas.NewSeries("revenue", []any{99.9, 100.0, 119.88, 0.0})
df, _ = df.WithColumn(rev)

<span class="tok-c">// Sort by revenue descending — NA sorts last, ties break deterministically.</span>
df, _ = df.SortBy([]string{"revenue"}, []bool{false})

<span class="tok-c">// Group by city, total the revenue, then summarise every numeric column.</span>
gb, _ := df.GroupBy("city")
totals, _ := gb.Sum("revenue")
fmt.Print(totals)
fmt.Print(df.Describe())

<span class="tok-c">// Inner-join against a lookup table on the shared key column.</span>
joined, _ := df.Merge(regions, "city", pandas.InnerJoin)
fmt.Print(joined)`
};
