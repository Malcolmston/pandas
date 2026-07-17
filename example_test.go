package pandas_test

import (
	"fmt"
	"os"
	"strings"

	"github.com/malcolmston/pandas"
)

func Example() {
	// Build a DataFrame from column data.
	df, _ := pandas.FromMap(map[string][]any{
		"city":  {"NYC", "LA", "NYC", "LA"},
		"month": {"Jan", "Jan", "Feb", "Feb"},
		"sales": {100.0, 80.0, 120.0, 90.0},
	}, []string{"city", "month", "sales"})

	// Group by city and take the mean of sales.
	gb, _ := df.GroupBy("city")
	means, _ := gb.Mean("sales")
	fmt.Print(means.String())

	// Output:
	// city  sales_mean
	// LA    85
	// NYC   110
}

func ExampleReadCSV() {
	data := "name,age,active\nAlice,30,true\nBob,25,false\n"
	df, _ := pandas.ReadCSV(strings.NewReader(data), nil)
	fmt.Printf("rows=%d cols=%d\n", df.NumRows(), df.NumCols())

	age := df.MustCol("age")
	mean, _ := age.Mean()
	fmt.Printf("mean age=%.1f dtype=%s\n", mean, age.DType())

	var b strings.Builder
	_ = df.WriteCSV(&b, nil)
	fmt.Print(b.String())

	// Output:
	// rows=2 cols=3
	// mean age=27.5 dtype=int64
	// name,age,active
	// Alice,30,true
	// Bob,25,false
}

func ExampleDataFrame_Merge() {
	left, _ := pandas.FromMap(map[string][]any{
		"id":   {int64(1), int64(2), int64(3)},
		"name": {"a", "b", "c"},
	}, []string{"id", "name"})
	right, _ := pandas.FromMap(map[string][]any{
		"id":  {int64(1), int64(2)},
		"qty": {int64(10), int64(20)},
	}, []string{"id", "qty"})

	inner, _ := left.Merge(right, "id", pandas.InnerJoin)
	fmt.Print(inner.String())

	// Output:
	// id  name  qty
	// 1   a     10
	// 2   b     20
}

func ExampleDataFrame_Describe() {
	df, _ := pandas.FromMap(map[string][]any{
		"x": {1.0, 2.0, 3.0, 4.0},
	}, []string{"x"})
	desc := df.Describe()
	// Write to stdout via CSV for a compact, stable rendering.
	_ = desc.WriteCSV(os.Stdout, nil)

	// Output:
	// stat,x
	// count,4
	// mean,2.5
	// std,1.2909944487358056
	// min,1
	// max,4
}
