package pandas

import (
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestReadCSVInference(t *testing.T) {
	data := "a,b,c,d\n1,1.5,true,x\n2,2.5,false,y\n,,,\n"
	df, err := ReadCSV(strings.NewReader(data), &ReadCSVOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if df.MustCol("a").DType() != Int64 {
		t.Fatalf("a dtype = %v", df.MustCol("a").DType())
	}
	if df.MustCol("b").DType() != Float64 {
		t.Fatalf("b dtype = %v", df.MustCol("b").DType())
	}
	if df.MustCol("c").DType() != Bool {
		t.Fatalf("c dtype = %v", df.MustCol("c").DType())
	}
	if df.MustCol("d").DType() != String {
		t.Fatalf("d dtype = %v", df.MustCol("d").DType())
	}
	// Third row is all-NA.
	if _, ok := df.MustCol("a").At(2); ok {
		t.Fatal("expected NA")
	}
}

func TestReadCSVNoHeader(t *testing.T) {
	data := "1,2\n3,4\n"
	df, err := ReadCSV(strings.NewReader(data), &ReadCSVOptions{NoHeader: true})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(df.Names(), []string{"col0", "col1"}) {
		t.Fatalf("names = %v", df.Names())
	}
	if df.NumRows() != 2 {
		t.Fatalf("rows = %d", df.NumRows())
	}
}

func TestReadCSVDelimiterAndNA(t *testing.T) {
	data := "a;b\n1;NULL\n2;5\n"
	df, err := ReadCSV(strings.NewReader(data), &ReadCSVOptions{Delimiter: ';', NAValues: []string{"NULL"}})
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := df.MustCol("b").At(0); ok {
		t.Fatal("NULL should be NA")
	}
}

func TestReadCSVEmpty(t *testing.T) {
	df, err := ReadCSV(strings.NewReader(""), nil)
	if err != nil {
		t.Fatal(err)
	}
	if df.NumRows() != 0 {
		t.Fatalf("rows = %d", df.NumRows())
	}
}

func TestReadCSVError(t *testing.T) {
	// Unterminated quote -> parse error.
	if _, err := ReadCSV(strings.NewReader("a\n\"unterminated"), nil); err == nil {
		t.Fatal("expected csv error")
	}
}

func TestWriteReadRoundTripFile(t *testing.T) {
	df, _ := FromMap(map[string][]any{
		"n": {int64(1), int64(2)},
		"v": {1.5, nil},
	}, []string{"n", "v"})
	path := filepath.Join(t.TempDir(), "out.csv")
	if err := df.WriteCSVFile(path, nil); err != nil {
		t.Fatal(err)
	}
	back, err := ReadCSVFile(path, nil)
	if err != nil {
		t.Fatal(err)
	}
	if back.NumRows() != 2 || back.NumCols() != 2 {
		t.Fatalf("roundtrip shape %d,%d", back.NumRows(), back.NumCols())
	}
	if _, ok := back.MustCol("v").At(1); ok {
		t.Fatal("NA lost in roundtrip")
	}
}

func TestReadCSVFileMissing(t *testing.T) {
	if _, err := ReadCSVFile("/no/such/file.csv", nil); err == nil {
		t.Fatal("expected open error")
	}
}

func TestWriteCSVFileError(t *testing.T) {
	df, _ := FromMap(map[string][]any{"a": {1.0}}, []string{"a"})
	if err := df.WriteCSVFile("/no/such/dir/out.csv", nil); err == nil {
		t.Fatal("expected create error")
	}
}
