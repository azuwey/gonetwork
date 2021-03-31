package matrix_test

import (
	"fmt"

	"github.com/azuwey/gonetwork/matrix"
)

func ExampleNew() {
	// Create a 2 x 3 matrix.
	mat, _ := matrix.New(2, 3, []float64{0, 1, 2, 3, 4, 5})

	fmt.Println(mat.Values)
	// Output:
	// [0 1 2 3 4 5]
}

func ExampleCopy() {
	// Create a 2 x 3 matrix.
	aMat, _ := matrix.New(2, 3, []float64{0, 1, 2, 3, 4, 5})

	// Copy the "a" matrix.
	bMat, _ := matrix.Copy(aMat)

	// Change the values of the "a" matrix.
	aMat.Scale(2, aMat)

	fmt.Println(aMat.Values)
	fmt.Println(bMat.Values)
	// Output:
	// [0 2 4 6 8 10]
	// [0 1 2 3 4 5]
}

func ExampleNew_zeros() {
	// Create a 2 x 3 zero matrix.
	mat, _ := matrix.New(2, 3, nil)

	fmt.Println(mat.Values)
	// Output:
	// [0 0 0 0 0 0]
}

func Example_add() {
	aMat, _ := matrix.New(2, 3, []float64{0, 1, 2, 3, 4, 5})
	bMat, _ := matrix.New(2, 3, []float64{0, 1, 2, 3, 4, 5})

	aMat.Add(aMat, bMat)
	fmt.Println(aMat.Values)
	// Output:
	// [0 2 4 6 8 10]
}

func Example_apply() {
	mat, _ := matrix.New(2, 3, []float64{0, 1, 2, 3, 4, 5})

	mat.Apply(func(v float64, r, c int, s []float64) float64 {
		fmt.Printf("Slice: %v;\n", s)
		fmt.Printf("Row: %d; Column: %d; Value: %f;\n", r, c, v)
		newValue := v + 2
		fmt.Printf("Row: %d; Column: %d; New value: %f;\n", r, c, newValue)
		return newValue
	}, mat)

	fmt.Println(mat.Values)
	// Output:
	// Slice: [0 1 2 3 4 5];
	// Row: 0; Column: 0; Value: 0.000000;
	// Row: 0; Column: 0; New value: 2.000000;
	// Slice: [2 1 2 3 4 5];
	// Row: 0; Column: 1; Value: 1.000000;
	// Row: 0; Column: 1; New value: 3.000000;
	// Slice: [2 3 2 3 4 5];
	// Row: 0; Column: 2; Value: 2.000000;
	// Row: 0; Column: 2; New value: 4.000000;
	// Slice: [2 3 4 3 4 5];
	// Row: 1; Column: 0; Value: 3.000000;
	// Row: 1; Column: 0; New value: 5.000000;
	// Slice: [2 3 4 5 4 5];
	// Row: 1; Column: 1; Value: 4.000000;
	// Row: 1; Column: 1; New value: 6.000000;
	// Slice: [2 3 4 5 6 5];
	// Row: 1; Column: 2; Value: 5.000000;
	// Row: 1; Column: 2; New value: 7.000000;
	// [2 3 4 5 6 7]
}

func Example_at() {
	mat, _ := matrix.New(2, 3, []float64{0, 1, 2, 3, 4, 5})

	val, _ := mat.At(1, 2)
	fmt.Println(val)
	// Output:
	// 5
}

func Example_multiply() {
	aMat, _ := matrix.New(2, 3, []float64{0, 1, 2, 3, 4, 5})
	bMat, _ := matrix.New(2, 3, []float64{0, 1, 2, 3, 4, 5})

	aMat.Multiply(aMat, bMat)
	fmt.Println(aMat.Values)
	// Output:
	// [0 1 4 9 16 25]
}

func Example_product() {
	aMat, _ := matrix.New(4, 3, []float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11})
	bMat, _ := matrix.New(3, 1, []float64{0, -1, 2})

	aMat.Product(aMat, bMat)
	fmt.Println(aMat.Values)
	fmt.Println(aMat.Rows, aMat.Columns)
	// Output:
	// [3 6 9 12]
	// 4 1
}

func Example_raw() {
	mat, _ := matrix.New(2, 3, []float64{0, 1, 2, 3, 4, 5})

	fmt.Println(mat.Values)
	// Output:
	// [0 1 2 3 4 5]
}

func Example_scale() {
	mat, _ := matrix.New(2, 3, []float64{0, 1, 2, 3, 4, 5})

	mat.Scale(2, mat)
	fmt.Println(mat.Values)
	// Output:
	// [0 2 4 6 8 10]
}

func Example_subtract() {
	aMat, _ := matrix.New(2, 3, []float64{2, 4, 8, 16, 32, 64})
	bMat, _ := matrix.New(2, 3, []float64{0, 1, 2, 3, 4, 5})

	aMat.Subtract(aMat, bMat)
	fmt.Println(aMat.Values)
	// Output:
	// [2 3 6 13 28 59]
}

func Example_transpose() {
	mat, _ := matrix.New(2, 3, []float64{0, 1, 2, 3, 4, 5})

	mat.Transpose(mat)
	fmt.Println(mat.Values)
	// Output:
	// [0 2 4 1 3 5]
}

func Example_export() {
	mat, _ := matrix.New(2, 3, []float64{0, 1, 2, 3, 4, 5})

	json, _ := mat.Export()
	fmt.Println(string(json))
	// Output:
	// {"values":[0,1,2,3,4,5],"rows":2,"Columns":3}
}

func Example_import() {
	mJSON := []byte(`{"values":[0,1,2,3,4,5],"rows":2,"Columns":3}`)
	mat := &matrix.Matrix{}

	mat.Import(mJSON)
	fmt.Println(mat.Values)
	fmt.Println(mat.Rows, mat.Columns)
	// Output:
	// [0 1 2 3 4 5]
	// 2 3
}
