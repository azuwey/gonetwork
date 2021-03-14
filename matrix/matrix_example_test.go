package matrix_test

import (
	"fmt"

	"github.com/azuwey/gonetwork/matrix"
)

func ExampleNew() {
	// Create a 2 x 3 matrix.
	m, _ := matrix.New(2, 3, []float64{0, 1, 2, 3, 4, 5})

	fmt.Println(m.Raw())
	// Output:
	// [0 1 2 3 4 5]
}

func ExampleNew_zeros() {
	// Create a 2 x 3 zero matrix.
	m, _ := matrix.New(2, 3, nil)

	fmt.Println(m.Raw())
	// Output:
	// [0 0 0 0 0 0]
}

func Example_add() {
	a, _ := matrix.New(2, 3, []float64{0, 1, 2, 3, 4, 5})
	b, _ := matrix.New(2, 3, []float64{0, 1, 2, 3, 4, 5})

	a.Add(a, b)
	fmt.Println(a.Raw())
	// Output:
	// [0 2 4 6 8 10]
}

func Example_apply() {
	a, _ := matrix.New(2, 3, []float64{0, 1, 2, 3, 4, 5})

	a.Apply(func(v float64, r, c int) float64 {
		fmt.Printf("Row: %d; Column: %d; Value: %f;\n", r, c, v)
		newValue := v + 2
		fmt.Printf("Row: %d; Column: %d; New value: %f;\n", r, c, v)
		return newValue
	}, a)

	fmt.Println(a.Raw())
	// Output:
	// Row: 0; Column: 0; Value: 0.000000;
	// Row: 0; Column: 0; New value: 0.000000;
	// Row: 0; Column: 1; Value: 1.000000;
	// Row: 0; Column: 1; New value: 1.000000;
	// Row: 0; Column: 2; Value: 2.000000;
	// Row: 0; Column: 2; New value: 2.000000;
	// Row: 1; Column: 0; Value: 3.000000;
	// Row: 1; Column: 0; New value: 3.000000;
	// Row: 1; Column: 1; Value: 4.000000;
	// Row: 1; Column: 1; New value: 4.000000;
	// Row: 1; Column: 2; Value: 5.000000;
	// Row: 1; Column: 2; New value: 5.000000;
	// [2 3 4 5 6 7]
}

func Example_at() {
	a, _ := matrix.New(2, 3, []float64{0, 1, 2, 3, 4, 5})

	v, _ := a.At(1, 2)
	fmt.Println(v)
	// Output:
	// 5
}

func Example_dimensions() {
	a, _ := matrix.New(2, 3, []float64{0, 1, 2, 3, 4, 5})

	r, c := a.Dimensions()
	fmt.Println(r, c)
	// Output:
	// 2 3
}

func Example_matrixProduct() {
	a, _ := matrix.New(4, 3, []float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11})
	b, _ := matrix.New(3, 1, []float64{0, -1, 2})

	a.MatrixProduct(a, b)
	fmt.Println(a.Raw())
	fmt.Println(a.Dimensions())
	// Output:
	// [3 6 9 12]
	// 4 1
}

func Example_multiply() {
	a, _ := matrix.New(2, 3, []float64{0, 1, 2, 3, 4, 5})
	b, _ := matrix.New(2, 3, []float64{0, 1, 2, 3, 4, 5})

	a.Multiply(a, b)
	fmt.Println(a.Raw())
	// Output:
	// [0 1 4 9 16 25]
}

func Example_raw() {
	a, _ := matrix.New(2, 3, []float64{0, 1, 2, 3, 4, 5})

	fmt.Println(a.Raw())
	// Output:
	// [0 1 2 3 4 5]
}

func Example_scale() {
	a, _ := matrix.New(2, 3, []float64{0, 1, 2, 3, 4, 5})

	a.Scale(2, a)
	fmt.Println(a.Raw())
	// Output:
	// [0 2 4 6 8 10]
}

func Example_subtract() {
	a, _ := matrix.New(2, 3, []float64{2, 4, 8, 16, 32, 64})
	b, _ := matrix.New(2, 3, []float64{0, 1, 2, 3, 4, 5})

	a.Subtract(a, b)
	fmt.Println(a.Raw())
	// Output:
	// [2 3 6 13 28 59]
}

func Example_transpose() {
	a, _ := matrix.New(2, 3, []float64{0, 1, 2, 3, 4, 5})

	a.Transpose(a)
	fmt.Println(a.Raw())
	// Output:
	// [0 3 1 4 2 5]
}

func Example_values() {
	a, _ := matrix.New(2, 3, []float64{0, 1, 2, 3, 4, 5})

	fmt.Println(a.Values())
	// Output:
	// [[0 1 2] [3 4 5]]
}
