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

func Example_multiply() {
	a, _ := matrix.New(2, 3, []float64{0, 1, 2, 3, 4, 5})
	b, _ := matrix.New(2, 3, []float64{0, 1, 2, 3, 4, 5})

	a.Multiply(a, b)
	fmt.Println(a.Raw())
	// Output:
	// [0 1 4 9 16 25]
}
