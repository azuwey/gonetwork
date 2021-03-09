package matrix_test

import (
	"fmt"

	"github.com/azuwey/gonetwork/matrix"
)

func ExampleNew() {
	// Create a 2 x 3 matrix.
	m, _ := matrix.New(2, 3, []float64{0, 1, 2, 3, 4, 5})
	fmt.Println(m.Raw())
}

func ExampleNew_zeros() {
	// Create a 2 x 3 zero matrix.
	m, _ := matrix.New(2, 3, nil)
	fmt.Println(m.Raw())
}
