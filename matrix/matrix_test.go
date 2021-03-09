package matrix

import (
	"errors"
	"fmt"
	"testing"
)

func TestNew(t *testing.T) {
	testCases := []struct {
		rows, cols             int
		values, expectedValues []float64
		expectedError          error
	}{
		{2, 2, []float64{0, 1, 2, 3}, []float64{0, 1, 2, 3}, nil},
		{2, 2, nil, []float64{0, 0, 0, 0}, nil},
		{0, 0, nil, nil, ErrZeroRowOrCol},
		{2, 2, []float64{0}, nil, ErrDataLength},
	}

	for tcIndex, tcValue := range testCases {
		tcValue, tcIndex := tcValue, tcIndex // capture range variables
		t.Run(fmt.Sprintf("[%d] %+v", tcIndex, tcValue), func(t *testing.T) {
			t.Parallel()

			m, err := New(tcValue.rows, tcValue.cols, tcValue.values)

			if tcValue.expectedError != nil {
				if !errors.Is(err, tcValue.expectedError) {
					t.Logf("Err should be %v but it's %v", tcValue.expectedError, err)
					t.Fail()
				}
			}

			if tcValue.expectedValues != nil {
				if len(m.values) != len(tcValue.expectedValues) {
					t.Logf("Lenght of the matrix should be %d but it's %d", len(tcValue.expectedValues), len(m.values))
					t.Fail()
				}

				for i, v := range m.values {
					if v != tcValue.expectedValues[i] {
						t.Logf("Value of the matrix should be %f but it's %f", tcValue.expectedValues[i], v)
						t.Fail()
					}
				}
			} else if m != nil {
				t.Logf("Matrix should be nil, but it's %+v", m)
				t.Fail()
			}
		})
	}
}

func TestAdd(t *testing.T) {
	testCases := []struct {
		aMatrix, bMatrix, dMatrix *Matrix
		expectedValues            []float64
		expectedError             error
	}{
		{&Matrix{[]float64{0, 1, 2, 3}, 2, 2}, &Matrix{[]float64{0, 1, 2, 3}, 2, 2}, &Matrix{[]float64{}, 0, 0}, []float64{0, 2, 4, 6}, nil},
		{&Matrix{[]float64{0, 1, 2, 3}, 2, 2}, nil, &Matrix{[]float64{}, 0, 0}, nil, ErrNilMatrix},
		{nil, &Matrix{[]float64{0, 1, 2, 3}, 2, 2}, &Matrix{[]float64{}, 0, 0}, nil, ErrNilMatrix},
		{&Matrix{[]float64{0, 1, 2, 3}, 2, 2}, &Matrix{[]float64{0, 1}, 1, 2}, &Matrix{[]float64{}, 0, 0}, nil, ErrDifferentDimesion},
	}

	for tcIndex, tcValue := range testCases {
		tcValue, tcIndex := tcValue, tcIndex // capture range variables
		t.Run(fmt.Sprintf("[%d] %+v", tcIndex, tcValue), func(t *testing.T) {
			t.Parallel()

			err := tcValue.dMatrix.Add(tcValue.aMatrix, tcValue.bMatrix)

			if tcValue.expectedError != nil {
				if !errors.Is(err, tcValue.expectedError) {
					t.Logf("Err should be %v but it's %v", tcValue.expectedError, err)
					t.Fail()
				}
			}

			if tcValue.expectedValues != nil {
				if len(tcValue.dMatrix.values) != len(tcValue.expectedValues) {
					t.Logf("Lenght of the matrix should be %d but it's %d", len(tcValue.expectedValues), len(tcValue.dMatrix.values))
					t.Fail()
				}

				for i, v := range tcValue.dMatrix.values {
					if v != tcValue.expectedValues[i] {
						t.Logf("Value of the matrix should be %f but it's %f", tcValue.expectedValues[i], v)
						t.Fail()
					}
				}
			}
		})
	}
}

func TestAddSelf(t *testing.T) {
	t.Parallel()

	expectedValues := []float64{0, 2, 4, 6}
	aMatrix := &Matrix{[]float64{0, 1, 2, 3}, 2, 2}
	bMatrix := &Matrix{[]float64{0, 1, 2, 3}, 2, 2}

	err := aMatrix.Add(aMatrix, bMatrix)

	if err != nil {
		t.Logf("Err should be %v but it's %v", nil, err)
		t.Fail()
	}

	for i, v := range aMatrix.values {
		if v != expectedValues[i] {
			t.Logf("Value of the matrix should be %f but it's %f", expectedValues[i], v)
			t.Fail()
		}
	}
}

func TestAt(t *testing.T) {
	testCases := []struct {
		rows, cols    int
		checkIndexes  []int
		values        []float64
		expectedError error
	}{
		{3, 2, []int{0, 1, 2, 3, 4, 5}, []float64{-1, -3, -5, -3, -8, -15}, nil},
		{3, 3, []int{6}, []float64{-1, -3, -5, -3, -8, -15}, ErrOutOfBounds},
	}

	for tcIndex, tcValue := range testCases {
		tcValue, tcIndex := tcValue, tcIndex // capture range variables
		t.Run(fmt.Sprintf("[%d] %+v", tcIndex, tcValue), func(t *testing.T) {
			t.Parallel()

			aMatrix := &Matrix{tcValue.values, tcValue.rows, tcValue.cols}

			for r := 0; r < tcValue.rows; r++ {
				for c := 0; c < tcValue.cols; c++ {
					v, err := aMatrix.At(r, c)
					if tcValue.expectedError != nil {
						if errors.Is(err, tcValue.expectedError) {
							t.Logf("Err should be %v but it's %v", tcValue.expectedError, err)
							t.Fail()
						}
						return
					}

					if v != tcValue.values[r*tcValue.cols+c] {
						t.Logf("Value of the matrix should be %f but it's %f", tcValue.values[r*tcValue.cols+c], v)
						t.Fail()
					}
				}
			}
		})
	}
}

func TestDimension(t *testing.T) {
	testCases := []struct {
		matrix                   *Matrix
		expectedRow, expectedCol int
	}{
		{&Matrix{[]float64{0, 1, 2, 3}, 2, 2}, 2, 2},
		{&Matrix{[]float64{0, 1, 2}, 1, 3}, 1, 3},
	}

	for tcIndex, tcValue := range testCases {
		tcValue, tcIndex := tcValue, tcIndex // capture range variables
		t.Run(fmt.Sprintf("[%d] %+v", tcIndex, tcValue), func(t *testing.T) {
			t.Parallel()

			row, col := tcValue.matrix.Dimensions()

			if row != tcValue.expectedRow {
				t.Logf("Row should be %d but it's %d", tcValue.expectedRow, row)
				t.Fail()
			}

			if col != tcValue.expectedCol {
				t.Logf("Col should be %d but it's %d", tcValue.expectedCol, col)
				t.Fail()
			}
		})
	}
}

func TestMultiply(t *testing.T) {
	testCases := []struct {
		aMatrix, bMatrix, dMatrix *Matrix
		expectedValues            []float64
		expectedError             error
	}{
		{&Matrix{[]float64{0, 1, 2, 3}, 2, 2}, &Matrix{[]float64{0, 1, 2, 3}, 2, 2}, &Matrix{[]float64{}, 0, 0}, []float64{0, 1, 4, 9}, nil},
		{&Matrix{[]float64{0, 1, 2, 3}, 2, 2}, nil, &Matrix{[]float64{}, 0, 0}, nil, ErrNilMatrix},
		{nil, &Matrix{[]float64{0, 1, 2, 3}, 2, 2}, &Matrix{[]float64{}, 0, 0}, nil, ErrNilMatrix},
		{&Matrix{[]float64{0, 1, 2, 3}, 2, 2}, &Matrix{[]float64{0, 1}, 1, 2}, &Matrix{[]float64{}, 0, 0}, nil, ErrDifferentDimesion},
	}

	for tcIndex, tcValue := range testCases {
		tcValue, tcIndex := tcValue, tcIndex // capture range variables
		t.Run(fmt.Sprintf("[%d] %+v", tcIndex, tcValue), func(t *testing.T) {
			t.Parallel()

			err := tcValue.dMatrix.Multiply(tcValue.aMatrix, tcValue.bMatrix)

			if tcValue.expectedError != nil {
				if !errors.Is(err, tcValue.expectedError) {
					t.Logf("Err should be %v but it's %v", tcValue.expectedError, err)
					t.Fail()
				}
			}

			if tcValue.expectedValues != nil {
				if len(tcValue.dMatrix.values) != len(tcValue.expectedValues) {
					t.Logf("Lenght of the matrix should be %d but it's %d", len(tcValue.expectedValues), len(tcValue.dMatrix.values))
					t.Fail()
				}

				for i, v := range tcValue.dMatrix.values {
					if v != tcValue.expectedValues[i] {
						t.Logf("Value of the matrix should be %f but it's %f", tcValue.expectedValues[i], v)
						t.Fail()
					}
				}
			}
		})
	}
}

func TestMultiplySelf(t *testing.T) {
	t.Parallel()

	expectedValues := []float64{0, 1, 4, 9}
	aMatrix := &Matrix{[]float64{0, 1, 2, 3}, 2, 2}
	bMatrix := &Matrix{[]float64{0, 1, 2, 3}, 2, 2}

	err := aMatrix.Multiply(aMatrix, bMatrix)

	if err != nil {
		t.Logf("Err should be %v but it's %v", nil, err)
		t.Fail()
	}

	for i, v := range aMatrix.values {
		if v != expectedValues[i] {
			t.Logf("Value of the matrix should be %f but it's %f", expectedValues[i], v)
			t.Fail()
		}
	}
}

func TestProduct(t *testing.T) {
	testCases := []struct {
		aMatrix, bMatrix, dMatrix *Matrix
		expectedValues            []float64
		expectedError             error
	}{
		{&Matrix{[]float64{0, 1, 2, 3, 4, 5}, 3, 2}, &Matrix{[]float64{0, -1}, 2, 1}, &Matrix{[]float64{}, 0, 0}, []float64{-1, -3, -5}, nil},
		{&Matrix{[]float64{0, 1, 2, 3}, 2, 2}, nil, &Matrix{[]float64{}, 0, 0}, nil, ErrNilMatrix},
		{nil, &Matrix{[]float64{0, 1, 2, 3}, 2, 2}, &Matrix{[]float64{}, 0, 0}, nil, ErrNilMatrix},
		{&Matrix{[]float64{0, 1, 2, 3, 4, 5}, 2, 3}, &Matrix{[]float64{0, -1}, 1, 2}, &Matrix{[]float64{}, 0, 0}, nil, ErrBadProductDimesion},
	}

	for tcIndex, tcValue := range testCases {
		tcValue, tcIndex := tcValue, tcIndex // capture range variables
		t.Run(fmt.Sprintf("[%d] %+v", tcIndex, tcValue), func(t *testing.T) {
			t.Parallel()

			err := tcValue.dMatrix.Product(tcValue.aMatrix, tcValue.bMatrix)

			if tcValue.expectedError != nil {
				if !errors.Is(err, tcValue.expectedError) {
					t.Logf("Err should be %v but it's %v", tcValue.expectedError, err)
					t.Fail()
				}
			}

			if tcValue.expectedValues != nil {
				if len(tcValue.dMatrix.values) != len(tcValue.expectedValues) {
					t.Logf("Lenght of the matrix should be %d but it's %d", len(tcValue.expectedValues), len(tcValue.dMatrix.values))
					t.Fail()
				}

				for i, v := range tcValue.dMatrix.values {
					if v != tcValue.expectedValues[i] {
						t.Logf("Value of the matrix should be %f but it's %f", tcValue.expectedValues[i], v)
						t.Fail()
					}
				}
			}
		})
	}
}

func TestProductSelf(t *testing.T) {
	t.Parallel()

	expectedValues := []float64{-1, -3, -5}
	aMatrix := &Matrix{[]float64{0, 1, 2, 3, 4, 5}, 3, 2}
	bMatrix := &Matrix{[]float64{0, -1}, 2, 1}

	err := aMatrix.Product(aMatrix, bMatrix)

	if err != nil {
		t.Logf("Err should be %v but it's %v", nil, err)
		t.Fail()
	}

	for i, v := range aMatrix.values {
		if v != expectedValues[i] {
			t.Logf("Value of the matrix should be %f but it's %f", expectedValues[i], v)
			t.Fail()
		}
	}
}

func TestRaw(t *testing.T) {
	t.Parallel()

	values := []float64{-1, -3, -5}
	aMatrix := &Matrix{values, 1, 3}

	for i, v := range aMatrix.Raw() {
		if v != values[i] {
			t.Logf("Value of the matrix should be %f but it's %f", values[i], v)
			t.Fail()
		}
	}
}

func TestScale(t *testing.T) {
	testCases := []struct {
		aMatrix, bMatrix *Matrix
		scalar           float64
		expectedValues   []float64
		expectedError    error
	}{
		{&Matrix{[]float64{0, 1, 2, 3, 4, 5}, 3, 2}, &Matrix{[]float64{}, 0, 0}, 2, []float64{0, 2, 4, 6, 8, 10}, nil},
		{nil, &Matrix{[]float64{}, 0, 0}, 2, nil, ErrNilMatrix},
	}

	for tcIndex, tcValue := range testCases {
		tcValue, tcIndex := tcValue, tcIndex // capture range variables
		t.Run(fmt.Sprintf("[%d] %+v", tcIndex, tcValue), func(t *testing.T) {
			t.Parallel()

			err := tcValue.bMatrix.Scale(tcValue.scalar, tcValue.aMatrix)

			if tcValue.expectedError != nil {
				if !errors.Is(err, tcValue.expectedError) {
					t.Logf("Err should be %v but it's %v", tcValue.expectedError, err)
					t.Fail()
				}
			}

			if tcValue.expectedValues != nil {
				if len(tcValue.bMatrix.values) != len(tcValue.expectedValues) {
					t.Logf("Lenght of the matrix should be %d but it's %d", len(tcValue.expectedValues), len(tcValue.bMatrix.values))
					t.Fail()
				}

				for i, v := range tcValue.bMatrix.values {
					if v != tcValue.expectedValues[i] {
						t.Logf("Value of the matrix should be %f but it's %f", tcValue.expectedValues[i], v)
						t.Fail()
					}
				}
			}
		})
	}
}

func TestScaleSelf(t *testing.T) {
	t.Parallel()

	expectedValues := []float64{0, 2, 4, 6, 8, 10}
	aMatrix := &Matrix{[]float64{0, 1, 2, 3, 4, 5}, 3, 2}
	scalar := 2.0

	err := aMatrix.Scale(scalar, aMatrix)

	if err != nil {
		t.Logf("Err should be %v but it's %v", nil, err)
		t.Fail()
	}

	for i, v := range aMatrix.values {
		if v != expectedValues[i] {
			t.Logf("Value of the matrix should be %f but it's %f", expectedValues[i], v)
			t.Fail()
		}
	}
}

func TestValues(t *testing.T) {
	t.Parallel()

	rows, cols := 3, 2
	aMatrix := &Matrix{[]float64{-1, -3, -5, -3, -8, -15}, rows, cols}
	expectedValues := [][]float64{{-1, -3}, {-5, -3}, {-8, -15}}

	v := aMatrix.Values()

	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			if v[r][c] != expectedValues[r][c] {
				t.Logf("Value of the matrix should be %f but it's %f", expectedValues[r][c], v[r][c])
				t.Fail()
			}
		}
	}
}

// TODO: Apply test
func TestApply(t *testing.T) {
	t.Parallel()

	rows, cols := 3, 2
	aMatrix := &Matrix{[]float64{0, 1, 2, 3, 4, 5}, rows, cols}
	expectedValues := []float64{2, 3, 4, 5, 6, 7}
	expectedRows := []int{0, 0, 1, 1, 2, 2}
	expectedCols := []int{0, 1, 0, 1, 0, 1}

	currentIndex := 0
	err := aMatrix.Apply(func(v float64, r, c int) float64 {
		if r != expectedRows[currentIndex] {
			t.Logf("Row should be %d but it's %d", expectedRows[currentIndex], r)
			t.Fail()
		}

		if c != expectedCols[currentIndex] {
			t.Logf("Column should be %d but it's %d", expectedCols[currentIndex], r)
			t.Fail()
		}

		currentIndex++
		return v + 2
	}, aMatrix)

	if err != nil {
		t.Logf("Err should be %v but it's %v", nil, err)
		t.Fail()
	}

	for i, v := range aMatrix.values {
		if v != expectedValues[i] {
			t.Logf("Value of the matrix should be %f but it's %f", expectedValues[i], v)
			t.Fail()
		}
	}
}

func TestApplyNilFunction(t *testing.T) {
	t.Parallel()

	rows, cols := 3, 2
	aMatrix := &Matrix{[]float64{0, 1, 2, 3, 4, 5}, rows, cols}

	err := aMatrix.Apply(nil, aMatrix)

	if !errors.Is(err, ErrNilFunction) {
		t.Logf("Err should be %v but it's %v", ErrNilFunction, err)
		t.Fail()
	}
}

func TestApplyNilMatrix(t *testing.T) {
	t.Parallel()

	rows, cols := 3, 2
	aMatrix := &Matrix{[]float64{0, 1, 2, 3, 4, 5}, rows, cols}

	err := aMatrix.Apply(func(v float64, r, c int) float64 {
		return v
	}, nil)

	if !errors.Is(err, ErrNilMatrix) {
		t.Logf("Err should be %v but it's %v", ErrNilMatrix, err)
		t.Fail()
	}
}