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
				return
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

func TestNewDataChange(t *testing.T) {
	t.Parallel()

	data := []float64{0, 1, 2, 3, 4, 5}
	m, err := New(2, 3, data)

	if !errors.Is(err, nil) {
		t.Logf("Err should be %v but it's %v", nil, err)
		t.Fail()
	}

	if len(m.values) != len(data) {
		t.Logf("Lenght of the matrix should be %d but it's %d", len(data), len(m.values))
		t.Fail()
	}

	for i, v := range m.values {
		if v != data[i] {
			t.Logf("Value of the matrix should be %f but it's %f", data[i], v)
			t.Fail()
		}
	}

	data[0] = 25
	if m.values[0] != data[0] {
		t.Logf("Value of the matrix should be %f but it's %f", data[0], m.values[0])
		t.Fail()
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
		{&Matrix{[]float64{0, 1, 2, 3}, 2, 2}, &Matrix{[]float64{0, 1}, 1, 2}, &Matrix{[]float64{}, 0, 0}, nil, ErrDifferentDimension},
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
				return
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

	if !errors.Is(err, nil) {
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

	if !errors.Is(err, nil) {
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

func TestAt(t *testing.T) {
	testCases := []struct {
		rows, cols    int
		checkIndexes  [][]int
		values        []float64
		expectedError error
	}{
		{3, 2, [][]int{{0, 1}, {0, 1}, {0, 1}}, []float64{-1, -3, -5, -3, -8, -15}, nil},
		{3, 2, [][]int{{2}}, []float64{-1, -3, -5, -3, -8, -15}, ErrOutOfBounds},
	}

	for tcIndex, tcValue := range testCases {
		tcValue, tcIndex := tcValue, tcIndex // capture range variables
		t.Run(fmt.Sprintf("[%d] %+v", tcIndex, tcValue), func(t *testing.T) {
			t.Parallel()

			aMatrix := &Matrix{tcValue.values, tcValue.rows, tcValue.cols}

			for rIndex, r := range tcValue.checkIndexes {
				for _, c := range r {
					v, err := aMatrix.At(rIndex, c)

					if tcValue.expectedError != nil {
						if !errors.Is(err, tcValue.expectedError) {
							t.Logf("Err should be %v but it's %v", tcValue.expectedError, err)
							t.Fail()
						}
						return
					}

					if v != tcValue.values[rIndex*tcValue.cols+c] {
						t.Logf("Value of the matrix should be %f but it's %f", tcValue.values[rIndex*tcValue.cols+c], v)
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

func TestMatrixProduct(t *testing.T) {
	testCases := []struct {
		aMatrix, bMatrix, dMatrix *Matrix
		expectedValues            []float64
		expectedError             error
	}{
		{&Matrix{[]float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, 4, 3}, &Matrix{[]float64{0, -1, 2}, 3, 1}, &Matrix{[]float64{}, 0, 0}, []float64{3, 6, 9, 12}, nil},
		{&Matrix{[]float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, 4, 3}, &Matrix{[]float64{0, -1, -2, -3, -4, -5}, 3, 2}, &Matrix{[]float64{}, 0, 0}, []float64{-10, -13, -28, -40, -46, -67, -64, -94}, nil},
		{&Matrix{[]float64{0, 1, 2, 3, 4, 5}, 2, 3}, &Matrix{[]float64{0, 1, 2, 3, 4, 5}, 3, 2}, &Matrix{[]float64{}, 0, 0}, []float64{10, 13, 28, 40}, nil},
		{&Matrix{[]float64{0, 1, 2, 3}, 2, 2}, nil, &Matrix{[]float64{}, 0, 0}, nil, ErrNilMatrix},
		{nil, &Matrix{[]float64{0, 1, 2, 3}, 2, 2}, &Matrix{[]float64{}, 0, 0}, nil, ErrNilMatrix},
		{&Matrix{[]float64{0, 1, 2, 3, 4, 5}, 2, 3}, &Matrix{[]float64{0, -1}, 1, 2}, &Matrix{[]float64{}, 0, 0}, nil, ErrBadProductDimension},
	}

	for tcIndex, tcValue := range testCases {
		tcValue, tcIndex := tcValue, tcIndex // capture range variables
		t.Run(fmt.Sprintf("[%d] %+v", tcIndex, tcValue), func(t *testing.T) {
			t.Parallel()

			err := tcValue.dMatrix.MatrixProduct(tcValue.aMatrix, tcValue.bMatrix)

			if tcValue.expectedError != nil {
				if !errors.Is(err, tcValue.expectedError) {
					t.Logf("Err should be %v but it's %v", tcValue.expectedError, err)
					t.Fail()
				}
				return
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

func TestMatrixProductSelf(t *testing.T) {
	t.Parallel()

	expectedValues := []float64{-1, -3, -5}
	aMatrix := &Matrix{[]float64{0, 1, 2, 3, 4, 5}, 3, 2}
	bMatrix := &Matrix{[]float64{0, -1}, 2, 1}

	err := aMatrix.MatrixProduct(aMatrix, bMatrix)

	if !errors.Is(err, nil) {
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

func TestMultiply(t *testing.T) {
	testCases := []struct {
		aMatrix, bMatrix, dMatrix *Matrix
		expectedValues            []float64
		expectedError             error
	}{
		{&Matrix{[]float64{0, 1, 2, 3}, 2, 2}, &Matrix{[]float64{0, 1, 2, 3}, 2, 2}, &Matrix{[]float64{}, 0, 0}, []float64{0, 1, 4, 9}, nil},
		{&Matrix{[]float64{0, 1, 2, 3}, 2, 2}, nil, &Matrix{[]float64{}, 0, 0}, nil, ErrNilMatrix},
		{nil, &Matrix{[]float64{0, 1, 2, 3}, 2, 2}, &Matrix{[]float64{}, 0, 0}, nil, ErrNilMatrix},
		{&Matrix{[]float64{0, 1, 2, 3}, 2, 2}, &Matrix{[]float64{0, 1}, 1, 2}, &Matrix{[]float64{}, 0, 0}, nil, ErrDifferentDimension},
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
				return
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

	if !errors.Is(err, nil) {
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

func TestSubtract(t *testing.T) {
	testCases := []struct {
		aMatrix, bMatrix, dMatrix *Matrix
		expectedValues            []float64
		expectedError             error
	}{
		{&Matrix{[]float64{1, 2, 3, 4}, 2, 2}, &Matrix{[]float64{0, 1, 2, 3}, 2, 2}, &Matrix{[]float64{}, 0, 0}, []float64{1, 1, 1, 1}, nil},
		{&Matrix{[]float64{0, 1, 2, 3}, 2, 2}, nil, &Matrix{[]float64{}, 0, 0}, nil, ErrNilMatrix},
		{nil, &Matrix{[]float64{0, 1, 2, 3}, 2, 2}, &Matrix{[]float64{}, 0, 0}, nil, ErrNilMatrix},
		{&Matrix{[]float64{0, 1, 2, 3}, 2, 2}, &Matrix{[]float64{0, 1}, 1, 2}, &Matrix{[]float64{}, 0, 0}, nil, ErrDifferentDimension},
	}

	for tcIndex, tcValue := range testCases {
		tcValue, tcIndex := tcValue, tcIndex // capture range variables
		t.Run(fmt.Sprintf("[%d] %+v", tcIndex, tcValue), func(t *testing.T) {
			t.Parallel()

			err := tcValue.dMatrix.Subtract(tcValue.aMatrix, tcValue.bMatrix)

			if tcValue.expectedError != nil {
				if !errors.Is(err, tcValue.expectedError) {
					t.Logf("Err should be %v but it's %v", tcValue.expectedError, err)
					t.Fail()
				}
				return
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

func TestSubtractSelf(t *testing.T) {
	t.Parallel()

	expectedValues := []float64{1, 1, 1, 1}
	aMatrix := &Matrix{[]float64{1, 2, 3, 4}, 2, 2}
	bMatrix := &Matrix{[]float64{0, 1, 2, 3}, 2, 2}

	err := aMatrix.Subtract(aMatrix, bMatrix)

	if !errors.Is(err, nil) {
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

func TestTranspose(t *testing.T) {
	testCases := []struct {
		aMatrix                    *Matrix
		expectedValues             []float64
		expectedRows, expectedCols int
		expectedError              error
	}{
		{&Matrix{[]float64{0, 1, 2, 3, 4, 5}, 3, 2}, []float64{0, 2, 4, 1, 3, 5}, 2, 3, nil},
		{&Matrix{[]float64{0, 1, 2, 3, 4, 5}, 2, 3}, []float64{0, 3, 1, 4, 2, 5}, 3, 2, nil},
		{&Matrix{[]float64{0, 1, 2}, 1, 3}, []float64{0, 1, 2}, 3, 1, nil},
		{nil, nil, 0, 0, ErrNilMatrix},
	}

	for tcIndex, tcValue := range testCases {
		tcValue, tcIndex := tcValue, tcIndex // capture range variables
		t.Run(fmt.Sprintf("[%d] %+v", tcIndex, tcValue), func(t *testing.T) {
			t.Parallel()

			err := tcValue.aMatrix.Transpose(tcValue.aMatrix)

			if tcValue.expectedError != nil {
				if !errors.Is(err, tcValue.expectedError) {
					t.Logf("Err should be %v but it's %v", tcValue.expectedError, err)
					t.Fail()
				}
				return
			}

			if tcValue.expectedValues != nil {
				if len(tcValue.aMatrix.values) != len(tcValue.expectedValues) {
					t.Logf("Lenght of the matrix should be %d but it's %d", len(tcValue.expectedValues), len(tcValue.aMatrix.values))
					t.Fail()
				}

				for i, v := range tcValue.aMatrix.values {
					if v != tcValue.expectedValues[i] {
						t.Logf("Value of the matrix should be %f but it's %f", tcValue.expectedValues[i], v)
						t.Fail()
					}
				}

				if tcValue.aMatrix.rows != tcValue.expectedRows {
					t.Logf("Number of rows should be %d but it's %d", tcValue.expectedRows, tcValue.aMatrix.rows)
				}

				if tcValue.aMatrix.columns != tcValue.expectedCols {
					t.Logf("Number of columns should be %d but it's %d", tcValue.expectedCols, tcValue.aMatrix.columns)
				}
			}
		})
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
