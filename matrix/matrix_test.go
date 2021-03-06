package matrix

import (
	"math/rand"
	"testing"
)

func TestNew(t *testing.T) {
	testCases := []struct {
		name                   string
		rows, columns          int
		values, expectedValues []float64
		expectedError          error
	}{
		{"Normal", 2, 2, []float64{0, 1, 2, 3}, []float64{0, 1, 2, 3}, nil},
		{"Zero", 2, 2, nil, []float64{0, 0, 0, 0}, nil},
		{"ErrZeroRow", 0, 0, nil, nil, ErrZeroRow},
		{"ErrZeroCol", 1, 0, nil, nil, ErrZeroCol},
		{"ErrDataLength", 2, 2, []float64{0}, nil, ErrDataLength},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			m, err := New(tc.rows, tc.columns, tc.values)
			if tc.expectedError != nil {
				if err != tc.expectedError {
					t.Errorf("Expected error is %v, but got %v", tc.expectedError, err)
				}
			} else if err != nil {
				t.Errorf("Expected error is %v, but got %v", nil, err)
			} else if m == nil {
				t.Error("Matrix should not be nil")
			} else {
				for idx, v := range m.Values {
					if v != tc.expectedValues[idx] {
						t.Errorf("Expected value is %f, but got %f", tc.expectedValues[idx], v)
					}
				}

				if m.Rows != tc.rows {
					t.Errorf("Expected number of rows is %d, but got %d", tc.rows, m.Rows)
				}

				if m.Columns != tc.columns {
					t.Errorf("Expected number of columns is %d, but got %d", tc.columns, m.Columns)
				}
			}
		})
	}

	t.Run("Data reflection", func(t *testing.T) {
		t.Parallel()

		r := rand.New(rand.NewSource(0))
		v := []float64{r.Float64()}
		m, err := New(1, 1, v)
		if err != nil {
			t.Errorf("Expected error is %v, but got %v", nil, err)
		} else {
			tmp := v[0]
			v[0] = r.Float64()
			if m.Values[0] == v[0] {
				t.Errorf("Expected value is %f, but got %f", tmp, m.Values[0])
			}
		}
	})
}

func TestCopy(t *testing.T) {
	testCases := []struct {
		name          string
		matrix        *Matrix
		expectedError error
	}{
		{"Normal", &Matrix{[]float64{1, 2, 3, 4, 5}, 1, 5}, nil},
		{"ErrNilMatrix", nil, ErrNilMatrix},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			m, err := Copy(tc.matrix)
			if tc.expectedError != nil {
				if err != tc.expectedError {
					t.Errorf("Expected error is %v, but got %v", tc.expectedError, err)
				}
			} else if err != nil {
				t.Errorf("Expected error is %v, but got %v", nil, err)
			} else if m == nil {
				t.Error("Matrix should not be nil")
			} else {
				for idx, v := range m.Values {
					if v != tc.matrix.Values[idx] {
						t.Errorf("Expected value is %f, but got %f", tc.matrix.Values[idx], v)
					}
				}
			}
		})
	}

	t.Run("Data reflection", func(t *testing.T) {
		t.Parallel()

		r := rand.New(rand.NewSource(0))
		v := []float64{r.Float64()}
		a, _ := New(1, 1, v)
		b, err := Copy(a)
		if err != nil {
			t.Errorf("Expected error is %v, but got %v", nil, err)
		} else {
			a.Values[0] = r.Float64()
			if b.Values[0] != v[0] {
				t.Errorf("Expected value is %f, but got %f", v[0], a.Values[0])
			}
		}
	})
}

func TestAdd(t *testing.T) {
	testCases := []struct {
		name           string
		a, b           *Matrix
		expectedValues []float64
		expectedError  error
	}{
		{"Normal", &Matrix{[]float64{1, 2}, 1, 2}, &Matrix{[]float64{1, 2}, 1, 2}, []float64{2, 4}, nil},
		{"ErrNilMatrix a", nil, &Matrix{[]float64{1, 2}, 1, 2}, nil, ErrNilMatrix},
		{"ErrNilMatrix b", &Matrix{[]float64{1, 2}, 1, 2}, nil, nil, ErrNilMatrix},
		{"ErrDifferentDimensions", &Matrix{[]float64{1, 2}, 1, 2}, &Matrix{[]float64{1, 2}, 2, 1}, nil, ErrDifferentDimensions},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			m := &Matrix{}
			err := m.Add(tc.a, tc.b)

			if tc.expectedError != nil {
				if err != tc.expectedError {
					t.Errorf("Expected error is %v, but got %v", tc.expectedError, err)
				}
			} else if err != nil {
				t.Errorf("Expected error is %v, but got %v", nil, err)
			} else if m == nil {
				t.Error("Matrix should not be nil")
			} else {
				for idx, v := range m.Values {
					if v != tc.expectedValues[idx] {
						t.Errorf("Expected value is %f, but got %f", tc.expectedValues[idx], v)
					}
				}
			}
		})
	}

	t.Run("Data reflection", func(t *testing.T) {
		t.Parallel()

		r := rand.New(rand.NewSource(0))
		v := []float64{r.Float64()}
		a, _ := New(1, 1, v)
		b, _ := New(1, 1, v)
		err := b.Add(a, b)
		if err != nil {
			t.Errorf("Expected error is %v, but got %v", nil, err)
		} else {
			a.Values[0] = r.Float64()
			if b.Values[0] != v[0]*2 {
				t.Errorf("Expected value is %f, but got %f", v[0]*2, b.Values[0])
			}
		}
	})
}

func TestApply(t *testing.T) {
	addFn := func(v float64, _ int, _ []float64) float64 {
		return v * 2
	}
	testCases := []struct {
		name           string
		matrix         *Matrix
		function       ApplyFn
		expectedValues []float64
		expectedError  error
	}{
		{"Normal", &Matrix{[]float64{1, 2}, 1, 2}, addFn, []float64{2, 4}, nil},
		{"ErrNilFunction", &Matrix{[]float64{1, 2}, 1, 2}, nil, nil, ErrNilFunction},
		{"ErrNilMatrix", nil, addFn, nil, ErrNilMatrix},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			m := &Matrix{}
			err := m.Apply(tc.function, tc.matrix)

			if tc.expectedError != nil {
				if err != tc.expectedError {
					t.Errorf("Expected error is %v, but got %v", tc.expectedError, err)
				}
			} else if err != nil {
				t.Errorf("Expected error is %v, but got %v", nil, err)
			} else if m == nil {
				t.Error("Matrix should not be nil")
			} else {
				for idx, v := range m.Values {
					if v != tc.expectedValues[idx] {
						t.Errorf("Expected value is %f, but got %f", tc.expectedValues[idx], v)
					}
				}
			}
		})
	}

	t.Run("Data reflection", func(t *testing.T) {
		t.Parallel()

		r := rand.New(rand.NewSource(0))
		v := []float64{r.Float64()}
		a, _ := New(1, 1, v)
		err := a.Apply(func(v float64, _ int, _ []float64) float64 {
			return v * 2
		}, a)
		if err != nil {
			t.Errorf("Expected error is %v, but got %v", nil, err)
		} else {
			tmp := v[0]
			v[0] = r.Float64()
			if a.Values[0] == v[0] {
				t.Errorf("Expected value is %f, but got %f", tmp, a.Values[0])
			}
		}
	})
}

func TestAt(t *testing.T) {
	type dimension struct {
		row, column int
	}
	testCases := []struct {
		name           string
		matrix         *Matrix
		dimensions     []dimension
		expectedValues []float64
		expectedError  error
	}{
		{"Normal [3][2]", &Matrix{[]float64{1, 2, 3, 4, 5, 6}, 3, 2}, []dimension{{0, 0}, {0, 1}, {1, 0}, {1, 1}, {2, 0}, {2, 1}}, []float64{1, 2, 3, 4, 5, 6}, nil},
		{"Normal [2][3]", &Matrix{[]float64{1, 2, 3, 4, 5, 6}, 2, 3}, []dimension{{0, 0}, {0, 1}, {0, 2}, {1, 0}, {1, 1}, {1, 2}}, []float64{1, 2, 3, 4, 5, 6}, nil},
		{"ErrRowOutOfBounds", &Matrix{[]float64{1, 2}, 1, 2}, []dimension{{1, 0}}, nil, ErrRowOutOfBounds},
		{"ErrColOutOfBounds", &Matrix{[]float64{1, 2}, 1, 2}, []dimension{{0, 3}}, nil, ErrColOutOfBounds},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			for idx, d := range tc.dimensions {
				v, err := tc.matrix.At(d.row, d.column)
				if tc.expectedError != nil {
					if err != tc.expectedError {
						t.Errorf("Expected error is %v, but got %v", tc.expectedError, err)
					}
				} else if err != nil {
					t.Errorf("Expected error is %v, but got %v", nil, err)
				} else {
					if v != tc.expectedValues[idx] {
						t.Errorf("Expected value is %f, but got %f", tc.expectedValues[idx], v)
					}
				}
			}
		})
	}
}

func TestMultiply(t *testing.T) {
	testCases := []struct {
		name           string
		a, b           *Matrix
		expectedValues []float64
		expectedError  error
	}{
		{"Normal", &Matrix{[]float64{0, 1, 2, 3}, 2, 2}, &Matrix{[]float64{0, 1, 2, 3}, 2, 2}, []float64{0, 1, 4, 9}, nil},
		{"ErrNilMatrix a", nil, &Matrix{[]float64{0, 1, 2, 3, 4, 5}, 3, 2}, nil, ErrNilMatrix},
		{"ErrNilMatrix b", &Matrix{[]float64{0, 1, 2, 3, 4, 5}, 2, 3}, nil, nil, ErrNilMatrix},
		{"ErrDifferentDimensions", &Matrix{[]float64{0, 1, 2, 3, 4, 5}, 2, 3}, &Matrix{[]float64{0, -1}, 1, 2}, nil, ErrDifferentDimensions},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			m := &Matrix{}
			err := m.Multiply(tc.a, tc.b)

			if tc.expectedError != nil {
				if err != tc.expectedError {
					t.Errorf("Expected error is %v, but got %v", tc.expectedError, err)
				}
			} else if err != nil {
				t.Errorf("Expected error is %v, but got %v", nil, err)
			} else if m == nil {
				t.Error("Matrix should not be nil")
			} else {
				for idx, v := range m.Values {
					if v != tc.expectedValues[idx] {
						t.Errorf("Expected value is %f, but got %f", tc.expectedValues[idx], v)
					}
				}
			}
		})
	}

	t.Run("Data reflection", func(t *testing.T) {
		t.Parallel()

		r := rand.New(rand.NewSource(0))
		v := []float64{r.Float64(), r.Float64()}
		a, _ := New(2, 1, v)
		b, _ := New(2, 1, v)
		err := a.Multiply(a, b)
		if err != nil {
			t.Errorf("Expected error is %v, but got %v", nil, err)
		} else {
			tmp := v[0]
			v[0] = r.Float64()
			if a.Values[0] == v[0] {
				t.Errorf("Expected value is %f, but got %f", tmp, a.Values[0])
			}
		}
	})
}

func TestProduct(t *testing.T) {
	testCases := []struct {
		name           string
		a, b           *Matrix
		expectedValues []float64
		expectedError  error
	}{
		{"Normal short", &Matrix{[]float64{0, 1, 2, 3, 4, 5}, 2, 3}, &Matrix{[]float64{0, 1, 2, 3, 4, 5}, 3, 2}, []float64{10, 13, 28, 40}, nil},
		{"Normal long", &Matrix{[]float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, 4, 3}, &Matrix{[]float64{0, -1, -2, -3, -4, -5}, 3, 2}, []float64{-10, -13, -28, -40, -46, -67, -64, -94}, nil},
		{"ErrNilMatrix a", nil, &Matrix{[]float64{0, 1, 2, 3, 4, 5}, 3, 2}, nil, ErrNilMatrix},
		{"ErrNilMatrix b", &Matrix{[]float64{0, 1, 2, 3, 4, 5}, 2, 3}, nil, nil, ErrNilMatrix},
		{"ErrBadProductDimension", &Matrix{[]float64{0, 1, 2, 3, 4, 5}, 2, 3}, &Matrix{[]float64{0, -1}, 1, 2}, nil, ErrBadProductDimension},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			m := &Matrix{}
			err := m.Product(tc.a, tc.b)

			if tc.expectedError != nil {
				if err != tc.expectedError {
					t.Errorf("Expected error is %v, but got %v", tc.expectedError, err)
				}
			} else if err != nil {
				t.Errorf("Expected error is %v, but got %v", nil, err)
			} else if m == nil {
				t.Error("Matrix should not be nil")
			} else {
				for idx, v := range m.Values {
					if v != tc.expectedValues[idx] {
						t.Errorf("Expected value is %f, but got %f", tc.expectedValues[idx], v)
					}
				}
			}
		})
	}

	t.Run("Data reflection", func(t *testing.T) {
		t.Parallel()

		r := rand.New(rand.NewSource(0))
		v := []float64{r.Float64(), r.Float64()}
		a, _ := New(1, 2, v)
		b, _ := New(2, 1, v)
		err := a.Product(a, b)
		if err != nil {
			t.Errorf("Expected error is %v, but got %v", nil, err)
		} else {
			tmp := v[0]
			v[0] = r.Float64()
			if a.Values[0] == v[0] {
				t.Errorf("Expected value is %f, but got %f", tmp, a.Values[0])
			}
		}
	})
}

func TestScale(t *testing.T) {
	testCases := []struct {
		name           string
		matrix         *Matrix
		scaler         float64
		expectedValues []float64
		expectedError  error
	}{
		{"Normal", &Matrix{[]float64{0, 1, 2, 3}, 2, 2}, 2, []float64{0, 2, 4, 6}, nil},
		{"ErrNilMatrix", nil, 0, nil, ErrNilMatrix},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			m := &Matrix{}
			err := m.Scale(tc.scaler, tc.matrix)

			if tc.expectedError != nil {
				if err != tc.expectedError {
					t.Errorf("Expected error is %v, but got %v", tc.expectedError, err)
				}
			} else if err != nil {
				t.Errorf("Expected error is %v, but got %v", nil, err)
			} else if m == nil {
				t.Error("Matrix should not be nil")
			} else {
				for idx, v := range m.Values {
					if v != tc.expectedValues[idx] {
						t.Errorf("Expected value is %f, but got %f", tc.expectedValues[idx], v)
					}
				}
			}
		})
	}

	t.Run("Data reflection", func(t *testing.T) {
		t.Parallel()

		r := rand.New(rand.NewSource(0))
		v := []float64{r.Float64()}
		a, _ := New(1, 1, v)
		scaler := 2.0
		err := a.Scale(scaler, a)
		if err != nil {
			t.Errorf("Expected error is %v, but got %v", nil, err)
		} else {
			tmp := v[0] * scaler
			v[0] = r.Float64()
			if a.Values[0] == v[0] {
				t.Errorf("Expected value is %f, but got %f", tmp, a.Values[0])
			}
		}
	})
}

func TestSubtract(t *testing.T) {
	testCases := []struct {
		name           string
		a, b           *Matrix
		expectedValues []float64
		expectedError  error
	}{
		{"Normal", &Matrix{[]float64{1, 2}, 1, 2}, &Matrix{[]float64{1, 2}, 1, 2}, []float64{0, 0}, nil},
		{"ErrNilMatrix a", nil, &Matrix{[]float64{1, 2}, 1, 2}, nil, ErrNilMatrix},
		{"ErrNilMatrix b", &Matrix{[]float64{1, 2}, 1, 2}, nil, nil, ErrNilMatrix},
		{"ErrDifferentDimensions", &Matrix{[]float64{1, 2}, 1, 2}, &Matrix{[]float64{1, 2}, 2, 1}, nil, ErrDifferentDimensions},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			m := &Matrix{}
			err := m.Subtract(tc.a, tc.b)

			if tc.expectedError != nil {
				if err != tc.expectedError {
					t.Errorf("Expected error is %v, but got %v", tc.expectedError, err)
				}
			} else if err != nil {
				t.Errorf("Expected error is %v, but got %v", nil, err)
			} else if m == nil {
				t.Error("Matrix should not be nil")
			} else {
				for idx, v := range m.Values {
					if v != tc.expectedValues[idx] {
						t.Errorf("Expected value is %f, but got %f", tc.expectedValues[idx], v)
					}
				}
			}
		})
	}

	t.Run("Data reflection", func(t *testing.T) {
		t.Parallel()

		r := rand.New(rand.NewSource(0))
		v := []float64{r.Float64()}
		a, _ := New(1, 1, v)
		b, _ := New(1, 1, v)
		err := b.Subtract(a, b)
		if err != nil {
			t.Errorf("Expected error is %v, but got %v", nil, err)
		} else {
			a.Values[0] = r.Float64()
			if b.Values[0] != 0.0 {
				t.Errorf("Expected value is %f, but got %f", 0.0, b.Values[0])
			}
		}
	})
}

func TestTranspose(t *testing.T) {
	testCases := []struct {
		name                       string
		matrix                     *Matrix
		expectedRows, expectedCols int
		expectedValues             []float64
		expectedError              error
	}{
		{"Normal", &Matrix{[]float64{0, 1, 2, 3, 4, 5}, 3, 2}, 2, 3, []float64{0, 3, 1, 4, 2, 5}, nil},
		{"ErrNilMatrix", nil, 0, 0, nil, ErrNilMatrix},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			m := &Matrix{}
			err := m.Transpose(tc.matrix)

			if tc.expectedError != nil {
				if err != tc.expectedError {
					t.Errorf("Expected error is %v, but got %v", tc.expectedError, err)
				}
			} else if err != nil {
				t.Errorf("Expected error is %v, but got %v", nil, err)
			} else if m == nil {
				t.Error("Matrix should not be nil")
			} else {
				for idx, v := range m.Values {
					if v != tc.expectedValues[idx] {
						t.Errorf("Expected value is %f, but got %f", tc.expectedValues[idx], v)
					}
				}

				if m.Rows != tc.expectedRows {
					t.Errorf("Expected number of rows is %d, but got %d", tc.expectedRows, m.Rows)
				}

				if m.Columns != tc.expectedCols {
					t.Errorf("Expected number of columns is %d, but got %d", tc.expectedCols, m.Columns)
				}
			}
		})
	}
}
