package matrix

// Matrix represents a mathematical Matrix.
type Matrix struct {
	Values  []float64 `json:"values"`  // Values of the matrix
	Rows    int       `json:"rows"`    // Number of rows
	Columns int       `json:"Columns"` // Number of columns
}

// ApplyFn represents a function that is applied to the matrix when Apply is called
type ApplyFn func(v float64, r, c int, s []float64) float64

// New creates a new Matrix with "r" rows and "c" columns, the "vals" must be arranged in row-major order.
// If "vals == nil", a new slice will be allocated with "r * c" size.
// If the length of the "vals" is "r * c" it will be used as the underlaying slice but the changes won't be reflected, otherwise it will return an error.
// It will also return an error if "r <= 0" of "c <= 0".
func New(r, c int, vals []float64) (*Matrix, error) {
	if r <= 0 {
		return nil, ErrZeroRow
	}

	if c <= 0 {
		return nil, ErrZeroCol
	}

	if vals == nil {
		vals = make([]float64, r*c)
	} else {
		tmp := make([]float64, len(vals))
		copy(tmp, vals)
		vals = tmp
	}

	if len(vals) != r*c {
		return nil, ErrDataLength
	}

	m := &Matrix{vals, r, c}

	return m, nil
}

// Copy creates a new Matrix with the same properties as the given Matrix.
// It will return an error if "m == nil".
func Copy(m *Matrix) (*Matrix, error) {
	if m == nil {
		return nil, ErrNilMatrix
	}

	vals := make([]float64, m.Rows*m.Columns)
	nMat := &Matrix{vals, m.Rows, m.Columns}
	copy(nMat.Values, m.Values)
	return nMat, nil
}

// Add adds "aMat" and "bMat" element-wise, placing the result in the receiver.
// It will return an error if the two matrices do not hase the same dimensions.
// It will also return an error if "aMat == nil" or "bMat == nil".
func (m *Matrix) Add(aMat, bMat *Matrix) error {
	if aMat == nil || bMat == nil {
		return ErrNilMatrix
	}

	aRows, aCols, bRows, bCols := aMat.Rows, aMat.Columns, bMat.Rows, bMat.Columns
	if aRows != bRows || aCols != bCols {
		return ErrDifferentDimensions
	}

	aVals, bVals := make([]float64, aRows*aCols), make([]float64, bRows*bCols)
	copy(aVals, aMat.Values)
	copy(bVals, bMat.Values)

	m.Rows = aRows
	m.Columns = aCols
	m.Values = make([]float64, m.Rows*m.Columns)

	for idx := range m.Values {
		m.Values[idx] = aVals[idx] + bVals[idx]
	}

	return nil
}

// Apply applies the function "fn" to each of the elements of "a", placing the resulting matrix in the receiver.
// The function "fn" takes the value of the element, the index of the row and the column, and it returns a new value for that element.
// It will return an error if "fn == nil" or "a == nil".
func (m *Matrix) Apply(fn ApplyFn, aMat *Matrix) error {
	if fn == nil {
		return ErrNilFunction
	}

	if aMat == nil {
		return ErrNilMatrix
	}

	aRows, aCols := aMat.Rows, aMat.Columns
	aVals := make([]float64, aRows*aCols)
	copy(aVals, aMat.Values)

	m.Rows, m.Columns = aRows, aCols
	m.Values = make([]float64, m.Rows*m.Columns)

	for idx := range m.Values {
		r := idx / m.Columns
		s := append(m.Values[:idx], aVals[idx:]...)
		m.Values[idx] = fn(aVals[idx], r, idx-(r*m.Columns), s)
	}

	return nil
}

// At returns the element at row "r", column "c".
// Indexing is zero-based, so the first row will be at "0" and the last row will be at "numberOfRows - 1" same for the columns.
// It will return an error if "r" bigger than the number of rows or "c" is bigger than the number columns.
func (m *Matrix) At(r, c int) (float64, error) {
	if r > m.Rows-1 {
		return 0, ErrRowOutOfBounds
	}

	if c > m.Columns-1 {
		return 0, ErrColOutOfBounds
	}

	return m.Values[r*m.Columns+c], nil
}

// Multiply performs element-wise multiplication of "a" and "b", placing the result in the receiver.
// It will return an error if the two matrices does not hase the same dimensions.
// It will also return an error if "b == nil" or "a == nil".
func (m *Matrix) Multiply(aMat, bMat *Matrix) error {
	if aMat == nil || bMat == nil {
		return ErrNilMatrix
	}

	aRows, aCols, bRows, bCols := aMat.Rows, aMat.Columns, bMat.Rows, bMat.Columns
	if aRows != bRows || aCols != bCols {
		return ErrDifferentDimensions
	}

	aVals, bVals := make([]float64, aRows*aCols), make([]float64, bRows*bCols)
	copy(aVals, aMat.Values)
	copy(bVals, bMat.Values)

	m.Rows = aRows
	m.Columns = aCols
	m.Values = make([]float64, m.Rows*m.Columns)

	for idx := range m.Values {
		m.Values[idx] = aVals[idx] * bVals[idx]
	}

	return nil
}

// Product performs matrix multiplication of "a" and "b", placing the result in the receiver.
// It will return an error if the number of columns in "a" not equal with the number of rows in "b".
// It will also return an error if "b == nil" or "a == nil".
func (m *Matrix) Product(aMat, bMat *Matrix) error {
	if aMat == nil || bMat == nil {
		return ErrNilMatrix
	}

	aRows, aCols, bRows, bCols := aMat.Rows, aMat.Columns, bMat.Rows, bMat.Columns
	if aCols != bRows {
		return ErrBadProductDimension
	}

	aVals, bVals := make([]float64, aRows*aCols), make([]float64, bRows*bCols)
	copy(aVals, aMat.Values)
	copy(bVals, bMat.Values)

	m.Rows = aRows
	m.Columns = bCols
	m.Values = make([]float64, m.Rows*m.Columns)

	for mIdx := range m.Values {
		for bIdx := 0; bIdx < bRows; bIdx++ {
			m.Values[mIdx] += aVals[((mIdx/bCols)*aCols)+bIdx] * bVals[(bIdx*bCols)+(mIdx%bCols)]
		}
	}

	return nil
}

// Scale multiplies the elements of "a" by "s", placing the result in the receiver.
// It will return an error if "a == nil".
func (m *Matrix) Scale(s float64, aMat *Matrix) error {
	if aMat == nil {
		return ErrNilMatrix
	}

	aRows, aCols := aMat.Rows, aMat.Columns
	aVals := make([]float64, aRows*aCols)
	copy(aVals, aMat.Values)

	m.Rows, m.Columns = aRows, aCols
	m.Values = make([]float64, m.Rows*m.Columns)

	for idx := range m.Values {
		m.Values[idx] = s * aVals[idx]
	}

	return nil
}

// Subtract subtracts "a" and "b" element-wise, placing the result in the receiver, in the order of "a - b"
// It will return an error if the two matrices do not hase the same dimensions.
// It will also return an error if "b == nil" or "a == nil".
func (m *Matrix) Subtract(aMat, bMat *Matrix) error {
	if aMat == nil || bMat == nil {
		return ErrNilMatrix
	}

	aRows, aCols, bRows, bCols := aMat.Rows, aMat.Columns, bMat.Rows, bMat.Columns
	if aRows != bRows || aCols != bCols {
		return ErrDifferentDimensions
	}

	aVals, bVals := make([]float64, aRows*aCols), make([]float64, bRows*bCols)
	copy(aVals, aMat.Values)
	copy(bVals, bMat.Values)

	m.Rows = aRows
	m.Columns = aCols
	m.Values = make([]float64, m.Rows*m.Columns)

	for idx := range m.Values {
		m.Values[idx] = aVals[idx] - bVals[idx]
	}

	return nil
}

// Transpose switches the row and column indices of the matrix, placing the result in the receiver.
// It will return an error if "a == nil".
func (m *Matrix) Transpose(aMat *Matrix) error {
	if aMat == nil {
		return ErrNilMatrix
	}

	aRows, aCols := aMat.Rows, aMat.Columns
	aVals := make([]float64, aRows*aCols)
	copy(aVals, aMat.Values)

	m.Rows, m.Columns = aCols, aRows
	m.Values = make([]float64, m.Rows*m.Columns)

	for idx := range m.Values {
		m.Values[idx] = aVals[(idx%aCols*aRows)+(idx/aCols)]
	}

	return nil
}
