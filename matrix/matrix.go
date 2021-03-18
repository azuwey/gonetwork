package matrix

// Matrix represents a mathematical matrix.
type Matrix struct {
	values        []float64
	rows, columns int
}

// ApplyFn represents a function that is applied to the matrix when Apply is called
type ApplyFn func(v float64, r, c int) float64

// New creates a new Matrix with "r" rows and "c" columns, the "v" must be arranged in row-major order.
// If "v == nil", a new slice will be allocated with "r * c" size.
// If the length of the "v" is "r * c" it will be used as the underlaying slice otherwise it will return an error.
// It will also return an error if "r <= 0" of "c <= 0".
func New(r, c int, v []float64) (*Matrix, error) {
	if r <= 0 {
		return nil, ErrZeroRow
	}

	if c <= 0 {
		return nil, ErrZeroCol
	}

	if v == nil {
		v = make([]float64, r*c)
	}

	if len(v) != r*c {
		return nil, ErrDataLength
	}

	m := &Matrix{v, r, c}

	return m, nil
}

// Copy creates a new Matrix with the same properties as the given Matrix.
// It will return an error if "a == nil".
func Copy(a *Matrix) (*Matrix, error) {
	if a == nil {
		return nil, ErrNilMatrix
	}

	values := make([]float64, a.rows*a.columns)
	mCopy := &Matrix{values, a.rows, a.columns}
	copy(mCopy.values, a.values)
	return mCopy, nil
}

// Add adds "a" and "b" element-wise, placing the result in the receiver.
// It will return an error if the two matrices do not have the same dimensions.
// It will also return an error if "a == nil" or "b == nil".
func (m *Matrix) Add(a, b *Matrix) error {
	if a == nil || b == nil {
		return ErrNilMatrix
	}

	aRows, aCols := a.Dimensions()
	bRows, bCols := b.Dimensions()
	if aRows != bRows || aCols != bCols {
		return ErrDifferentDimensions
	}

	aVals, bVals := make([]float64, aRows*aCols), make([]float64, bRows*bCols)
	copy(aVals, a.values)
	copy(bVals, b.values)

	m.rows = aRows
	m.columns = aCols
	m.values = make([]float64, m.rows*m.columns)

	for i := range m.values {
		m.values[i] = aVals[i] + bVals[i]
	}

	return nil
}

// Apply applies the function "fn" to each of the elements of "a", placing the resulting matrix in the receiver.
// The function "fn" takes the value of the element, the index of the row and the column, and it returns a new value for that element.
// It will return an error if "fn == nil" or "a == nil".
func (m *Matrix) Apply(fn ApplyFn, a *Matrix) error {
	if fn == nil {
		return ErrNilFunction
	}

	if a == nil {
		return ErrNilMatrix
	}

	aRows, aCols := a.Dimensions()
	aVals := make([]float64, aRows*aCols)
	copy(aVals, a.values)

	m.rows, m.columns = aRows, aCols
	m.values = make([]float64, m.rows*m.columns)

	for i := range m.values {
		r := i / m.columns
		m.values[i] = fn(aVals[i], r, i-(r*m.columns))
	}

	return nil
}

// At returns the element at row "r", column "c".
// Indexing is zero-based, so the first row will be at "0" and the last row will be at "numberOfRows - 1" same for the columns.
// It will return an error if "r" bigger than the number of rows or "c" is bigger than the number columns.
func (m *Matrix) At(r, c int) (float64, error) {
	if r > m.rows-1 {
		return 0, ErrRowOutOfBounds
	}

	if c > m.columns-1 {
		return 0, ErrColOutOfBounds
	}

	return m.values[r*m.columns+c], nil
}

// Dimensions returns the number of rows and columns in the matrix.
func (m *Matrix) Dimensions() (int, int) {
	return m.rows, m.columns
}

// MatrixProduct performs matrix multiplication of "a" and "b", placing the result in the receiver.
// It will return an error if the number of columns in "a" not equal with the number of rows in "b".
// It will also return an error if "b == nil" or "a == nil".
func (m *Matrix) MatrixProduct(a, b *Matrix) error {
	if a == nil || b == nil {
		return ErrNilMatrix
	}

	aRows, aCols := a.Dimensions()
	bRows, bCols := b.Dimensions()
	if aCols != bRows {
		return ErrBadProductDimension
	}

	aVals, bVals := make([]float64, aRows*aCols), make([]float64, bRows*bCols)
	copy(aVals, a.values)
	copy(bVals, b.values)

	m.rows = aRows
	m.columns = bCols
	m.values = make([]float64, m.rows*m.columns)

	for mIndex := range m.values {
		for bIndex := 0; bIndex < bRows; bIndex++ {
			m.values[mIndex] += aVals[((mIndex/bCols)*aCols)+bIndex] * bVals[(bIndex*bCols)+(mIndex%bCols)]
		}
	}

	return nil
}

// Multiply performs element-wise multiplication of "a" and "b", placing the result in the receiver.
// It will return an error if the two matrices does not have the same dimensions.
// It will also return an error if "b == nil" or "a == nil".
func (m *Matrix) Multiply(a, b *Matrix) error {
	if a == nil || b == nil {
		return ErrNilMatrix
	}

	aRows, aCols := a.Dimensions()
	bRows, bCols := b.Dimensions()
	if aRows != bRows || aCols != bCols {
		return ErrDifferentDimensions
	}

	aVals, bVals := make([]float64, aRows*aCols), make([]float64, bRows*bCols)
	copy(aVals, a.values)
	copy(bVals, b.values)

	m.rows = aRows
	m.columns = aCols
	m.values = make([]float64, m.rows*m.columns)

	for i := range m.values {
		m.values[i] = aVals[i] * bVals[i]
	}

	return nil
}

// Raw returns the underlying slice.
// Changes on this slice will be refrected on the matrix.
func (m *Matrix) Raw() []float64 {
	return m.values
}

// Scale multiplies the elements of "a" by "f", placing the result in the receiver.
// It will return an error if "a == nil".
func (m *Matrix) Scale(f float64, a *Matrix) error {
	if a == nil {
		return ErrNilMatrix
	}

	aRows, aCols := a.Dimensions()
	aVals := make([]float64, aRows*aCols)
	copy(aVals, a.values)

	m.rows, m.columns = aRows, aCols
	m.values = make([]float64, m.rows*m.columns)

	for i := range m.values {
		m.values[i] = f * aVals[i]
	}

	return nil
}

// Subtract subtracts "a" and "b" element-wise, placing the result in the receiver, in the order of "a - b"
// It will return an error if the two matrices do not have the same dimensions.
// It will also return an error if "b == nil" or "a == nil".
func (m *Matrix) Subtract(a, b *Matrix) error {
	if a == nil || b == nil {
		return ErrNilMatrix
	}

	aRows, aCols := a.Dimensions()
	bRows, bCols := b.Dimensions()
	if aRows != bRows || aCols != bCols {
		return ErrDifferentDimensions
	}

	aVals, bVals := make([]float64, aRows*aCols), make([]float64, bRows*bCols)
	copy(aVals, a.values)
	copy(bVals, b.values)

	m.rows = aRows
	m.columns = aCols
	m.values = make([]float64, m.rows*m.columns)

	for i := range m.values {
		m.values[i] = aVals[i] - bVals[i]
	}

	return nil
}

// Transpose switches the row and column indices of the matrix, placing the result in the receiver.
// It will return an error if "a == nil".
func (m *Matrix) Transpose(a *Matrix) error {
	if a == nil {
		return ErrNilMatrix
	}

	aRows, aCols := a.Dimensions()
	aVals := make([]float64, aRows*aCols)
	copy(aVals, a.values)

	m.rows, m.columns = aCols, aRows
	m.values = make([]float64, m.rows*m.columns)

	for i := range m.values {
		m.values[i] = aVals[(i%aRows*aCols)+(i/aRows)]
	}

	return nil
}

// Values returns the values of the matrix in a "[][]float64" format.
func (m *Matrix) Values() [][]float64 {
	values := make([][]float64, m.rows)

	for rIndex := range values {
		values[rIndex] = make([]float64, m.columns)
		for cIndex := range values[rIndex] {
			values[rIndex][cIndex] = m.values[rIndex*m.columns+cIndex]
		}
	}

	return values
}
