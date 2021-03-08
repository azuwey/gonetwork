package matrix

// Matrix represents a mathematical matrix.
type Matrix struct {
	Values        []float64
	rows, columns int
}

// New creates a new Matrix with r rows and c columns, the data must be arranged in row-major order.
// If `v == nil`, a new slice will be allocated with `r * c` size,
// else if the lenght of the `data` is `r * c` it will be used as values otherwise it will return an error.
// It will also return an error if `r <= 0` of `c <= 0`.
func New(r, c int, v []float64) (*Matrix, error) {
	if r <= 0 || c <= 0 {
		return nil, ErrZeroRowOrCol
	}

	if v == nil {
		v = make([]float64, r*c)
	}

	if len(v) != r*c {
		return nil, ErrDataLength
	}

	m := &Matrix{make([]float64, r*c), r, c}
	copy(m.Values, v)

	return m, nil
}

// Add adds `a` and `b` element-wise, placing the result in the receiver.
// It will return an error if the two matrices do not have the same dimensions.
// It will also return an error if `b == nil` or `a == nil`.
func (m *Matrix) Add(a, b *Matrix) error {
	if a == nil || b == nil {
		return ErrNilMatrix
	}

	aRows, aCols := a.Dimensions()
	bRows, bCols := b.Dimensions()
	if aRows != bRows || aCols != bCols {
		return ErrDifferentDimesion
	}

	aVals, bVals := make([]float64, aRows*aCols), make([]float64, bRows*bCols)
	copy(aVals, a.Values)
	copy(bVals, b.Values)

	m.rows = aRows
	m.columns = aCols
	m.Values = make([]float64, m.rows*m.columns)

	for i := range m.Values {
		m.Values[i] = aVals[i] + bVals[i]
	}

	return nil
}

// Dimensions returns the number of rows and columns in the matrix.
func (m *Matrix) Dimensions() (int, int) {
	return m.rows, m.columns
}

// Multiply performs element-wise multiplication of `a` and `b`, placing the result in the receiver.
// It will return an error if the two matrices do not have the same dimensions.
// It will also return an error if `b == nil` or `a == nil`.
func (m *Matrix) Multiply(a, b *Matrix) error {
	if a == nil || b == nil {
		return ErrNilMatrix
	}

	aRows, aCols := a.Dimensions()
	bRows, bCols := b.Dimensions()
	if aRows != bRows || aCols != bCols {
		return ErrDifferentDimesion
	}

	aVals, bVals := make([]float64, aRows*aCols), make([]float64, bRows*bCols)
	copy(aVals, a.Values)
	copy(bVals, b.Values)

	m.rows = aRows
	m.columns = aCols
	m.Values = make([]float64, m.rows*m.columns)

	for i := range m.Values {
		m.Values[i] = aVals[i] * bVals[i]
	}

	return nil
}

// Product performs matrix multiplication of `a` and `b`, placing the result in the receiver.
// It will return an error if the number of columns in the `a` not equal with the number of rows in the `b`.
// It will also return an error if `b == nil` or `a == nil`.
func (m *Matrix) Product(a, b *Matrix) error {
	if a == nil || b == nil {
		return ErrNilMatrix
	}

	aRows, aCols := a.Dimensions()
	bRows, bCols := b.Dimensions()
	if aCols != bRows {
		return ErrBadProductDimesion
	}

	aVals, bVals := make([]float64, aRows*aCols), make([]float64, bRows*bCols)
	copy(aVals, a.Values)
	copy(bVals, b.Values)

	m.rows = aRows
	m.columns = bCols
	m.Values = make([]float64, m.rows*m.columns)

	for mIndex := range m.Values {
		for bIndex, value := range bVals {
			m.Values[mIndex] += value * aVals[bIndex+(mIndex*aCols)]
		}
	}

	return nil
}

// Scale multiplies the elements of a by f, placing the result in the receiver.
// It will return an error if `a == nil`.
func (m *Matrix) Scale(f float64, a *Matrix) error {
	if a == nil {
		return ErrNilMatrix
	}

	aVals := make([]float64, a.rows*a.columns)
	copy(aVals, a.Values)

	m.rows, m.columns = a.Dimensions()
	m.Values = make([]float64, m.rows*m.columns)

	for i := range m.Values {
		m.Values[i] = f * aVals[i]
	}

	return nil
}
