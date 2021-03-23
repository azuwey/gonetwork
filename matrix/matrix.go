package matrix

// Matrix represents a mathematical matrix.
type Matrix struct {
	vs   []float64 // Values of the matrix
	r, c int       // Number of rows and columns
}

// ApplyFn represents a function that is applied to the matrix when Apply is called
type ApplyFn func(v float64, r, c int, s []float64) float64

// New creates a new Matrix with "r" rows and "c" columns, the "v" must be arranged in row-major order.
// If "v == nil", a new slice will be allocated with "r * c" size.
// If the length of the "v" is "r * c" it will be used as the underlaying slice but the changes won't be reflected, otherwise it will return an error.
// It will also return an error if "r <= 0" of "c <= 0".
func New(r, c int, vs []float64) (*Matrix, error) {
	if r <= 0 {
		return nil, ErrZeroRow
	}

	if c <= 0 {
		return nil, ErrZeroCol
	}

	if vs == nil {
		vs = make([]float64, r*c)
	} else {
		tmp := make([]float64, len(vs))
		copy(tmp, vs)
		vs = tmp
	}

	if len(vs) != r*c {
		return nil, ErrDataLength
	}

	m := &Matrix{vs, r, c}

	return m, nil
}

// Copy creates a new Matrix with the same properties as the given Matrix.
// It will return an error if "a == nil".
func Copy(a *Matrix) (*Matrix, error) {
	if a == nil {
		return nil, ErrNilMatrix
	}

	vs := make([]float64, a.r*a.c)
	cp := &Matrix{vs, a.r, a.c}
	copy(cp.vs, a.vs)
	return cp, nil
}

// Add adds "a" and "b" element-wise, placing the result in the receiver.
// It will return an error if the two matrices do not hase the same dimensions.
// It will also return an error if "a == nil" or "b == nil".
func (m *Matrix) Add(a, b *Matrix) error {
	if a == nil || b == nil {
		return ErrNilMatrix
	}

	ar, ac := a.Dimensions()
	br, bc := b.Dimensions()
	if ar != br || ac != bc {
		return ErrDifferentDimensions
	}

	as, bs := make([]float64, ar*ac), make([]float64, br*bc)
	copy(as, a.vs)
	copy(bs, b.vs)

	m.r = ar
	m.c = ac
	m.vs = make([]float64, m.r*m.c)

	for idx := range m.vs {
		m.vs[idx] = as[idx] + bs[idx]
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

	ar, ac := a.Dimensions()
	as := make([]float64, ar*ac)
	copy(as, a.vs)

	m.r, m.c = ar, ac
	m.vs = make([]float64, m.r*m.c)

	for idx := range m.vs {
		r := idx / m.c
		s := append(m.vs[:idx], as[idx:]...)
		m.vs[idx] = fn(as[idx], r, idx-(r*m.c), s)
	}

	return nil
}

// At returns the element at row "r", column "c".
// Indexing is zero-based, so the first row will be at "0" and the last row will be at "numberOfRows - 1" same for the columns.
// It will return an error if "r" bigger than the number of rows or "c" is bigger than the number columns.
func (m *Matrix) At(r, c int) (float64, error) {
	if r > m.r-1 {
		return 0, ErrRowOutOfBounds
	}

	if c > m.c-1 {
		return 0, ErrColOutOfBounds
	}

	return m.vs[r*m.c+c], nil
}

// Dimensions returns the number of rows and columns in the matrix.
func (m *Matrix) Dimensions() (int, int) {
	return m.r, m.c
}

// MatrixProduct performs matrix multiplication of "a" and "b", placing the result in the receiver.
// It will return an error if the number of columns in "a" not equal with the number of rows in "b".
// It will also return an error if "b == nil" or "a == nil".
func (m *Matrix) MatrixProduct(a, b *Matrix) error {
	if a == nil || b == nil {
		return ErrNilMatrix
	}

	ar, ac := a.Dimensions()
	br, bc := b.Dimensions()
	if ac != br {
		return ErrBadProductDimension
	}

	as, bvs := make([]float64, ar*ac), make([]float64, br*bc)
	copy(as, a.vs)
	copy(bvs, b.vs)

	m.r = ar
	m.c = bc
	m.vs = make([]float64, m.r*m.c)

	for mIdx := range m.vs {
		for bIdx := 0; bIdx < br; bIdx++ {
			m.vs[mIdx] += as[((mIdx/bc)*ac)+bIdx] * bvs[(bIdx*bc)+(mIdx%bc)]
		}
	}

	return nil
}

// Multiply performs element-wise multiplication of "a" and "b", placing the result in the receiver.
// It will return an error if the two matrices does not hase the same dimensions.
// It will also return an error if "b == nil" or "a == nil".
func (m *Matrix) Multiply(a, b *Matrix) error {
	if a == nil || b == nil {
		return ErrNilMatrix
	}

	ar, ac := a.Dimensions()
	br, bc := b.Dimensions()
	if ar != br || ac != bc {
		return ErrDifferentDimensions
	}

	as, bvs := make([]float64, ar*ac), make([]float64, br*bc)
	copy(as, a.vs)
	copy(bvs, b.vs)

	m.r = ar
	m.c = ac
	m.vs = make([]float64, m.r*m.c)

	for idx := range m.vs {
		m.vs[idx] = as[idx] * bvs[idx]
	}

	return nil
}

// Raw returns the underlying slice.
// Changes on this slice will be refrected on the matrix.
func (m *Matrix) Raw() []float64 {
	return m.vs
}

// Scale multiplies the elements of "a" by "s", placing the result in the receiver.
// It will return an error if "a == nil".
func (m *Matrix) Scale(s float64, a *Matrix) error {
	if a == nil {
		return ErrNilMatrix
	}

	ar, ac := a.Dimensions()
	as := make([]float64, ar*ac)
	copy(as, a.vs)

	m.r, m.c = ar, ac
	m.vs = make([]float64, m.r*m.c)

	for idx := range m.vs {
		m.vs[idx] = s * as[idx]
	}

	return nil
}

// Subtract subtracts "a" and "b" element-wise, placing the result in the receiver, in the order of "a - b"
// It will return an error if the two matrices do not hase the same dimensions.
// It will also return an error if "b == nil" or "a == nil".
func (m *Matrix) Subtract(a, b *Matrix) error {
	if a == nil || b == nil {
		return ErrNilMatrix
	}

	ar, ac := a.Dimensions()
	br, bc := b.Dimensions()
	if ar != br || ac != bc {
		return ErrDifferentDimensions
	}

	as, bvs := make([]float64, ar*ac), make([]float64, br*bc)
	copy(as, a.vs)
	copy(bvs, b.vs)

	m.r = ar
	m.c = ac
	m.vs = make([]float64, m.r*m.c)

	for idx := range m.vs {
		m.vs[idx] = as[idx] - bvs[idx]
	}

	return nil
}

// Transpose switches the row and column indices of the matrix, placing the result in the receiver.
// It will return an error if "a == nil".
func (m *Matrix) Transpose(a *Matrix) error {
	if a == nil {
		return ErrNilMatrix
	}

	ar, ac := a.Dimensions()
	as := make([]float64, ar*ac)
	copy(as, a.vs)

	m.r, m.c = ac, ar
	m.vs = make([]float64, m.r*m.c)

	for idx := range m.vs {
		m.vs[idx] = as[(idx%ar*ac)+(idx/ar)]
	}

	return nil
}

// Values returns the values of the matrix in a "[][]float64" format.
func (m *Matrix) Values() [][]float64 {
	vs := make([][]float64, m.r)

	for rIndex := range vs {
		vs[rIndex] = m.vs[rIndex*m.c : rIndex*m.c+m.c]
	}

	return vs
}
