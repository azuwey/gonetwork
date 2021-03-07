package matrix

import "errors"

// Value is an alias for float64
type Value float64

// Matrix represents a mathematical matrix
type Matrix struct {
	values []Value
	rows int
	columns int
}

// New returns a new Matrix
func New(rows, cols int, elements ...Value) (*Matrix, error) {
	if rows <= 0 || cols <= 0 {
		return nil, errors.New("Rows and columns must be greater than zero")
	}

	if len(elements) != rows * cols {
		return nil, errors.New("Length of elements must be equal to rows times columns")
	}

	matrix := &Matrix{make([]Value, rows * cols), rows, cols}
	copy(matrix.values, elements)

	return matrix, nil
}

// Zeros returns a new Matrix with zero values
func Zeros(rows, cols int) (*Matrix, error) {
	return New(rows, cols, make([]Value, rows * cols)...)
}

// AddScalar calculate the sum of the scalar and each element in the matrix
func (m *Matrix) AddScalar(s Value) (*Matrix) {
	r, _ := New(m.rows, m.columns, m.values...)

	for i := range m.values {
		r.values[i] += s
	}

	return r
}

// AddElementWise calculate the sum of the two matrices, the dimensions of the matrices must be same
func (m *Matrix) AddElementWise(em *Matrix) (*Matrix, error) {
	if m.rows != em.rows || m.columns != em.columns || len(m.values) != len(em.values) {
		return nil, errors.New("The dimensions of the matrices must be same")
	}

	r, _ := New(m.rows, m.columns, m.values...)

	for i := range r.values {
		r.values[i] += em.values[i]
	}

	return r, nil
}

// MultiplyScalar calculate the product of the scalar and each element in the matrix
func (m *Matrix) MultiplyScalar(s Value) *Matrix {
	r, _ := New(m.rows, m.columns, m.values...)

	for i := range m.values {
		r.values[i] *= s
	}

	return r
}

// MultiplyElementWise calculate the product of the two matrices, the dimensions of the matrices must be same
func (m *Matrix) MultiplyElementWise(em *Matrix) (*Matrix, error) {
	if m.rows != em.rows || m.columns != em.columns || len(m.values) != len(em.values) {
		return nil, errors.New("The dimensions of the matrices must be same")
	}

	r, _ := New(m.rows, m.columns, m.values...)

	for i := range r.values {
		r.values[i] *= em.values[i]
	}

	return r, nil
}

// MultiplyMatrix calculate the product of the two matrices
func (m *Matrix) MultiplyMatrix(em *Matrix) (*Matrix, error) {
	if m.columns != em.rows {
		return nil, errors.New("The columns of the original matrix must be equal with the rows of the other matrix")
	}

	r, _ := Zeros(m.rows, em.columns)

	for rIndex := range r.values {
		for emIndex, value := range em.values {
			r.values[rIndex] += value * m.values[emIndex + (rIndex * m.columns)]
		}
	}

	return r, nil
}