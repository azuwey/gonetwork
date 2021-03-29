package matrix

import (
	"sync"
)

type Matrix interface {
	// Dimensions returns the dimensions of the matrix
	Dimensions() (rows int, columns int)

	// Copy returns a new instance of the matrix
	Copy() Matrix

	// Sum performs element-wise addition, placing the result in the receiver
	Sum(b Matrix) error

	// Raw returns the underlaying slice of the matrix
	Raw() []float64
}

type matrix struct {
	*sync.Mutex
	Rows    int
	Columns int
	Values  []float64
}

func New(rows, columns int, values []float64) (Matrix, error) {
	if rows <= 0 {
		return nil, ErrZeroRow
	}

	if columns <= 0 {
		return nil, ErrZeroColumns
	}

	if values == nil {
		return nil, ErrNilValues
	}

	if len(values) != rows*columns {
		return nil, ErrDataLength
	}

	return &matrix{&sync.Mutex{}, rows, columns, values}, nil
}

func Zero(rows, columns int) (Matrix, error) {
	if rows <= 0 {
		return nil, ErrZeroRow
	}

	if columns <= 0 {
		return nil, ErrZeroColumns
	}

	return &matrix{&sync.Mutex{}, rows, columns, make([]float64, rows*columns)}, nil
}

func Empty() Matrix {
	return &matrix{&sync.Mutex{}, 0, 0, make([]float64, 0)}
}

func (m *matrix) Copy() Matrix {
	m.Lock()
	defer m.Unlock()

	values := make([]float64, m.Rows*m.Columns)
	copy(values, m.Values)
	return &matrix{&sync.Mutex{}, m.Rows, m.Columns, values}
}

func (m *matrix) Sum(matrix Matrix) error {
	mCopy := matrix.Copy()

	m.Lock()
	defer m.Unlock()
	mRows, mColumns := mCopy.Dimensions()

	if m.Rows != mRows {
		return ErrDifferentRows
	}

	if m.Columns != mColumns {
		return ErrDifferentColumns
	}

	values := mCopy.Raw()

	for idx := range m.Values {
		m.Values[idx] += values[idx]
	}

	return nil
}

func (m *matrix) Dimensions() (int, int) {
	return m.Rows, m.Columns
}

func (m *matrix) Raw() []float64 {
	m.Lock()
	defer m.Unlock()

	values := make([]float64, m.Rows*m.Columns)
	copy(values, m.Values)

	return values
}
