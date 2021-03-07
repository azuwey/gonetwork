package math

import "errors"

// Matrix represents a mathematical matrix
type Matrix struct {
	values [][]float64
}

// InitMatrix is initialize a new Matrix with 0 values
func InitMatrix(rows, cols int) *Matrix {
	matrix := &Matrix{}
	matrix.values = make([][]float64, rows)

	for rowIndex := range matrix.values {
		matrix.values[rowIndex] = make([]float64, cols)
	}

	return matrix
}

// Add is sum the values of the original matrix and the summand.
// The summand can be a *Matrix with the same dimensions as the original matrix,
// or it can be a float64 number
func (m *Matrix) Add(summand interface{}) error {
	if summandMatrix, isMatrix := summand.(*Matrix); isMatrix {
		for rowIndex, row := range m.values {
			for colIndex := range row {
				m.values[rowIndex][colIndex] += summandMatrix.values[rowIndex][colIndex];
			}
		}
		return nil;
	}

	if summandFloat, isFloat := summand.(float64); isFloat {
		for rowIndex, row := range m.values {
			for colIndex := range row {
				m.values[rowIndex][colIndex] += summandFloat;
			}
		}
		return nil;
	}

	return errors.New("Not supported summand type, should be float64 or *Matrix")
}

// Multiply is multiply the values of the original matrix and the multiplier
// the multiplier can be a *Matrix with the same dimensions as the original matrix,
// or it can be a float64 number
func (m *Matrix) Multiply(multiplier interface{}) error {
	if multiplierMatrix, isMatrix := multiplier.(*Matrix); isMatrix {
		for rowIndex, row := range m.values {
			for colIndex := range row {
				m.values[rowIndex][colIndex] *= multiplierMatrix.values[rowIndex][colIndex];
			}
		}
		return nil;
	}

	if multiplierFloat, isFloat := multiplier.(float64); isFloat {
		for rowIndex, row := range m.values {
			for colIndex := range row {
				m.values[rowIndex][colIndex] *= multiplierFloat;
			}
		}
		return nil;
	}

	return errors.New("Not supported summand type, should be float64 or *Matrix")
}
