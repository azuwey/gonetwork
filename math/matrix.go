package math

// Matrix represents a mathematical matrix
type Matrix struct {
	values [][]float64
}

// InitMatrix is initialize a new Matrix
func InitMatrix(rows, cols int) *Matrix {
	matrix := new(Matrix)
	matrix.values = make([][]float64, rows)

	for rowIndex := range matrix.values {
		matrix.values[rowIndex] = make([]float64, cols)
	}

	return matrix
}

// Add is add the summand value to all the values in the matrix
func (m *Matrix) Add(summand float64) {
	for rowIndex, row := range m.values {
		for colIndex := range row {
			m.values[rowIndex][colIndex] += summand;
		}
	}
}

// Multiply is multiply all the values in the matrix by the given multiplier
func (m *Matrix) Multiply(multiplier float64) {
	for rowIndex, row := range m.values {
		for colIndex := range row {
			m.values[rowIndex][colIndex] *= multiplier;
		}
	}
}
