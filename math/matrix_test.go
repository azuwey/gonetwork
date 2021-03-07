package math

import "testing"

func TestInit(t *testing.T) {
	rows := 3
	cols := 5

	matrix := InitMatrix(rows, cols);

	if len(matrix.values) != rows {
		t.Logf("length of matrix.values should be %d", rows)
		t.Fail()
	}

	for rowIndex, row := range matrix.values {
		if len(row) != cols {
			t.Logf("length of matrix.values[%d] should be %d", rowIndex, cols)
			t.Fail()
		}

		for colIndex, col := range row {
			if col != 0 {
				t.Logf("value of matrix.values[%d][%d] should be %d", rowIndex, colIndex, 0)
				t.Fail()
			}
		}
	}
}

func TestAdd(t *testing.T) {
	rows := 2
	cols := 3
	summand := 5.0

	matrix := InitMatrix(rows, cols)
	matrix.Add(summand);

	for rowIndex, row := range matrix.values {
		for colIndex, col := range row {
			if col != summand {
				t.Logf("value of matrix.values[%d][%d] should be %f", rowIndex, colIndex, summand)
				t.Fail()
			}
		}
	}
}

func TestMultiply(t *testing.T) {
	rows := 2
	cols := 3
	newValue := 5.0
	multiplier := 3.0

	matrix := InitMatrix(rows, cols)

	for rowIndex, row := range matrix.values {
		for colIndex := range row {
			matrix.values[rowIndex][colIndex] = newValue
		}
	}

	matrix.Multiply(multiplier);

	for rowIndex, row := range matrix.values {
		for colIndex, col := range row {
			if col != newValue * multiplier {
				t.Logf("value of matrix.values[%d][%d] should be %f", rowIndex, colIndex, newValue * multiplier)
				t.Fail()
			}
		}
	}
}
