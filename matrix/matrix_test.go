package matrix

import "testing"

func TestNew(t *testing.T) {
	rows := 2
	cols := 2
	values := []Value{0, 1, 2, 3}

	rMatrix, err := New(rows, cols, values...);

	if err != nil {
		t.Log("Err should be nil", err)
		t.Fail()
	}

	if (len(rMatrix.values) != rows * cols) {
		t.Logf("length of rMatrix.values should be %d but it's %d", rows * cols, len(rMatrix.values))
		t.Fail()
	}

	for i := range rMatrix.values {
		if (rMatrix.values[i] != values[i]) {
			t.Logf("value of rMatrix.values[%d] should be %f, but it's %f", i, values[i], rMatrix.values[i])
			t.Fail()
		}
	}
}

func TestNewBadRow(t *testing.T) {
	rows := 0
	cols := 2
	values := []Value{0}

	_, err := New(rows, cols, values...);

	if err == nil {
		t.Log("Err should not be nil")
		t.Fail()
	}
}

func TestNewBadColumn(t *testing.T) {
	rows := 1
	cols := 0
	values := []Value{0}

	_, err := New(rows, cols, values...);

	if err == nil {
		t.Log("Err should not be nil")
		t.Fail()
	}
}

func TestNewBadElementLength(t *testing.T) {
	rows := 2
	cols := 1
	values := []Value{0}

	_, err := New(rows, cols, values...);

	if err == nil {
		t.Log("Err should not be nil")
		t.Fail()
	}
}

func TestZeros(t *testing.T) {
	rows := 3
	cols := 5

	rMatrix, err := Zeros(rows, cols);

	if err != nil {
		t.Log("Err should be nil", err)
		t.Fail()
	}

	if (len(rMatrix.values) != rows * cols) {
		t.Logf("length of rMatrix.values should be %d but it's %d", rows * cols, len(rMatrix.values))
		t.Fail()
	}

	for i := range rMatrix.values {
		if (rMatrix.values[i] != 0) {
			t.Logf("value of rMatrix.values[%d] should be %d, but it's %f", i, 0, rMatrix.values[i])
			t.Fail()
		}
	}
}

func TestAddScalar(t *testing.T) {
	rows := 2
	cols := 2
	scalar := Value(5.0)
	values := []Value{0, 1, 2, 3}

	matrix, _ := New(rows, cols, values...)
	rMatrix := matrix.AddScalar(scalar);

	for i := range values {
		if (rMatrix.values[i] != values[i] + scalar) {
			t.Logf("value of rMatrix.values[%d] should be %f, but it's %f", i, values[i] + scalar, rMatrix.values[i])
			t.Fail()
		}
	}
}

func TestAddElementWise(t *testing.T) {
	rows := 2
	cols := 2
	values := []Value{0, 1, 2, 3}

	aMatrix, _ := New(rows, cols, values...)
	bMatrix, _ := New(rows, cols, values...)
	rMatrix, err := aMatrix.AddElementWise(bMatrix)

	if err != nil {
		t.Log("Err should be nil", err)
		t.Fail()
	}

	for i := range rMatrix.values {
		if (rMatrix.values[i] != aMatrix.values[i] + bMatrix.values[i]) {
			t.Logf("value of rMatrix.values[%d] should be %f, but it's %f", i, aMatrix.values[i] + bMatrix.values[i], rMatrix.values[i])
			t.Fail()
		}
	}
}

func TestAddElementWiseBadMatrix(t *testing.T) {
	aRows := 2
	aCols := 2
	bRows := 1
	bCols := 2
	aValues := []Value{0, 1, 2, 3}
	aBalues := []Value{0, 1}

	aMatrix, _ := New(aRows, aCols, aValues...)
	bMatrix, _ := New(bRows, bCols, aBalues...)
	_, err := aMatrix.AddElementWise(bMatrix)

	if err == nil {
		t.Log("Err should not be nil")
		t.Fail()
	}
}

func TestMultiplyScalar(t *testing.T) {
	rows := 2
	cols := 2
	scalar := Value(5.0)
	values := []Value{0, 1, 2, 3}

	matrix, _ := New(rows, cols, values...)
	rMatrix := matrix.MultiplyScalar(scalar);

	for i := range values {
		if (rMatrix.values[i] != values[i] * scalar) {
			t.Logf("value of rMatrix.values[%d] should be %f, but it's %f", i, values[i] * scalar, rMatrix.values[i])
			t.Fail()
		}
	}
}

func TestMultiplyElementWise(t *testing.T) {
	rows := 2
	cols := 2
	values := []Value{0, 1, 2, 3}

	aMatrix, _ := New(rows, cols, values...)
	bMatrix, _ := New(rows, cols, values...)
	rMatrix, err := bMatrix.MultiplyElementWise(aMatrix)

	if err != nil {
		t.Log("Err should be nil", err)
		t.Fail()
	}

	for i := range rMatrix.values {
		if (rMatrix.values[i] != aMatrix.values[i] * bMatrix.values[i]) {
			t.Logf("value of rMatrix.values[%d] should be %f, but it's %f", i, aMatrix.values[i] * bMatrix.values[i], rMatrix.values[i])
			t.Fail()
		}
	}
}

func TestMultiplyElementWiseBadMatrix(t *testing.T) {
	aRows := 2
	aCols := 2
	bRows := 1
	bCols := 2
	aValues := []Value{0, 1, 2, 3}
	aBalues := []Value{0, 1}

	aMatrix, _ := New(aRows, aCols, aValues...)
	bMatrix, _ := New(bRows, bCols, aBalues...)
	_, err := aMatrix.AddElementWise(bMatrix)

	if err == nil {
		t.Log("Err should not be nil")
		t.Fail()
	}
}

func TestMultiplyMatrix(t *testing.T) {
	aRows := 3
	aCols := 2
	bRows := 2
	bCols := 1
	aValues := []Value{0, 1, 2, 3, 4, 5}
	bValues := []Value{0, -1}
	rValues := []Value{-1, -3, -5}

	aMatrix, _ := New(aRows, aCols, aValues...)
	bMatrix, _ := New(bRows, bCols, bValues...)
	rMatrix, err := aMatrix.MultiplyMatrix(bMatrix)

	if err != nil {
		t.Log("Err should be nil", err)
		t.Fail()
	}

	if len(rMatrix.values) != len(rValues) {
		t.Logf("length of rMatrix.values should be %d but it's %d", len(rValues), len(rMatrix.values))
		t.Fail()
	}

	for i := range rMatrix.values {
		if (rMatrix.values[i] != rValues[i]) {
			t.Logf("value of rMatrix.values[%d] should be %f, but it's %f", i, rValues[i], rMatrix.values[i])
			t.Fail()
		}
	}
}

func TestMultiplyMatrixBadMatrix(t *testing.T) {
	aRows := 2
	aCols := 3
	bRows := 2
	bCols := 1
	aValues := []Value{0, 1, 2, 3, 4, 5}
	bValues := []Value{0, -1}

	aMatrix, _ := New(aRows, aCols, aValues...)
	bMatrix, _ := New(bRows, bCols, bValues...)
	_, err := aMatrix.MultiplyMatrix(bMatrix)

	if err == nil {
		t.Log("Err should not be nil")
		t.Fail()
	}
}
