package matrix

import "errors"

// ErrZeroRow is returned by New when number of rows is equal to, or lower than zero.
var ErrZeroRow = errors.New("Matrix: number of rows must be greater than zero")

// ErrZeroCol is returned by New when number of columns equal to, or lower than zero.
var ErrZeroCol = errors.New("Matrix: number of rows must be greater than zero")

// ErrDataLength is returned by New when lenght of data is not equal to `r * c`
var ErrDataLength = errors.New("Matrix: length of the data must be equal to `r * c`")

// ErrNilFunction is returned by Apply when `fn` is nil.
var ErrNilFunction = errors.New("Matrix: function must not be nil")

// ErrRowOutOfBounds is returned by At when the supplied row is out of bounds.
var ErrRowOutOfBounds = errors.New("Matrix: row out of bounds")

// ErrColOutOfBounds is returned by At when the supplied column is out of bounds.
var ErrColOutOfBounds = errors.New("Matrix: column out of bounds")

// ErrNilMatrix is returned by any operation that is require a matrix as argument.
var ErrNilMatrix = errors.New("Matrix: matrix must not be nil")

// ErrBadProductDimension is returned by MatrixProduct when the number of columns in `a` not equal to the number of rows in `b` matrix.
var ErrBadProductDimension = errors.New("Matrix: the number of columns in `a` matrix must be equal to the number of rows in `b` matrix")

// ErrDifferentDimensions is returned by any operation that is require two matrix as argument, and the dimensions of the matrices not the same.
var ErrDifferentDimensions = errors.New("Matrix: matrix must not be nil")
