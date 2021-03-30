package matrix

import "errors"

// ErrZeroRow is returned by New when number of rows is equal to, or less than zero.
var ErrZeroRow = errors.New("matrix: number of rows must be greater than zero")

// ErrZeroCol is returned by New when number of columns equal to, or less than zero.
var ErrZeroCol = errors.New("matrix: number of rows must be greater than zero")

// ErrDataLength is returned by New when lenght of data is not equal to `r * c`
var ErrDataLength = errors.New("matrix: length of the data must be equal to `r * c`")

// ErrNilFunction is returned by Apply when `fn` is nil.
var ErrNilFunction = errors.New("matrix: function must not be nil")

// ErrRowOutOfBounds is returned by At when the supplied row is out of bounds.
var ErrRowOutOfBounds = errors.New("matrix: row out of bounds")

// ErrColOutOfBounds is returned by At when the supplied column is out of bounds.
var ErrColOutOfBounds = errors.New("matrix: column out of bounds")

// ErrNilMatrix is returned by any operation that is require a matrix as argument.
var ErrNilMatrix = errors.New("matrix: matrix must not be nil")

// ErrBadProductDimension is returned by MatrixProduct when the number of columns in `a` not equal to the number of rows in `b` matrix.
var ErrBadProductDimension = errors.New("matrix: the number of columns in `a` matrix must be equal to the number of rows in `b` matrix")

// ErrDifferentDimensions is returned by any operation that is require two matrix as argument, and the dimensions of the matrices not the same.
var ErrDifferentDimensions = errors.New("matrix: the dimensions of the matrices must be the same")
