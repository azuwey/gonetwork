package matrix

import "errors"

// Errors these errors can be returned from any function in the matrix package
var (
	ErrZeroRowOrCol       = errors.New("Rows and columns must be greater than zero")
	ErrDataLength         = errors.New("Length of the data must be equal to `r * c`")
	ErrNilMatrix          = errors.New("Matrix cannot be nil")
	ErrDifferentDimesion  = errors.New("The dimensions of the matrices must be same")
	ErrBadProductDimesion = errors.New("The number of columns in `a` matrix must be equal to the number of rows in `b` matrix")
)
