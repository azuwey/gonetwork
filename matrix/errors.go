package matrix

import "errors"

// ErrZeroRow is returned by any operation when the supplied number of rows is equal to, or less than zero.
var ErrZeroRow = errors.New("number of rows must be greater than zero")

// ErrZeroColumns is returned by any operation when the supplied number of columns is equal to, or less than zero.
var ErrZeroColumns = errors.New("number of columns must be greater than zero")

// ErrNilValues is returned by any operation when the supplied values is nil.
var ErrNilValues = errors.New("number of columns must be greater than zero")

// ErrNotMatrix is returned by any operation when the supplied parameter is not a Matrix instance.
// var ErrNotMatrix = errors.New("the supplied parameter is not a Matrix instance")

// ErrDifferentRows is returned by any operation when the supplied matrix does not have the same number of rows as the number of rows in the receiver
var ErrDifferentRows = errors.New("number of rows in the supplied matrix must the same as the number of rows in the receiver")

// ErrDifferentColumns is returned by any operation when the supplied matrix does not have the same number of columns as the number of columns in the receiver
var ErrDifferentColumns = errors.New("number of columns in the supplied matrix must the same as the number of columns in the receiver")

// ErrDataLength is returned by New when the length of the supplied values is not equal to the number of rows times the number of columns
var ErrDataLength = errors.New("length of the values must be equal to the number of rows times the number of columns")
