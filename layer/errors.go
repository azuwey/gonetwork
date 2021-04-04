package layer

import "errors"

// ErrZeroRow is returned by New when the number of rows is zero.
var ErrZeroRow = errors.New("layer: the number of rows cannot be zero")

// ErrZeroColumn is returned by New when the number of columns is zero.
var ErrZeroColumn = errors.New("layer: the number of columns cannot be zero")

// ErrZeroDepth is returned by New when the depth is zero.
var ErrZeroDepth = errors.New("layer: the depth be zero")

// ErrOutOfRangeRow is returned by New when the number of rows is out of range.
var ErrOutOfRangeRow = errors.New("layer: the number of rows is out of range")

// ErrOutOfRangeColumn is returned by New when the number of columns is out of range.
var ErrOutOfRangeColumn = errors.New("layer: the number of columns is out of range")

// ErrOutOfRangeDepth is returned by New when the number of columns is out of range.
var ErrOutOfRangeDepth = errors.New("layer: the number of columns is out of range")
