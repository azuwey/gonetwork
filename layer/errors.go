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

// ErrBadWeightsDimension is returned by New when the dimension of weights does not match the provided shape.
var ErrBadWeightsDimension = errors.New("layer: the dimension of weights does not match the provided shape")

// ErrBadBiasesDimension is returned by New when the dimension of biases does not match the provided shape.
var ErrBadBiasesDimension = errors.New("layer: the dimension of biases does not match the provided shape")

// ErrNotExistActivationFn is returned by New when the provided activation function does not exists.
var ErrNotExistActivationFn = errors.New("layer: the provided activation function does not exists")

// ErrNilRand is returned by New when the r is nil.
var ErrNilRand = errors.New("layer: the rand cannot be nil")

// ErrNilInput is returned by Forwardprop when the input matrix is nil.
var ErrNilInput = errors.New("layer: the input matrix cannot be nil")

// ErrBadInputShape is returned by Forwardprop when the input matrix shape does not match the input shape
var ErrBadInputShape = errors.New("layer: the provided input matrix does not match the input shape")

// ErrNilTarget is returned by Backprop when the target matrix is nil.
var ErrNilTarget = errors.New("layer: the target matrix cannot be nil")

// ErrBadTargetShape is returned by Backprop when the target matrix shape does not match the output shape
var ErrBadTargetShape = errors.New("layer: the provided target matrix does not match the output shape")
