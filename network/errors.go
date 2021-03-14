package network

import "errors"

// These errors can be returned from any function in the network package
var (
	ErrNilMatrix          = errors.New("Matrix cannot be nil")
	ErrNilFn              = errors.New("Activation function cannot be nil")
	ErrZeroNode           = errors.New("Node must be greater than zero")
	ErrDifferentDimension = errors.New("The dimensions of the matrices must be same")
)
