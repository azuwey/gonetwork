package network

import "errors"

// These errors can be returned from any function in the network package
var (
	ErrZeroNode = errors.New("Node must be greater than zero")
)
