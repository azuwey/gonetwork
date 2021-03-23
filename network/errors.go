package network

import "errors"

// ErrNilActivationFn is returned by New when `activationFunction` is nil in any layer except the input layer.
var ErrNilActivationFn = errors.New("Network: activation function must not be nil")

// ErrNilLayerStruct is returned by New when `ls` is nil.
var ErrNilLayerStructure = errors.New("Network: layer struct must not be nil")

// ErrLayerStructureLength is returned by New when lenght of `ls` is less than three.
var ErrLayerStructureLength = errors.New("Network: length of the layer structure must be equal to, or greater than three")

// ErrLearningRateRange is returned by New when `lr` is equal to, or less than zero, or greater than one.
var ErrLearningRateRange = errors.New("Network: learning rate range must be greater than zero and less than one")

// ErrNilRand is returned by New when `r` is nil.
var ErrNilRand = errors.New("Network: random source must not be nil")

// ErrNilMatrix is returned by Train when `t` is nil.
var ErrNilTargetSlice = errors.New("Network: traget slice must not be nil")

// ErrNilMatrix is returned by any operation that is require a input slice as argument.
var ErrNilInputSlice = errors.New("Network: input slice must not be nil")
