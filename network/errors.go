package network

import "errors"

// ErrNilActivationFn is returned by New when `activationFunction` is not exists in ActivationFunctions this is not validation on the input layer.
var ErrActivationFnNotExist = errors.New("network: activation function must be exist in ActivationFunctions map")

// ErrNilLayerStruct is returned by New when `ls` is nil.
var ErrNilLayerStructure = errors.New("network: layer struct must not be nil")

// ErrLayerStructureLength is returned by New when lenght of `ls` is less than three.
var ErrLayerStructureLength = errors.New("network: length of the layer structure must be equal to, or greater than three")

// ErrLearningRateRange is returned by New when `lr` is equal to, or less than zero, or greater than one.
var ErrLearningRateRange = errors.New("network: learning rate range must be greater than zero and less than one")

// ErrNilRand is returned by New when `r` is nil.
var ErrNilRand = errors.New("network: random source must not be nil")

// ErrNilMatrix is returned by Train when `t` is nil.
var ErrNilTargetSlice = errors.New("network: traget slice must not be nil")

// ErrNilMatrix is returned by Train when the length of `t` is not equal to the number of nodes in the output layer.
var ErrBadTargetSlice = errors.New("network: length of the target slice needs to be the same as the number of nodes in the output layer")

// ErrNilMatrix is returned by any operation that is require a input slice as argument.
var ErrNilInputSlice = errors.New("network: input slice must not be nil")
