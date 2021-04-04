package cnn

import (
	"math/rand"

	"github.com/azuwey/gonetwork/activationfn"
	"github.com/azuwey/gonetwork/matrix"
)

// Layer represents a layer in the fully connected neural network.
type Layer struct {
	weights            *matrix.Matrix
	biases             *matrix.Matrix
	activationFunction *activationfn.ActivationFunction
}

type CNN struct {
	learningRate float64
	filters      []*matrix.Matrix
	rand         *rand.Rand
}

// New creates a new convolution neural network
func New() (*CNN, error) {
	return nil, nil
}
