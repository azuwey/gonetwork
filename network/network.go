package network

import (
	"errors"
	"math/rand"

	"github.com/azuwey/gonetwork/matrix"
)

// Network represents the structure of a neural network
type Network struct {
	inputNodes, hiddenNodes, outputNodes int
	hWeights, oWeights, hBias, oBias     *matrix.Matrix
}

// New creates a new neural network with "i" input nodes, "h" hidden nodes and "o" output nodes.
// If any of theses nodes are smaller or equal than 0 it will return an error.
func New(i, h, o int) (*Network, error) {
	if i <= 0 || h <= 0 || o <= 0 {
		return nil, ErrZeroNode
	}

	randomize := func(v float64, r, c int) float64 {
		return rand.Float64()*2 - 1
	}

	hWeights, err := matrix.New(h, i, make([]float64, h*i))
	if !errors.Is(err, nil) {
		return nil, err
	}

	oWeights, err := matrix.New(o, h, make([]float64, o*h))
	if !errors.Is(err, nil) {
		return nil, err
	}

	hBias, err := matrix.New(h, 1, make([]float64, h))
	if !errors.Is(err, nil) {
		return nil, err
	}

	oBias, err := matrix.New(o, 1, make([]float64, o))
	if !errors.Is(err, nil) {
		return nil, err
	}

	hWeights.Apply(randomize, hWeights)
	oWeights.Apply(randomize, oWeights)
	hBias.Apply(randomize, hBias)
	oBias.Apply(randomize, oBias)

	n := &Network{i, h, o, hWeights, oWeights, hBias, oBias}
	return n, nil
}

// FeedForward ...
func (n *Network) FeedForward(i *matrix.Matrix, fn func(v float64, r, c int) float64) (*matrix.Matrix, error) {
	hiddenValues := &matrix.Matrix{}
	err := hiddenValues.MatrixProduct(n.hWeights, i)
	if !errors.Is(err, nil) {
		return nil, err
	}

	err = hiddenValues.Add(n.hBias, hiddenValues)
	if !errors.Is(err, nil) {
		return nil, err
	}

	hiddenValues.Apply(fn, hiddenValues)

	outputValues := &matrix.Matrix{}
	err = outputValues.MatrixProduct(n.oWeights, hiddenValues)
	if !errors.Is(err, nil) {
		return nil, err
	}

	err = outputValues.Add(n.oBias, outputValues)
	if !errors.Is(err, nil) {
		return nil, err
	}

	outputValues.Apply(fn, outputValues)

	return outputValues, nil
}
