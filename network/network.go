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
	learningRate                         float64
}

// New creates a new neural network with "i" input nodes, "h" hidden nodes and "o" output nodes.
// If any of theses nodes are smaller or equal than 0 it will return an error.
func New(i, h, o int, lr float64) (*Network, error) {
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

	n := &Network{i, h, o, hWeights, oWeights, hBias, oBias, lr}
	return n, nil
}

// FeedForward ...
func (n *Network) FeedForward(i *matrix.Matrix, fn func(v float64, r, c int) float64) (*matrix.Matrix, error) {
	if i == nil {
		return nil, ErrNilMatrix
	}

	if fn == nil {
		return nil, ErrNilFn
	}

	hiddenValues := &matrix.Matrix{}
	if err := hiddenValues.MatrixProduct(n.hWeights, i); !errors.Is(err, nil) {
		return nil, err
	}

	if err := hiddenValues.Add(n.hBias, hiddenValues); !errors.Is(err, nil) {
		return nil, err
	}

	if err := hiddenValues.Apply(fn, hiddenValues); !errors.Is(err, nil) {
		return nil, err
	}

	outputValues := &matrix.Matrix{}
	if err := outputValues.MatrixProduct(n.oWeights, hiddenValues); !errors.Is(err, nil) {
		return nil, err
	}

	if err := outputValues.Add(n.oBias, outputValues); !errors.Is(err, nil) {
		return nil, err
	}

	if err := outputValues.Apply(fn, outputValues); !errors.Is(err, nil) {
		return nil, err
	}

	return outputValues, nil
}

// Train ...
func (n *Network) Train(i *matrix.Matrix, t *matrix.Matrix, aFn matrix.ApplyFn, dFn matrix.ApplyFn) error {
	if i == nil {
		return ErrNilMatrix
	}

	if t == nil {
		return ErrNilMatrix
	}

	if aFn == nil {
		return ErrNilFn
	}

	if dFn == nil {
		return ErrNilFn
	}

	hiddenValues := &matrix.Matrix{}
	if err := hiddenValues.MatrixProduct(n.hWeights, i); !errors.Is(err, nil) {
		return err
	}

	if err := hiddenValues.Add(n.hBias, hiddenValues); !errors.Is(err, nil) {
		return err
	}

	hiddenRows, hiddenCols := hiddenValues.Dimensions()
	hiddenRawValues := make([]float64, hiddenRows*hiddenCols)
	copy(hiddenRawValues, hiddenValues.Raw())

	hiddenOriginal, err := matrix.New(hiddenRows, hiddenCols, hiddenRawValues)
	if !errors.Is(err, nil) {
		return err
	}

	if err := hiddenValues.Apply(aFn, hiddenValues); !errors.Is(err, nil) {
		return err
	}

	outputValues := &matrix.Matrix{}
	if err := outputValues.MatrixProduct(n.oWeights, hiddenValues); !errors.Is(err, nil) {
		return err
	}

	if err := outputValues.Add(n.oBias, outputValues); !errors.Is(err, nil) {
		return err
	}

	outputRows, outputCols := outputValues.Dimensions()
	outputRawValues := make([]float64, outputRows*outputCols)
	copy(outputRawValues, outputValues.Raw())

	outputOriginal, err := matrix.New(outputRows, outputCols, outputRawValues)
	if !errors.Is(err, nil) {
		return err
	}

	if err := outputValues.Apply(aFn, outputValues); !errors.Is(err, nil) {
		return err
	}

	outputErrors := &matrix.Matrix{}
	if err := outputErrors.Subtract(t, outputValues); !errors.Is(err, nil) {
		return err
	}

	outputGradient := &matrix.Matrix{}
	if err := outputGradient.Apply(dFn, outputOriginal); !errors.Is(err, nil) {
		return err
	}

	if err := outputGradient.Multiply(outputGradient, outputErrors); !errors.Is(err, nil) {
		return err
	}

	if err := outputGradient.Scale(n.learningRate, outputGradient); !errors.Is(err, nil) {
		return err
	}

	hiddenTransposed := &matrix.Matrix{}
	if err := hiddenTransposed.Transpose(hiddenValues); !errors.Is(err, nil) {
		return err
	}

	hiddenDeltas := &matrix.Matrix{}
	if err := hiddenDeltas.MatrixProduct(outputGradient, hiddenTransposed); !errors.Is(err, nil) {
		return err
	}

	if err := n.oWeights.Add(hiddenDeltas, n.oWeights); !errors.Is(err, nil) {
		return err
	}

	if err := n.oBias.Add(n.oBias, outputGradient); !errors.Is(err, nil) {
		return err
	}

	hiddenErrors := &matrix.Matrix{}
	if err := hiddenErrors.Transpose(n.oWeights); !errors.Is(err, nil) {
		return err
	}

	if err := hiddenErrors.MatrixProduct(hiddenErrors, outputErrors); !errors.Is(err, nil) {
		return err
	}

	hiddenGradient := &matrix.Matrix{}
	if err := hiddenGradient.Apply(dFn, hiddenOriginal); !errors.Is(err, nil) {
		return err
	}

	if err := hiddenGradient.Multiply(hiddenGradient, hiddenErrors); !errors.Is(err, nil) {
		return err
	}

	if err := hiddenGradient.Scale(n.learningRate, hiddenGradient); !errors.Is(err, nil) {
		return err
	}

	inputTransposed := &matrix.Matrix{}
	if err := inputTransposed.Transpose(i); !errors.Is(err, nil) {
		return err
	}

	inputDeltas := &matrix.Matrix{}
	if err := inputDeltas.MatrixProduct(hiddenGradient, inputTransposed); !errors.Is(err, nil) {
		return err
	}

	if err := n.hWeights.Add(inputDeltas, n.hWeights); !errors.Is(err, nil) {
		return err
	}

	if err := n.hBias.Add(n.hBias, hiddenGradient); !errors.Is(err, nil) {
		return err
	}

	return nil
}
