package network

import (
	"errors"
	"math/rand"

	"github.com/azuwey/gonetwork/matrix"
)

// Layer represents a layer in the network.
type Layer struct {
	weights *matrix.Matrix
	biases  *matrix.Matrix
}

// Network represents the structure of a neural network.
type Network struct {
	learningRate       float64
	layers             []*Layer
	activationFunction *ActivationFunction
	rand               *rand.Rand
}

// New creates a new neural network with "ls" layer structure,
// the first element in the "ls" is the number of input nodes,
// the last element in the "ls" is the number of output nodes.
// It will return an error if "ls == nil || len(ls) < 3", "lr <= 0 || lr > 1", "a == nil", "r == nil" .
func New(ls []int, lr float64, a *ActivationFunction, r *rand.Rand) (*Network, error) {
	if ls == nil || len(ls) < 3 {
		return nil, nil // TODO: Error
	}

	if lr <= 0 || lr > 1 {
		return nil, nil // TODO: Error
	}

	if a == nil {
		return nil, nil // TODO: Error
	}

	if r == nil {
		return nil, nil // TODO: Error
	}

	ly := make([]*Layer, len(ls)-1)
	rnd := func(v float64, _, _ int) float64 {
		return r.Float64()*2 - 1
	}

	for i, n := range ls[1:] {
		w, err := matrix.New(n, ls[i], make([]float64, n*ls[i]))
		if !errors.Is(err, nil) {
			return nil, err
		}

		b, err := matrix.New(n, 1, make([]float64, n))
		if !errors.Is(err, nil) {
			return nil, err
		}

		w.Apply(rnd, w)
		b.Apply(rnd, b)
		ly[i] = &Layer{w, b}
	}

	n := &Network{lr, ly, a, r}

	return n, nil
}

func (n *Network) calculateLayerValues(iArr []float64) ([]*matrix.Matrix, error) {
	if iArr == nil {
		return nil, nil // TODO: Error
	}

	iMat, err := matrix.New(len(iArr), 1, iArr)
	if !errors.Is(err, nil) {
		return nil, err
	}

	lVals := make([]*matrix.Matrix, len(n.layers)+1)
	lVals[0] = iMat

	for i := range lVals[1:] {
		v := &matrix.Matrix{}
		if err := v.MatrixProduct(n.layers[i].weights, lVals[i]); !errors.Is(err, nil) {
			return nil, err
		}

		if err := v.Add(n.layers[i].biases, v); !errors.Is(err, nil) {
			return nil, err
		}

		if err := v.Apply(n.activationFunction.aFn, v); !errors.Is(err, nil) {
			return nil, err
		}

		lVals[i+1] = v
	}

	return lVals, nil
}

// Predict ...
func (n *Network) Predict(iArr []float64) ([]float64, error) {
	if iArr == nil {
		return nil, nil // TODO: Error
	}

	lVals, err := n.calculateLayerValues(iArr)
	if !errors.Is(err, nil) {
		return nil, err
	}

	return lVals[len(lVals)-1].Raw(), nil
}

// Train ...
func (n *Network) Train(iArr, tArr []float64) error {
	if iArr == nil {
		return nil // TODO: Error
	}

	if tArr == nil {
		return nil // TODO: Error
	}

	tMat, err := matrix.New(len(tArr), 1, tArr)
	if !errors.Is(err, nil) {
		return err
	}

	lVals, err := n.calculateLayerValues(iArr)
	if !errors.Is(err, nil) {
		return err
	}

	lastErrVal := &matrix.Matrix{}
	if lastErrVal.Subtract(tMat, lVals[len(lVals)-1]); !errors.Is(err, nil) {
		return err
	}

	for i := len(n.layers) - 1; i >= 0; i-- {
		e := &matrix.Matrix{}
		if i == len(n.layers)-1 {
			if e.Subtract(tMat, lVals[i+1]); !errors.Is(err, nil) {
				return err
			}
		} else {
			if err := e.Transpose(n.layers[i+1].weights); !errors.Is(err, nil) {
				return err
			}

			if err := e.MatrixProduct(e, lastErrVal); !errors.Is(err, nil) {
				return err
			}
			lastErrVal = e
		}

		g := &matrix.Matrix{}
		if err := g.Apply(n.activationFunction.dFn, lVals[i+1]); !errors.Is(err, nil) {
			return err
		}

		if err := g.Multiply(e, g); !errors.Is(err, nil) {
			return err
		}

		d := &matrix.Matrix{}
		if err := d.Transpose(lVals[i]); !errors.Is(err, nil) {
			return err
		}

		if err := d.MatrixProduct(g, d); !errors.Is(err, nil) {
			return err
		}

		if err := d.Scale(n.learningRate, d); !errors.Is(err, nil) {
			return err
		}

		if err := n.layers[i].weights.Add(n.layers[i].weights, d); !errors.Is(err, nil) {
			return err
		}

		if err := n.layers[i].biases.Add(n.layers[i].biases, g); !errors.Is(err, nil) {
			return err
		}
	}

	return nil
}
