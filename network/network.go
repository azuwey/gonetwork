package network

import (
	"errors"
	"math/rand"

	"github.com/azuwey/gonetwork/matrix"
)

// Layer used to generate the layers in the network.
type Layer struct {
	Nodes              int
	ActivationFunction *ActivationFunction
}

type layer struct {
	weights            *matrix.Matrix
	biases             *matrix.Matrix
	activationFunction *ActivationFunction
}

// Network represents the structure of a neural network.
type Network struct {
	learningRate float64
	layers       []*layer
	rand         *rand.Rand
}

// New creates a new neural network with "ls" layer structure,
// the first element in the "ls" represents the input layer,
// the last element in the "ls" represents the output layer.
// It will return an error if "ls == nil || len(ls) < 3", "lr <= 0 || lr > 1", "r == nil".
// It will also return an error if any of the layers activationFunction is nill except for the input layer.
func New(ls []Layer, lr float64, r *rand.Rand) (*Network, error) {
	if ls == nil || len(ls) < 3 {
		return nil, ErrLayerStructureLength
	}

	if lr <= 0 || lr > 1 {
		return nil, ErrLearningRateRange
	}

	if r == nil {
		return nil, ErrNilRand
	}

	ly := make([]*layer, len(ls)-1)
	rnd := func(v float64, _, _ int, _ []float64) float64 {
		return r.Float64()*2 - 1
	}

	for idx, n := range ls[1:] {
		if n.ActivationFunction == nil {
			return nil, ErrNilActivationFn
		}

		w, err := matrix.New(n.Nodes, ls[idx].Nodes, make([]float64, n.Nodes*ls[idx].Nodes))
		if !errors.Is(err, nil) {
			return nil, err
		}

		b, _ := matrix.New(n.Nodes, 1, make([]float64, n.Nodes))

		w.Apply(rnd, w)
		b.Apply(rnd, b)
		ly[idx] = &layer{w, b, n.ActivationFunction}
	}

	n := &Network{lr, ly, r}

	return n, nil
}

func (n *Network) calculateLayerValues(i []float64) ([]*matrix.Matrix, error) {
	iMat, err := matrix.New(len(i), 1, i)
	if !errors.Is(err, nil) {
		return nil, err
	}

	lVals := make([]*matrix.Matrix, len(n.layers)+1)
	lVals[0] = iMat

	for idx := range lVals[1:] {
		v := &matrix.Matrix{}

		v.Product(n.layers[idx].weights, lVals[idx])
		v.Add(n.layers[idx].biases, v)
		v.Apply(n.layers[idx].activationFunction.aFn(v), v)

		lVals[idx+1] = v
	}

	return lVals, nil
}

// Predict ...
func (n *Network) Predict(i []float64) ([]float64, error) {
	if i == nil {
		return nil, ErrNilInputSlice
	}

	lVals, err := n.calculateLayerValues(i)
	if !errors.Is(err, nil) {
		return nil, err
	}

	return lVals[len(lVals)-1].Values, nil
}

// Train ...
func (n *Network) Train(i, t []float64) error {
	if i == nil {
		return ErrNilInputSlice
	}

	if t == nil {
		return ErrNilTargetSlice
	}

	tMat, err := matrix.New(len(t), 1, t)
	if !errors.Is(err, nil) {
		return err
	}

	lVals, err := n.calculateLayerValues(i)
	if !errors.Is(err, nil) {
		return err
	}

	lastErrVal := &matrix.Matrix{}
	for idx := len(n.layers) - 1; idx >= 0; idx-- {
		e := &matrix.Matrix{}
		if idx == len(n.layers)-1 {
			if e.Subtract(tMat, lVals[idx+1]); !errors.Is(err, nil) {
				return err
			}
		} else {
			if err := e.Transpose(n.layers[idx+1].weights); !errors.Is(err, nil) {
				return err
			}

			if err := e.Product(e, lastErrVal); !errors.Is(err, nil) {
				return err
			}
		}
		lastErrVal = e

		g := &matrix.Matrix{}
		if err := g.Apply(n.layers[idx].activationFunction.dFn(lVals[idx+1]), lVals[idx+1]); !errors.Is(err, nil) {
			return err
		}

		if err := g.Multiply(e, g); !errors.Is(err, nil) {
			return err
		}

		d := &matrix.Matrix{}
		if err := d.Transpose(lVals[idx]); !errors.Is(err, nil) {
			return err
		}

		if err := d.Product(g, d); !errors.Is(err, nil) {
			return err
		}

		if err := d.Scale(n.learningRate, d); !errors.Is(err, nil) {
			return err
		}

		if err := n.layers[idx].weights.Add(n.layers[idx].weights, d); !errors.Is(err, nil) {
			return err
		}

		if err := n.layers[idx].biases.Add(n.layers[idx].biases, g); !errors.Is(err, nil) {
			return err
		}
	}

	return nil
}
