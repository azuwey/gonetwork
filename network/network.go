package network

import (
	"errors"
	"math/rand"

	"github.com/azuwey/gonetwork/matrix"
)

// LayerDescriptor used to generate the layers in the network.
type LayerDescriptor struct {
	Nodes              int       `json:"nodes"`
	ActivationFunction string    `json:"activationFunction"`
	Weights            []float64 `json:"weights"`
	Biases             []float64 `json:"biases"`
}

type Model struct {
	LearningRate float64           `json:"learningRate"`
	Layers       []LayerDescriptor `json:"layers"`
}

// Layer represents a layer in the network
type Layer struct {
	weights            *matrix.Matrix
	biases             *matrix.Matrix
	activationFunction *ActivationFunction
}

// Network represents the structure of a neural network.
type Network struct {
	learningRate float64
	layers       []*Layer
	rand         *rand.Rand
}

// layerValues is used by calculateLayerValues to return both activated and unactivated values
type layerValues struct {
	activated   *matrix.Matrix
	unactivated *matrix.Matrix
}

// New creates a new neural network with "ls" layer structure,
// the first element in the "ls" represents the input layer,
// the last element in the "ls" represents the output layer.
// It will return an error if "ls == nil || len(ls) < 3", "lr <= 0 || lr > 1", "r == nil".
// It will also return an error if any of the layers activationFunction is nill except for the input layer.
func New(model *Model, r *rand.Rand) (*Network, error) {
	if model.Layers == nil || len(model.Layers) < 3 {
		return nil, ErrLayerStructureLength
	}

	if model.LearningRate <= 0 || model.LearningRate > 1 {
		return nil, ErrLearningRateRange
	}

	if r == nil {
		return nil, ErrNilRand
	}

	lyrs := make([]*Layer, len(model.Layers)-1)
	rnd := func(v float64, _, _ int, _ []float64) float64 {
		return r.Float64()*2 - 1
	}

	for idx, lyr := range model.Layers[1:] {
		w, err := matrix.New(lyr.Nodes, model.Layers[idx].Nodes, lyr.Weights)
		if !errors.Is(err, nil) {
			return nil, err
		}

		b, _ := matrix.New(lyr.Nodes, 1, lyr.Weights)

		w.Apply(rnd, w)
		b.Apply(rnd, b)

		aFn, ok := ActivationFunctions[lyr.ActivationFunction]
		if !ok {
			return nil, ErrActivationFnNotExist
		}

		lyrs[idx] = &Layer{w, b, aFn}
	}

	n := &Network{model.LearningRate, lyrs, r}

	return n, nil
}

func (n *Network) calculateLayerValues(i []float64) ([]*layerValues, error) {
	iMat, err := matrix.New(len(i), 1, i)
	if !errors.Is(err, nil) {
		return nil, err
	}

	vals := make([]*layerValues, len(n.layers)+1)
	vals[0] = &layerValues{iMat, nil}

	for idx := range vals[1:] {
		uV := &matrix.Matrix{}

		uV.Product(n.layers[idx].weights, vals[idx].activated)
		uV.Add(n.layers[idx].biases, uV)

		aV, _ := matrix.Copy(uV)
		aV.Apply(n.layers[idx].activationFunction.aFn(aV), aV)

		vals[idx+1] = &layerValues{aV, uV}
	}

	return vals, nil
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

	return lVals[len(lVals)-1].activated.Values, nil
}

// Train ...
func (n *Network) Train(i, t []float64) error {
	if i == nil {
		return ErrNilInputSlice
	}

	if t == nil {
		return ErrNilTargetSlice
	}

	lVals, err := n.calculateLayerValues(i)
	if !errors.Is(err, nil) {
		return err
	}

	tMat, err := matrix.New(len(t), 1, t)
	if !errors.Is(err, nil) {
		return err
	}

	if lVals[len(lVals)-1].activated.Rows != tMat.Rows {
		return ErrBadTargetSlice
	}

	lastErrVal := &matrix.Matrix{}
	for idx := len(n.layers) - 1; idx >= 0; idx-- {
		e := &matrix.Matrix{}
		if idx == len(n.layers)-1 {
			e.Subtract(tMat, lVals[idx+1].activated)
		} else {
			e.Transpose(n.layers[idx+1].weights)
			e.Product(e, lastErrVal)
		}
		lastErrVal = e

		g := &matrix.Matrix{}
		g.Apply(n.layers[idx].activationFunction.dFn(lVals[idx+1].unactivated), lVals[idx+1].unactivated)
		g.Multiply(e, g)

		d := &matrix.Matrix{}
		d.Transpose(lVals[idx].activated)
		d.Product(g, d)
		d.Scale(n.learningRate, d)

		n.layers[idx].weights.Add(n.layers[idx].weights, d)
		n.layers[idx].biases.Add(n.layers[idx].biases, g)
	}

	return nil
}
