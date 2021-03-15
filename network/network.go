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

// calculateLayerValues returns the calculated values of a "l" layer, with "w" weights, "b" biases.
// It also performs the normalizations of the values, but also returns the not normalized values.
// If any of the matrices nil it will return an error.
// The return tuple will be in this order (notNormalizedValues, normalizedValues, error).
func calculateLayerValues(l, w, b *matrix.Matrix, fn func(v float64, r, c int) float64) (*matrix.Matrix, *matrix.Matrix, error) {
	if l == nil {
		return nil, nil, ErrNilMatrix
	}

	if w == nil {
		return nil, nil, ErrNilMatrix
	}

	if b == nil {
		return nil, nil, ErrNilMatrix
	}

	v := &matrix.Matrix{}
	if err := v.MatrixProduct(w, l); !errors.Is(err, nil) {
		return nil, nil, err
	}

	if err := v.Add(b, v); !errors.Is(err, nil) {
		return nil, nil, err
	}

	nv, err := matrix.Copy(v)
	if !errors.Is(err, nil) {
		return nil, nil, err
	}

	if err := nv.Apply(fn, nv); !errors.Is(err, nil) {
		return nil, nil, err
	}

	return nv, v, nil
}

// Predict ...
func (n *Network) Predict(i *matrix.Matrix, fn func(v float64, r, c int) float64) (*matrix.Matrix, error) {
	if i == nil {
		return nil, ErrNilMatrix
	}

	if fn == nil {
		return nil, ErrNilFn
	}

	_, normalizedHiddenValues, err := calculateLayerValues(i, n.hWeights, n.hBias, fn)
	if !errors.Is(err, nil) {
		return nil, err
	}

	_, normalizedOutputValues, err := calculateLayerValues(normalizedHiddenValues, n.oWeights, n.oBias, fn)
	if !errors.Is(err, nil) {
		return nil, err
	}

	return normalizedOutputValues, nil
}

// Train ...
func (n *Network) Train(i *matrix.Matrix, t *matrix.Matrix, aFn matrix.ApplyFn, dFn matrix.ApplyFn) error {
	/* Error checks */
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

	hv, nhv, err := calculateLayerValues(i, n.hWeights, n.hBias, aFn)
	if !errors.Is(err, nil) {
		return err
	}

	ov, nov, err := calculateLayerValues(nhv, n.oWeights, n.oBias, aFn)
	if !errors.Is(err, nil) {
		return err
	}

	/* Calculate output errors */
	oe := &matrix.Matrix{}
	if err := oe.Subtract(t, nov); !errors.Is(err, nil) {
		return err
	}

	/* Calculate output gradient */
	og := &matrix.Matrix{}
	if err := og.Apply(dFn, ov); !errors.Is(err, nil) {
		return err
	}

	if err := og.Multiply(og, oe); !errors.Is(err, nil) {
		return err
	}

	if err := og.Scale(n.learningRate, og); !errors.Is(err, nil) {
		return err
	}

	/* To calculate the deltas between the output and the hidden nodes, we transpose the hidden nodes */
	ht := &matrix.Matrix{}
	if err := ht.Transpose(nhv); !errors.Is(err, nil) {
		return err
	}

	/* Calculate the deltas between the output and the hidden nodes */
	hd := &matrix.Matrix{}
	if err := hd.MatrixProduct(og, ht); !errors.Is(err, nil) {
		return err
	}

	/* Apply the calculated deltas to the output weights */
	if err := n.oWeights.Add(n.oWeights, hd); !errors.Is(err, nil) {
		return err
	}

	/* Apply the gradient to the output biases */
	if err := n.oBias.Add(n.oBias, og); !errors.Is(err, nil) {
		return err
	}

	/* Calculate hidden errors */
	he := &matrix.Matrix{}
	if err := he.Transpose(n.oWeights); !errors.Is(err, nil) {
		return err
	}

	if err := he.MatrixProduct(he, oe); !errors.Is(err, nil) {
		return err
	}

	/* Calculate hidden gradient */
	hg := &matrix.Matrix{}
	if err := hg.Apply(dFn, hv); !errors.Is(err, nil) {
		return err
	}

	if err := hg.Multiply(hg, he); !errors.Is(err, nil) {
		return err
	}

	if err := hg.Scale(n.learningRate, hg); !errors.Is(err, nil) {
		return err
	}

	/* To calculate the deltas between the hidden and the input nodes, we transpose the input nodes */
	it := &matrix.Matrix{}
	if err := it.Transpose(i); !errors.Is(err, nil) {
		return err
	}

	/* Calculate the deltas between the hidden and the input nodes */
	id := &matrix.Matrix{}
	if err := id.MatrixProduct(hg, it); !errors.Is(err, nil) {
		return err
	}

	/* Apply the calculated deltas to the hidden weights */
	if err := n.hWeights.Add(n.hWeights, id); !errors.Is(err, nil) {
		return err
	}

	/* Apply the gradient to the hidden biases */
	if err := n.hBias.Add(n.hBias, hg); !errors.Is(err, nil) {
		return err
	}

	return nil
}
