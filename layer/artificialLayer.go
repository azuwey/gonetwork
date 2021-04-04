package layer

import (
	"math/rand"

	"github.com/azuwey/gonetwork/activationfn"
	"github.com/azuwey/gonetwork/matrix"
)

type artificialLayer layer

// NewArtificialLayer creates a new artificial layer.
func NewArtificialLayer(d LayerDescriptor, r *rand.Rand) (*artificialLayer, error) {
	if d.OutputShape.Rows == 0 {
		return nil, ErrZeroRow
	}

	if d.OutputShape.Columns != 1 {
		return nil, ErrOutOfRangeColumn
	}

	if d.OutputShape.Depth != 1 {
		return nil, ErrOutOfRangeDepth
	}

	if d.InputShape.Rows == 0 {
		return nil, ErrZeroRow
	}

	if d.InputShape.Columns != 1 {
		return nil, ErrOutOfRangeColumn
	}

	if d.InputShape.Depth != 1 {
		return nil, ErrOutOfRangeDepth
	}

	if d.Weights != nil && len(d.Weights) > 0 && len(d.Weights) != 1 {
		return nil, nil // TODO: error
	}

	if d.Weights != nil && len(d.Weights) == 1 && len(d.Weights[0]) != d.OutputShape.Rows*d.InputShape.Rows {
		return nil, nil // TODO: error
	}

	if d.Biases != nil && len(d.Biases) > 0 && len(d.Biases) != 1 {
		return nil, nil // TODO: error
	}

	if d.Biases != nil && len(d.Biases) == 1 && len(d.Biases[0]) != d.OutputShape.Rows {
		return nil, nil // TODO: error
	}

	aFn, ok := activationfn.ActivationFunctions[d.ActivationFn]
	if !ok {
		return nil, nil // TODO: Error
	}

	if r == nil {
		return nil, nil // TODO: Error
	}

	var wv []float64 = nil
	if len(d.Weights) == 1 {
		wv = d.Weights[0]
	} else {
		wv = make([]float64, d.OutputShape.Rows*d.InputShape.Rows)
		for i := range wv {
			wv[i] = rand.Float64()*2 - 1
		}
	}

	w, _ := matrix.New(d.OutputShape.Rows, d.InputShape.Rows, wv)

	var bv []float64 = nil
	if len(d.Biases) == 1 {
		bv = d.Biases[0]
	}

	b, _ := matrix.New(d.OutputShape.Rows, 1, bv)
	return &artificialLayer{
		aFn, d.LearningRate, []*matrix.Matrix{w}, []*matrix.Matrix{b}, &matrix.Matrix{}, &matrix.Matrix{}, d.Previous, d.Next,
	}, nil
}

func (l *artificialLayer) Forwardprop(input *matrix.Matrix) error {
	if input == nil {
		return nil // TODO: error
	}

	l.unactivated, _ = matrix.Copy(input)
	l.unactivated.Product(l.weights[0], l.unactivated)
	l.unactivated.Add(l.biases[0], l.unactivated)

	l.activated, _ = matrix.Copy(l.unactivated)
	l.activated.Apply(l.activationFn.ActivationFn(l.activated), l.activated)

	return nil
}

func (l *artificialLayer) Backprop(target *matrix.Matrix) error {
	if target == nil {
		return nil // TODO: error
	}

	e := &matrix.Matrix{}

	if l.previous == nil {
		e.Subtract(target, l.activated)
	} else {
		e.Transpose(l.weights[0])
		e.Product(e, target)
	}

	g := &matrix.Matrix{}
	g.Apply(l.activationFn.DeactivationFn(l.unactivated), l.unactivated)
	g.Multiply(e, g)

	d := &matrix.Matrix{}
	d.Transpose(l.activated)
	d.Product(g, d)
	d.Scale(*l.learningRate, d) // TODO: shared learning rate

	if l.next == nil {
		return nil
	} else {
		return l.next.Backprop(e)
	}
}
