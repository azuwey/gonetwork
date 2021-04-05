package layer

import (
	"math/rand"

	"github.com/azuwey/gonetwork/activationfn"
	"github.com/azuwey/gonetwork/common"
	"github.com/azuwey/gonetwork/matrix"
)

type ArtificialLayerDescriptor struct {
	LayerDescriptor
	ActivationFn string    `json:"activationFn"`
	Weights      []float64 `json:"weights"`
	Biases       []float64 `json:"biases"`
}

type artificialLayer struct {
	layer
	activationFn    *activationfn.ActivationFunction
	weights, biases *matrix.Matrix
}

// NewArtificialLayer creates a new artificial layer.
func NewArtificialLayer(d ArtificialLayerDescriptor, r *rand.Rand) (*artificialLayer, error) {
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

	if d.Weights != nil && len(d.Weights) != d.OutputShape.Rows*d.InputShape.Rows {
		return nil, ErrBadWeightsDimension
	}

	if d.Biases != nil && len(d.Biases) != d.OutputShape.Rows {
		return nil, ErrBadBiasesDimension
	}

	aFn, ok := activationfn.ActivationFunctions[d.ActivationFn]
	if !ok {
		return nil, ErrNotExistActivationFn
	}

	if r == nil {
		return nil, ErrNilRand
	}

	var wv []float64 = nil
	if d.Weights != nil {
		wv = d.Weights
	} else {
		wv = make([]float64, d.OutputShape.Rows*d.InputShape.Rows)
		for i := range wv {
			wv[i] = rand.Float64()*2 - 1
		}
	}

	w, _ := matrix.New(d.OutputShape.Rows, d.InputShape.Rows, wv)

	var bv []float64 = nil
	if d.Biases != nil {
		bv = d.Biases
	}

	b, _ := matrix.New(d.OutputShape.Rows, 1, bv)

	if d.UUID == "" {
		d.UUID = common.GenerateUUID(10, r)
	}
	layer := layer{d.UUID, d.InputShape, d.OutputShape, nil, nil, d.LearningRate, &matrix.Matrix{}, &matrix.Matrix{}}
	return &artificialLayer{layer, aFn, w, b}, nil
}

func (l *artificialLayer) Forwardprop(input *matrix.Matrix) ([]float64, error) {
	if input == nil {
		return nil, ErrNilInput
	}

	if input.Rows != l.InputShape.Rows || input.Columns != l.InputShape.Columns || len(input.Values) != l.InputShape.Rows*l.InputShape.Columns {
		return nil, ErrBadInputShape
	}

	l.deactivated, _ = matrix.Copy(input)
	l.deactivated.Product(l.weights, l.deactivated)
	l.deactivated.Add(l.biases, l.deactivated)

	l.activated, _ = matrix.Copy(l.deactivated)
	l.activated.Apply(l.activationFn.ActivationFn(l.activated), l.activated)

	if l.Next == nil {
		return l.activated.Values, nil
	} else {
		/* i, _ := matrix.Copy(l.activated)
		i.Transpose(i) */
		return l.Next.Forwardprop(l.activated)
	}
}

func (l *artificialLayer) Backprop(target *matrix.Matrix) error {
	if target == nil {
		return nil // TODO: error
	}

	/* if target.Rows != l.OutputShape.Rows || target.Columns != l.OutputShape.Columns || len(target.Values) != l.OutputShape.Rows*l.OutputShape.Columns {
		return nil // TODO: error
	} */

	e := &matrix.Matrix{}

	if l.Next == nil {
		e.Subtract(target, l.activated)
	} else {
		e.Transpose(l.weights)
		e.Product(e, target)
	}

	g := &matrix.Matrix{}
	if err := g.Apply(l.activationFn.DeactivationFn(l.deactivated), l.deactivated); err != nil {
		return err
	}
	if err := g.Multiply(e, g); err != nil {
		return err
	}

	d := &matrix.Matrix{}
	// Need previous layer value
	if err := d.Transpose(l.activated); err != nil {
		return err
	}
	if err := d.Product(g, d); err != nil {
		return err
	}
	if err := d.Scale(*l.learningRate, d); err != nil {
		return err
	}

	if err := l.weights.Add(l.weights, d); err != nil {
		return err
	}
	if err := l.biases.Add(l.biases, g); err != nil {
		return err
	}

	if l.Previous == nil {
		return nil
	} else {
		return l.Previous.Backprop(e)
	}
}
