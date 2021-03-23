package network

import (
	"math"

	"github.com/azuwey/gonetwork/matrix"
)

// ActivationFunction is an alias for the type of the activation functions
type ActivationFunction struct {
	aFn, dFn func(*matrix.Matrix) matrix.ApplyFn
}

// LogisticSigmoid ...
var LogisticSigmoid *ActivationFunction = &ActivationFunction{
	aFn: func(_ *matrix.Matrix) matrix.ApplyFn {
		return func(v float64, _, _ int, _ []float64) float64 {
			return 1 / (1 + math.Exp(-v))
		}
	},
	dFn: func(_ *matrix.Matrix) matrix.ApplyFn {
		return func(v float64, _, _ int, _ []float64) float64 {
			return (1 - v) * v
		}
	},
}

// TanH ...
var TanH *ActivationFunction = &ActivationFunction{
	aFn: func(_ *matrix.Matrix) matrix.ApplyFn {
		return func(v float64, _, _ int, _ []float64) float64 {
			return math.Tanh(v)
		}
	},
	dFn: func(_ *matrix.Matrix) matrix.ApplyFn {
		return func(v float64, _, _ int, _ []float64) float64 {
			return 1 - math.Pow(v, 2)
		}
	},
}

// ReLU ...
var ReLU *ActivationFunction = &ActivationFunction{
	aFn: func(_ *matrix.Matrix) matrix.ApplyFn {
		return func(v float64, _, _ int, _ []float64) float64 {
			return math.Max(0, v)
		}
	},
	dFn: func(_ *matrix.Matrix) matrix.ApplyFn {
		return func(v float64, _, _ int, _ []float64) float64 {
			if v >= 0 {
				return 1
			} else {
				return 0
			}
		}
	},
}

// LeakyReLU ...
var LeakyReLU *ActivationFunction = &ActivationFunction{
	aFn: func(_ *matrix.Matrix) matrix.ApplyFn {
		return func(v float64, _, _ int, _ []float64) float64 {
			if v >= 0 {
				return v
			} else {
				return 0.01 * v
			}
		}
	},
	dFn: func(_ *matrix.Matrix) matrix.ApplyFn {
		return func(v float64, _, _ int, _ []float64) float64 {
			if v >= 0 {
				return 1
			} else {
				return 0.01
			}
		}
	},
}

// Softmax ...
var Softmax *ActivationFunction = &ActivationFunction{
	aFn: func(m *matrix.Matrix) matrix.ApplyFn {
		sum := 0.0
		for _, v := range m.Values {
			sum += math.Exp(v)
		}

		return func(v float64, _, _ int, _ []float64) float64 {
			return math.Exp(v) / sum
		}
	},
	dFn: func(_ *matrix.Matrix) matrix.ApplyFn {
		return func(v float64, _, _ int, _ []float64) float64 {
			return v * (1 - v)
		}
	},
}
