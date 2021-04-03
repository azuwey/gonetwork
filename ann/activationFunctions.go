package ann

import (
	"math"

	"github.com/azuwey/gonetwork/matrix"
)

// ActivationFunction is an alias for the type of the activation functions
type ActivationFunction struct {
	aFn, dFn func(*matrix.Matrix) matrix.ApplyFn
}

func calculateApplySum(s []float64, aFn func(float64) float64) float64 {
	sum := 0.0
	for _, v := range s {
		sum += aFn(v)
	}

	return sum
}

func calculateMax(s []float64) float64 {
	max := math.Inf(-1)
	for _, v := range s {
		max = math.Max(max, v)
	}

	return max
}

// LogisticSigmoid ...
var LogisticSigmoid *ActivationFunction = &ActivationFunction{
	aFn: func(_ *matrix.Matrix) matrix.ApplyFn {
		return func(v float64, _ int, _ []float64) float64 {
			return 1 / (1 + math.Exp(-v))
		}
	},
	dFn: func(_ *matrix.Matrix) matrix.ApplyFn {
		return func(v float64, _ int, _ []float64) float64 {
			v = (1 / (1 + math.Exp(-v)))
			return v * (1 - v)
		}
	},
}

// TanH ...
var TanH *ActivationFunction = &ActivationFunction{
	aFn: func(_ *matrix.Matrix) matrix.ApplyFn {
		return func(v float64, _ int, _ []float64) float64 {
			return math.Tanh(v)
		}
	},
	dFn: func(_ *matrix.Matrix) matrix.ApplyFn {
		return func(v float64, _ int, _ []float64) float64 {
			return 1 - math.Pow(math.Tanh(v), 2)
		}
	},
}

// ReLU ...
var ReLU *ActivationFunction = &ActivationFunction{
	aFn: func(_ *matrix.Matrix) matrix.ApplyFn {
		return func(v float64, _ int, _ []float64) float64 {
			return math.Max(0, v)
		}
	},
	dFn: func(_ *matrix.Matrix) matrix.ApplyFn {
		return func(v float64, _ int, _ []float64) float64 {
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
		return func(v float64, _ int, _ []float64) float64 {
			if v >= 0 {
				return v
			} else {
				return 0.01 * v
			}
		}
	},
	dFn: func(_ *matrix.Matrix) matrix.ApplyFn {
		return func(v float64, _ int, _ []float64) float64 {
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
		sum := calculateApplySum(m.Values, func(v float64) float64 {
			return math.Exp(v)
		})
		return func(v float64, _ int, _ []float64) float64 {
			return math.Exp(v) / sum
		}
	},
	dFn: func(m *matrix.Matrix) matrix.ApplyFn {
		sum := calculateApplySum(m.Values, func(v float64) float64 {
			return math.Exp(v)
		})
		var vF float64
		return func(v float64, idx int, _ []float64) float64 {
			v = math.Exp(v) / sum
			if idx == 0 {
				vF = v
				p := v * (1 - v)
				return p
			} else {
				return -vF * v
			}
		}
	},
}

// StableSoftmax ...
var StableSoftmax *ActivationFunction = &ActivationFunction{
	aFn: func(m *matrix.Matrix) matrix.ApplyFn {
		max := calculateMax(m.Values)
		sum := calculateApplySum(m.Values, func(v float64) float64 {
			return math.Exp(v - max)
		})
		return func(v float64, _ int, _ []float64) float64 {
			return math.Exp(v-max) / sum
		}
	},
	dFn: func(m *matrix.Matrix) matrix.ApplyFn {
		max := -calculateMax(m.Values)
		sum := calculateApplySum(m.Values, func(v float64) float64 {
			return math.Exp(v + max)
		})
		var vF float64
		return func(v float64, idx int, _ []float64) float64 {
			v = math.Exp(v+max) / sum
			v = v / sum
			if idx == 0 {
				vF = v
				p := v * (1 - v)
				return p
			} else {
				return -vF * v
			}
		}
	},
}

// ActivationFunctions ...
var ActivationFunctions = map[string]*ActivationFunction{
	"LogisticSigmoid": LogisticSigmoid,
	"TanH":            TanH,
	"ReLU":            ReLU,
	"LeakyReLU":       LeakyReLU,
	"Softmax":         Softmax,
	"StableSoftmax":   StableSoftmax,
}
