package network

import (
	"math"

	"github.com/azuwey/gonetwork/matrix"
)

// ActivationFunction is an alias for the type of the activation functions
type ActivationFunction struct {
	aFn, dFn matrix.ApplyFn
}

// List of all the available activation functions
var (
	LogisticSigmoid *ActivationFunction = &ActivationFunction{
		aFn: func(v float64, _, _ int) float64 {
			return 1 / (1 + math.Exp(-v))
		},
		dFn: func(v float64, r, c int) float64 {
			return v * (1 - v)
		},
	}
)
