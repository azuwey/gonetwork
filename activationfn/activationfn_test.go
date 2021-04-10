package activationfn

import (
	"math"
	"testing"

	"github.com/azuwey/gonetwork/matrix"
)

func isFloatInThreshold(v float64, t float64, th float64) bool {
	if math.Abs(v-t) <= th {
		return true
	} else {
		return false
	}
}

func TestActivationFunction_activate(t *testing.T) {
	testCases := []struct {
		name, activationFunctionName                                 string
		inputs, exceptedActivatedOutputs, exceptedDeactivatedOutputs []float64
	}{
		{"LogisticSigmoid", "LogisticSigmoid", []float64{0.5}, []float64{0.62245}, []float64{0.23500}},
		{"TanH", "TanH", []float64{0.5}, []float64{0.46211}, []float64{0.78644}},
		{"ReLU", "ReLU", []float64{0.5, -0.1}, []float64{0.5, 0}, []float64{1, 0}},
		{"LeakyReLU", "LeakyReLU", []float64{0.5, -0.1}, []float64{0.5, -0.001}, []float64{1, 0.01}},
		{"Softmax", "Softmax", []float64{1.43, -0.4, 0.23}, []float64{0.684178, 0.109751, 0.206070}, []float64{0.216078, -0.075090, -0.140989}},
		{"StableSoftmax", "StableSoftmax", []float64{1000, 2000, 3000}, []float64{0, 0, 1}, []float64{0, 0, 0}},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			m, _ := matrix.New(len(tc.inputs), 1, tc.inputs)
			aFn := ActivationFunctions[tc.activationFunctionName].ActivationFn(m)

			m.Apply(aFn, m)
			for idx, out := range tc.exceptedActivatedOutputs {
				if !isFloatInThreshold(m.Values[idx], out, 0.00001) {
					t.Errorf("expected activated output is %f, but got %f", out, m.Values[idx])
				}
			}

			m, _ = matrix.New(len(tc.inputs), 1, tc.inputs)
			dFn := ActivationFunctions[tc.activationFunctionName].DeactivationFn(m)
			m.Apply(dFn, m)
			for idx, out := range tc.exceptedDeactivatedOutputs {
				if !isFloatInThreshold(m.Values[idx], out, 0.00001) {
					t.Errorf("expected deactivated output is %f, but got %f", out, m.Values[idx])
				}
			}
		})
	}
}
