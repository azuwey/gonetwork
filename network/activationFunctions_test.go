package network

import (
	"testing"

	"github.com/azuwey/gonetwork/matrix"
)

func TestActivationFunction_activate(t *testing.T) {
	testCases := []struct {
		name                                                         string
		activationFunction                                           *ActivationFunction
		inputs, exceptedActivatedOutputs, exceptedDeactivatedOutputs []float64
	}{
		{"LogisticSigmoid", LogisticSigmoid, []float64{0.5}, []float64{0.62245}, []float64{0.23500}},
		{"TanH", TanH, []float64{0.5}, []float64{0.46211}, []float64{0.78644}},
		{"ReLU", ReLU, []float64{0.5, -0.1}, []float64{0.5, 0}, []float64{1, 0}},
		{"LeakyReLU", LeakyReLU, []float64{0.5, -0.1}, []float64{0.5, -0.001}, []float64{1, 0.01}},
		{"Softmax", Softmax, []float64{0.1, 0.2}, []float64{0.475021, (1 - 0.475021)}, []float64{0.249376, 0.249376}},
		{"StableSoftmax", StableSoftmax, []float64{1000, 2000}, []float64{0, 1}, []float64{0, 0}},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			m, _ := matrix.New(len(tc.inputs), 1, tc.inputs)
			aFn := tc.activationFunction.aFn(m)
			dFn := tc.activationFunction.dFn(m)

			for idx, i := range tc.inputs {
				aOut := aFn(i, idx, 0, tc.inputs)
				if !isFloatInThreshold(aOut, tc.exceptedActivatedOutputs[idx], 0.00001) {
					t.Errorf("expected activated output is %f, but got %f", tc.exceptedActivatedOutputs[idx], aOut)
				}

				dOut := dFn(i, idx, 0, tc.inputs)
				if !isFloatInThreshold(dOut, tc.exceptedDeactivatedOutputs[idx], 0.00001) {
					t.Errorf("expected deactivated output is %f, but got %f", tc.exceptedDeactivatedOutputs[idx], dOut)
				}
			}
		})
	}
}
