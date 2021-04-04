package layer

import (
	"math/rand"
	"testing"

	"github.com/azuwey/gonetwork/activationfn"
)

func TestNewArtificialLayer(t *testing.T) {
	// rnd := rand.New(rand.NewSource(0))
	testCases := []struct {
		name          string
		rand          *rand.Rand
		layer         LayerDescriptor
		expectedError error
	}{
		{"normal", rand.New(rand.NewSource(0)), LayerDescriptor{}, nil},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			l, err := NewArtificialLayer(tc.layer, tc.rand)
			if tc.expectedError != nil {
				if err != tc.expectedError {
					t.Errorf("expected error is %v, but got %v", tc.expectedError, err)
				}
			} else if err != nil {
				t.Errorf("expected error is %v, but got %v", nil, err)
			} else if l == nil {
				t.Error("layer should not be nil")
			} else {
				if l.activationFn != activationfn.ActivationFunctions[tc.layer.ActivationFn] {
					t.Errorf("the activation function should be %v, but got %v", *activationfn.ActivationFunctions[tc.layer.ActivationFn], *l.activationFn)
				}

				if l.learningRate != tc.layer.LearningRate {
					t.Errorf("expected learning rate is %f, but got %f", *tc.layer.LearningRate, *l.learningRate)
				}

				if len(l.weights) != 1 {
					t.Errorf("expected length of weights is %d, but got %d", 1, len(l.weights))
				}

				if len(l.weights[0].Values) != tc.layer.OutputShape.Rows*tc.layer.InputShape.Rows {
					t.Errorf("expected length of weights[0] is %d, but got %d", tc.layer.OutputShape.Rows*tc.layer.InputShape.Rows, len(l.weights[0].Values))
				}

				if len(l.biases) != 1 {
					t.Errorf("expected length of biases is %d, but got %d", 1, len(l.biases))
				}

				if len(l.biases[0].Values) != tc.layer.OutputShape.Rows {
					t.Errorf("expected length of biases[0] is %d, but got %d", tc.layer.OutputShape.Rows, len(l.biases[0].Values))
				}
			}
		})
	}
}
