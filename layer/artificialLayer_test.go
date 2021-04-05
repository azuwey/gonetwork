package layer

import (
	"math"
	"math/rand"
	"testing"

	"github.com/azuwey/gonetwork/activationfn"
	"github.com/azuwey/gonetwork/matrix"
)

func TestNewArtificialLayer(t *testing.T) {
	learningRate := 0.01
	testCases := []struct {
		name          string
		rand          *rand.Rand
		layer         ArtificialLayerDescriptor
		expectedUUID  string
		expectedError error
	}{
		{"With UUID", rand.New(rand.NewSource(0)), ArtificialLayerDescriptor{
			LayerDescriptor{"mdN6RA0rI3", "", "", Shape{2, 1, 1}, Shape{4, 1, 1}, &learningRate}, "ReLU", nil, nil,
		}, "mdN6RA0rI3", nil},
		{"Without weights and biases", rand.New(rand.NewSource(0)), ArtificialLayerDescriptor{
			LayerDescriptor{"", "", "", Shape{2, 1, 1}, Shape{4, 1, 1}, &learningRate}, "ReLU", nil, nil,
		}, "mUNERA0rI3", nil},
		{"With weights, without biases", rand.New(rand.NewSource(0)), ArtificialLayerDescriptor{
			LayerDescriptor{"", "", "", Shape{2, 1, 1}, Shape{4, 1, 1}, &learningRate}, "ReLU", make([]float64, 2*4), nil,
		}, "mUNERA0rI3", nil},
		{"Without weights, with biases", rand.New(rand.NewSource(0)), ArtificialLayerDescriptor{
			LayerDescriptor{"", "", "", Shape{2, 1, 1}, Shape{4, 1, 1}, &learningRate}, "ReLU", nil, make([]float64, 4),
		}, "mUNERA0rI3", nil},
		{"With weights, with biases", rand.New(rand.NewSource(0)), ArtificialLayerDescriptor{
			LayerDescriptor{"", "", "", Shape{2, 1, 1}, Shape{4, 1, 1}, &learningRate}, "ReLU", make([]float64, 2*4), make([]float64, 4),
		}, "mUNERA0rI3", nil},
		{"ErrZeroRow output", rand.New(rand.NewSource(0)), ArtificialLayerDescriptor{
			LayerDescriptor{"", "", "", Shape{2, 1, 1}, Shape{0, 1, 1}, &learningRate}, "ReLU", nil, nil,
		}, "", ErrZeroRow},
		{"ErrOutOfRangeColumn output", rand.New(rand.NewSource(0)), ArtificialLayerDescriptor{
			LayerDescriptor{"", "", "", Shape{2, 1, 1}, Shape{4, 2, 1}, &learningRate}, "ReLU", nil, nil,
		}, "", ErrOutOfRangeColumn},
		{"ErrOutOfRangeDepth output", rand.New(rand.NewSource(0)), ArtificialLayerDescriptor{
			LayerDescriptor{"", "", "", Shape{2, 1, 1}, Shape{4, 1, 2}, &learningRate}, "ReLU", nil, nil,
		}, "", ErrOutOfRangeDepth},
		{"ErrZeroRow output", rand.New(rand.NewSource(0)), ArtificialLayerDescriptor{
			LayerDescriptor{"", "", "", Shape{0, 1, 1}, Shape{4, 1, 1}, &learningRate}, "ReLU", nil, nil,
		}, "", ErrZeroRow},
		{"ErrOutOfRangeColumn output", rand.New(rand.NewSource(0)), ArtificialLayerDescriptor{
			LayerDescriptor{"", "", "", Shape{2, 2, 1}, Shape{4, 1, 1}, &learningRate}, "ReLU", nil, nil,
		}, "", ErrOutOfRangeColumn},
		{"ErrOutOfRangeDepth output", rand.New(rand.NewSource(0)), ArtificialLayerDescriptor{
			LayerDescriptor{"", "", "", Shape{2, 1, 2}, Shape{4, 1, 1}, &learningRate}, "ReLU", nil, nil,
		}, "", ErrOutOfRangeDepth},
		{"ErrBadWeightsDimension", rand.New(rand.NewSource(0)), ArtificialLayerDescriptor{
			LayerDescriptor{"", "", "", Shape{2, 1, 1}, Shape{4, 1, 1}, &learningRate}, "ReLU", []float64{}, nil,
		}, "", ErrBadWeightsDimension},
		{"ErrBadBiasesDimension", rand.New(rand.NewSource(0)), ArtificialLayerDescriptor{
			LayerDescriptor{"", "", "", Shape{2, 1, 1}, Shape{4, 1, 1}, &learningRate}, "ReLU", nil, []float64{},
		}, "", ErrBadBiasesDimension},
		{"ErrNotExistActivationFn", rand.New(rand.NewSource(0)), ArtificialLayerDescriptor{
			LayerDescriptor{"", "", "", Shape{2, 1, 1}, Shape{4, 1, 1}, &learningRate}, "", nil, nil,
		}, "", ErrNotExistActivationFn},
		{"ErrNilRand", nil, ArtificialLayerDescriptor{
			LayerDescriptor{"", "", "", Shape{2, 1, 1}, Shape{4, 1, 1}, &learningRate}, "ReLU", nil, nil,
		}, "", ErrNilRand},
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

				if len(l.weights.Values) != tc.layer.OutputShape.Rows*tc.layer.InputShape.Rows {
					t.Errorf("expected length of weights[0] is %d, but got %d", tc.layer.OutputShape.Rows*tc.layer.InputShape.Rows, len(l.weights.Values))
				}

				if len(l.biases.Values) != tc.layer.OutputShape.Rows {
					t.Errorf("expected length of biases[0] is %d, but got %d", tc.layer.OutputShape.Rows, len(l.biases.Values))
				}

				if l.UUID != tc.expectedUUID {
					t.Errorf("expected UUID is %s, but got %s", tc.expectedUUID, l.UUID)
				}

				if l.Next != nil {
					t.Errorf("expected Next is %v, but got %v", nil, l.Next)
				}

				if l.Previous != nil {
					t.Errorf("expected Previous is %v, but got %v", nil, l.Next)
				}
			}
		})
	}
}

func TestForwardpropArtificialLayer(t *testing.T) {
	learningRate := 0.01
	testCases := []struct {
		name               string
		rand               *rand.Rand
		layers             []ArtificialLayerDescriptor
		input              *matrix.Matrix
		expectedPrediction []float64
		expectedError      error
	}{
		{"Single layer", rand.New(rand.NewSource(0)), []ArtificialLayerDescriptor{
			{LayerDescriptor{"mdN6RA0rI0", "", "", Shape{2, 1, 1}, Shape{4, 1, 1}, &learningRate}, "ReLU",
				[]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}, []float64{0.01, 0.02, 0.03, 0.04},
			}}, &matrix.Matrix{Values: []float64{0.5, 0.5}, Rows: 2, Columns: 1}, []float64{0.16, 0.37, 0.58, 0.79}, nil,
		},
		{"Dual layer", rand.New(rand.NewSource(0)), []ArtificialLayerDescriptor{
			{LayerDescriptor{"mdN6RA0rI0", "mdN6RA0rI1", "", Shape{2, 1, 1}, Shape{4, 1, 1}, &learningRate}, "ReLU",
				[]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}, []float64{0.01, 0.02, 0.03, 0.04},
			}, {LayerDescriptor{"mdN6RA0rI1", "", "mdN6RA0rI0", Shape{4, 1, 1}, Shape{1, 1, 1}, &learningRate}, "ReLU",
				[]float64{0.1, 0.2, 0.3, 0.4}, []float64{0.01},
			}}, &matrix.Matrix{Values: []float64{0.5, 0.5}, Rows: 2, Columns: 1}, []float64{0.59}, nil,
		},
		{"ErrNilInput", rand.New(rand.NewSource(0)), []ArtificialLayerDescriptor{
			{LayerDescriptor{"mdN6RA0rI0", "", "", Shape{2, 1, 1}, Shape{4, 1, 1}, &learningRate}, "ReLU",
				[]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}, []float64{0.01, 0.02, 0.03, 0.04},
			}}, nil, []float64{0.16, 0.37, 0.58, 0.79}, ErrNilInput,
		},
		{"ErrBadInputShape empty input values", rand.New(rand.NewSource(0)), []ArtificialLayerDescriptor{
			{LayerDescriptor{"mdN6RA0rI0", "", "", Shape{2, 1, 1}, Shape{4, 1, 1}, &learningRate}, "ReLU",
				[]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}, []float64{0.01, 0.02, 0.03, 0.04},
			}}, &matrix.Matrix{Values: []float64{}, Rows: 2, Columns: 1}, []float64{0.16, 0.37, 0.58, 0.79}, ErrBadInputShape,
		},
		{"ErrBadInputShape bad number of rows", rand.New(rand.NewSource(0)), []ArtificialLayerDescriptor{
			{LayerDescriptor{"mdN6RA0rI0", "", "", Shape{2, 1, 1}, Shape{4, 1, 1}, &learningRate}, "ReLU",
				[]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}, []float64{0.01, 0.02, 0.03, 0.04},
			}}, &matrix.Matrix{Values: []float64{0.5, 0.5}, Rows: 1, Columns: 1}, []float64{0.16, 0.37, 0.58, 0.79}, ErrBadInputShape,
		},
		{"ErrBadInputShape bad number of columns", rand.New(rand.NewSource(0)), []ArtificialLayerDescriptor{
			{LayerDescriptor{"mdN6RA0rI0", "", "", Shape{2, 1, 1}, Shape{4, 1, 1}, &learningRate}, "ReLU",
				[]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}, []float64{0.01, 0.02, 0.03, 0.04},
			}}, &matrix.Matrix{Values: []float64{0.5, 0.5}, Rows: 2, Columns: 2}, []float64{0.16, 0.37, 0.58, 0.79}, ErrBadInputShape,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			layers := make(map[string]*artificialLayer)

			for _, d := range tc.layers {
				layers[d.UUID], _ = NewArtificialLayer(d, tc.rand)
			}

			var fl *artificialLayer
			for _, d := range tc.layers {
				l := layers[d.UUID]
				if d.PreviousLayerUUID != "" {
					l.Previous = layers[d.PreviousLayerUUID]
				} else {
					fl = l
				}

				if d.NextLayerUUID != "" {
					l.Next = layers[d.NextLayerUUID]
				}

				layers[d.UUID] = l
			}

			prediction, err := fl.Forwardprop(tc.input)
			if tc.expectedError != nil {
				if err != tc.expectedError {
					t.Errorf("expected error is %v, but got %v", tc.expectedError, err)
				}
			} else if err != nil {
				t.Errorf("expected error is %v, but got %v", nil, err)
			} else {
				for idx, ep := range tc.expectedPrediction {
					if math.Abs(prediction[idx]-ep) > 0.00001 {
						t.Errorf("expected prediction[%d] is %f +-0.00001, but got %f", idx, ep, prediction[idx])
					}
				}
			}
		})
	}
}

func TestBackwardpropArtificialLayer(t *testing.T) {
	learningRate := 0.1
	testCases := []struct {
		name            string
		rand            *rand.Rand
		layers          []ArtificialLayerDescriptor
		input, target   *matrix.Matrix
		expectedWeights [][]float64
		expectedError   error
	}{
		/*{"Single layer already optimized", rand.New(rand.NewSource(0)), []ArtificialLayerDescriptor{
			{LayerDescriptor{"mdN6RA0rI0", "", "", Shape{2, 1, 1}, Shape{4, 1, 1}, &learningRate}, "ReLU",
				[]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}, []float64{0.01, 0.02, 0.03, 0.04},
			}},
			&matrix.Matrix{Values: []float64{0.5, 0.5}, Rows: 2, Columns: 1},
			&matrix.Matrix{Values: []float64{0.16, 0.37, 0.58, 0.79}, Rows: 4, Columns: 1}, [][]float64{
				{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8},
			}, nil,
		},*/
		{"Single layer not optimized", rand.New(rand.NewSource(0)), []ArtificialLayerDescriptor{
			{LayerDescriptor{"mdN6RA0rI0", "", "", Shape{2, 1, 1}, Shape{4, 1, 1}, &learningRate}, "ReLU",
				[]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}, []float64{0.01, 0.02, 0.03, 0.04},
			}},
			&matrix.Matrix{Values: []float64{0.5, 0.5}, Rows: 2, Columns: 1},
			&matrix.Matrix{Values: []float64{0.1, 0.7, 0.2, 0.9}, Rows: 4, Columns: 1}, [][]float64{
				{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8},
			}, nil,
		},
		/*{"Dual layer", rand.New(rand.NewSource(0)), []ArtificialLayerDescriptor{
			{LayerDescriptor{"mdN6RA0rI0", "mdN6RA0rI1", "", Shape{2, 1, 1}, Shape{4, 1, 1}, &learningRate}, "ReLU",
				[]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}, []float64{0.01, 0.02, 0.03, 0.04},
			}, {LayerDescriptor{"mdN6RA0rI1", "", "mdN6RA0rI0", Shape{4, 1, 1}, Shape{1, 1, 1}, &learningRate}, "ReLU",
				[]float64{0.1, 0.2, 0.3, 0.4}, []float64{0.01},
			}}, &matrix.Matrix{Values: []float64{0.5, 0.5}, Rows: 2, Columns: 1}, []float64{0.59}, nil,
		},
		{"ErrNilInput", rand.New(rand.NewSource(0)), []ArtificialLayerDescriptor{
			{LayerDescriptor{"mdN6RA0rI0", "", "", Shape{2, 1, 1}, Shape{4, 1, 1}, &learningRate}, "ReLU",
				[]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}, []float64{0.01, 0.02, 0.03, 0.04},
			}}, nil, []float64{0.16, 0.37, 0.58, 0.79}, ErrNilInput,
		},
		{"ErrBadInputShape empty input values", rand.New(rand.NewSource(0)), []ArtificialLayerDescriptor{
			{LayerDescriptor{"mdN6RA0rI0", "", "", Shape{2, 1, 1}, Shape{4, 1, 1}, &learningRate}, "ReLU",
				[]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}, []float64{0.01, 0.02, 0.03, 0.04},
			}}, &matrix.Matrix{Values: []float64{}, Rows: 2, Columns: 1}, []float64{0.16, 0.37, 0.58, 0.79}, ErrBadInputShape,
		},
		{"ErrBadInputShape bad number of rows", rand.New(rand.NewSource(0)), []ArtificialLayerDescriptor{
			{LayerDescriptor{"mdN6RA0rI0", "", "", Shape{2, 1, 1}, Shape{4, 1, 1}, &learningRate}, "ReLU",
				[]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}, []float64{0.01, 0.02, 0.03, 0.04},
			}}, &matrix.Matrix{Values: []float64{0.5, 0.5}, Rows: 1, Columns: 1}, []float64{0.16, 0.37, 0.58, 0.79}, ErrBadInputShape,
		},
		{"ErrBadInputShape bad number of columns", rand.New(rand.NewSource(0)), []ArtificialLayerDescriptor{
			{LayerDescriptor{"mdN6RA0rI0", "", "", Shape{2, 1, 1}, Shape{4, 1, 1}, &learningRate}, "ReLU",
				[]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}, []float64{0.01, 0.02, 0.03, 0.04},
			}}, &matrix.Matrix{Values: []float64{0.5, 0.5}, Rows: 2, Columns: 2}, []float64{0.16, 0.37, 0.58, 0.79}, ErrBadInputShape,
		},*/
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			layers := make(map[string]*artificialLayer)

			for _, d := range tc.layers {
				layers[d.UUID], _ = NewArtificialLayer(d, tc.rand)
			}

			var ll *artificialLayer
			for _, d := range tc.layers {
				l := layers[d.UUID]
				if d.PreviousLayerUUID != "" {
					l.Previous = layers[d.PreviousLayerUUID]
				}

				if d.NextLayerUUID != "" {
					l.Next = layers[d.NextLayerUUID]
				} else {
					ll = l
				}

				layers[d.UUID] = l
			}

			ll.Forwardprop(tc.input)
			err := ll.Backprop(tc.target)
			if tc.expectedError != nil {
				if err != tc.expectedError {
					t.Errorf("expected error is %v, but got %v", tc.expectedError, err)
				}
			} else if err != nil {
				t.Errorf("expected error is %v, but got %v", nil, err)
			} else {
				for idx, d := range tc.layers {
					wvs := layers[d.UUID].weights.Values
					for wIdx, wv := range wvs {
						if math.Abs(wv-tc.expectedWeights[idx][wIdx]) > 0.00001 {
							t.Errorf("expected weights[%d][%d] is %f +-0.00001, but got %f", idx, wIdx, tc.expectedWeights[idx][wIdx], wv)
						}
					}
				}
			}
		})
	}
}
