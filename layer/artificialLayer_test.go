package layer

import (
	"math"
	"math/rand"
	"testing"

	"github.com/azuwey/gonetwork/activationfn"
	"github.com/azuwey/gonetwork/matrix"
)

func TestNew_artificialLayer(t *testing.T) {
	learningRate := 0.01
	testCases := []struct {
		name             string
		rand             *rand.Rand
		layerDescription ArtificialLayerDescriptor
		expectedUUID     string
		expectedError    error
	}{
		{"With UUID", rand.New(rand.NewSource(0)), ArtificialLayerDescriptor{
			LayerDescriptor{"ARTIFICIAL_mdN6RA0rI3", "", Shape{2, 1, 1}, Shape{4, 1, 1}, &learningRate}, "ReLU", nil, nil,
		}, "ARTIFICIAL_mdN6RA0rI3", nil},
		{"Without weights and biases", rand.New(rand.NewSource(0)), ArtificialLayerDescriptor{
			LayerDescriptor{"", "", Shape{2, 1, 1}, Shape{4, 1, 1}, &learningRate}, "ReLU", nil, nil,
		}, "ARTIFICIAL_mUNERA0rI3", nil},
		{"With weights, without biases", rand.New(rand.NewSource(0)), ArtificialLayerDescriptor{
			LayerDescriptor{"", "", Shape{2, 1, 1}, Shape{4, 1, 1}, &learningRate}, "ReLU", make([]float64, 2*4), nil,
		}, "ARTIFICIAL_mUNERA0rI3", nil},
		{"Without weights, with biases", rand.New(rand.NewSource(0)), ArtificialLayerDescriptor{
			LayerDescriptor{"", "", Shape{2, 1, 1}, Shape{4, 1, 1}, &learningRate}, "ReLU", nil, make([]float64, 4),
		}, "ARTIFICIAL_mUNERA0rI3", nil},
		{"With weights, with biases", rand.New(rand.NewSource(0)), ArtificialLayerDescriptor{
			LayerDescriptor{"", "", Shape{2, 1, 1}, Shape{4, 1, 1}, &learningRate}, "ReLU", make([]float64, 2*4), make([]float64, 4),
		}, "ARTIFICIAL_mUNERA0rI3", nil},
		{"ErrZeroRow output", rand.New(rand.NewSource(0)), ArtificialLayerDescriptor{
			LayerDescriptor{"", "", Shape{2, 1, 1}, Shape{0, 1, 1}, &learningRate}, "ReLU", nil, nil,
		}, "", ErrZeroRow},
		{"ErrOutOfRangeColumn output", rand.New(rand.NewSource(0)), ArtificialLayerDescriptor{
			LayerDescriptor{"", "", Shape{2, 1, 1}, Shape{4, 2, 1}, &learningRate}, "ReLU", nil, nil,
		}, "", ErrOutOfRangeColumn},
		{"ErrOutOfRangeDepth output", rand.New(rand.NewSource(0)), ArtificialLayerDescriptor{
			LayerDescriptor{"", "", Shape{2, 1, 1}, Shape{4, 1, 2}, &learningRate}, "ReLU", nil, nil,
		}, "", ErrOutOfRangeDepth},
		{"ErrZeroRow output", rand.New(rand.NewSource(0)), ArtificialLayerDescriptor{
			LayerDescriptor{"", "", Shape{0, 1, 1}, Shape{4, 1, 1}, &learningRate}, "ReLU", nil, nil,
		}, "", ErrZeroRow},
		{"ErrOutOfRangeColumn output", rand.New(rand.NewSource(0)), ArtificialLayerDescriptor{
			LayerDescriptor{"", "", Shape{2, 2, 1}, Shape{4, 1, 1}, &learningRate}, "ReLU", nil, nil,
		}, "", ErrOutOfRangeColumn},
		{"ErrOutOfRangeDepth output", rand.New(rand.NewSource(0)), ArtificialLayerDescriptor{
			LayerDescriptor{"", "", Shape{2, 1, 2}, Shape{4, 1, 1}, &learningRate}, "ReLU", nil, nil,
		}, "", ErrOutOfRangeDepth},
		{"ErrBadWeightsDimension", rand.New(rand.NewSource(0)), ArtificialLayerDescriptor{
			LayerDescriptor{"", "", Shape{2, 1, 1}, Shape{4, 1, 1}, &learningRate}, "ReLU", []float64{}, nil,
		}, "", ErrBadWeightsDimension},
		{"ErrBadBiasesDimension", rand.New(rand.NewSource(0)), ArtificialLayerDescriptor{
			LayerDescriptor{"", "", Shape{2, 1, 1}, Shape{4, 1, 1}, &learningRate}, "ReLU", nil, []float64{},
		}, "", ErrBadBiasesDimension},
		{"ErrNotExistActivationFn", rand.New(rand.NewSource(0)), ArtificialLayerDescriptor{
			LayerDescriptor{"", "", Shape{2, 1, 1}, Shape{4, 1, 1}, &learningRate}, "", nil, nil,
		}, "", ErrNotExistActivationFn},
		{"ErrNilRand", nil, ArtificialLayerDescriptor{
			LayerDescriptor{"", "", Shape{2, 1, 1}, Shape{4, 1, 1}, &learningRate}, "ReLU", nil, nil,
		}, "", ErrNilRand},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			l, err := NewArtificialLayer(tc.layerDescription, tc.rand)
			if tc.expectedError != nil {
				if err != tc.expectedError {
					t.Errorf("expected error is %v, but got %v", tc.expectedError, err)
				}
			} else if err != nil {
				t.Errorf("expected error is %v, but got %v", nil, err)
			} else if l == nil {
				t.Error("layer should not be nil")
			} else {
				if l.activationFn != activationfn.ActivationFunctions[tc.layerDescription.ActivationFn] {
					t.Errorf("the activation function should be %v, but got %v", *activationfn.ActivationFunctions[tc.layerDescription.ActivationFn], *l.activationFn)
				}

				if l.learningRate != tc.layerDescription.LearningRate {
					t.Errorf("expected learning rate is %f, but got %f", *tc.layerDescription.LearningRate, *l.learningRate)
				}

				if len(l.weights.Values) != tc.layerDescription.OutputShape.Rows*tc.layerDescription.InputShape.Rows {
					t.Errorf("expected length of weights[0] is %d, but got %d", tc.layerDescription.OutputShape.Rows*tc.layerDescription.InputShape.Rows, len(l.weights.Values))
				}

				if len(l.biases.Values) != tc.layerDescription.OutputShape.Rows {
					t.Errorf("expected length of biases[0] is %d, but got %d", tc.layerDescription.OutputShape.Rows, len(l.biases.Values))
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

func TestForwardprop_artificialLayer(t *testing.T) {
	learningRate := 0.01
	testCases := []struct {
		name               string
		rand               *rand.Rand
		layerDescriptions  []ArtificialLayerDescriptor
		input              *matrix.Matrix
		expectedPrediction []float64
		expectedError      error
	}{
		{"Single layer", rand.New(rand.NewSource(0)), []ArtificialLayerDescriptor{
			{LayerDescriptor{"ARTIFICIAL_mdN6RA0rI0", "", Shape{2, 1, 1}, Shape{4, 1, 1}, &learningRate}, "ReLU",
				[]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}, []float64{0.01, 0.02, 0.03, 0.04},
			}}, &matrix.Matrix{Values: []float64{0.5, 0.5}, Rows: 2, Columns: 1}, []float64{0.16, 0.37, 0.58, 0.79}, nil,
		},
		{"Dual layer", rand.New(rand.NewSource(0)), []ArtificialLayerDescriptor{
			{LayerDescriptor{"ARTIFICIAL_mdN6RA0rI0", "ARTIFICIAL_mdN6RA0rI1", Shape{2, 1, 1}, Shape{4, 1, 1}, &learningRate}, "ReLU",
				[]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}, []float64{0.01, 0.02, 0.03, 0.04},
			}, {LayerDescriptor{"ARTIFICIAL_mdN6RA0rI1", "", Shape{4, 1, 1}, Shape{1, 1, 1}, &learningRate}, "ReLU",
				[]float64{0.1, 0.2, 0.3, 0.4}, []float64{0.01},
			}}, &matrix.Matrix{Values: []float64{0.5, 0.5}, Rows: 2, Columns: 1}, []float64{0.59}, nil,
		},
		{"ErrNilInput", rand.New(rand.NewSource(0)), []ArtificialLayerDescriptor{
			{LayerDescriptor{"ARTIFICIAL_mdN6RA0rI0", "", Shape{2, 1, 1}, Shape{4, 1, 1}, &learningRate}, "ReLU",
				[]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}, []float64{0.01, 0.02, 0.03, 0.04},
			}}, nil, []float64{0.16, 0.37, 0.58, 0.79}, ErrNilInput,
		},
		{"ErrBadInputShape empty input values", rand.New(rand.NewSource(0)), []ArtificialLayerDescriptor{
			{LayerDescriptor{"ARTIFICIAL_mdN6RA0rI0", "", Shape{2, 1, 1}, Shape{4, 1, 1}, &learningRate}, "ReLU",
				[]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}, []float64{0.01, 0.02, 0.03, 0.04},
			}}, &matrix.Matrix{Values: []float64{}, Rows: 2, Columns: 1}, []float64{0.16, 0.37, 0.58, 0.79}, ErrBadInputShape,
		},
		{"ErrBadInputShape bad number of rows", rand.New(rand.NewSource(0)), []ArtificialLayerDescriptor{
			{LayerDescriptor{"ARTIFICIAL_mdN6RA0rI0", "", Shape{2, 1, 1}, Shape{4, 1, 1}, &learningRate}, "ReLU",
				[]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}, []float64{0.01, 0.02, 0.03, 0.04},
			}}, &matrix.Matrix{Values: []float64{0.5, 0.5}, Rows: 1, Columns: 1}, []float64{0.16, 0.37, 0.58, 0.79}, ErrBadInputShape,
		},
		{"ErrBadInputShape bad number of columns", rand.New(rand.NewSource(0)), []ArtificialLayerDescriptor{
			{LayerDescriptor{"ARTIFICIAL_mdN6RA0rI0", "", Shape{2, 1, 1}, Shape{4, 1, 1}, &learningRate}, "ReLU",
				[]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}, []float64{0.01, 0.02, 0.03, 0.04},
			}}, &matrix.Matrix{Values: []float64{0.5, 0.5}, Rows: 2, Columns: 2}, []float64{0.16, 0.37, 0.58, 0.79}, ErrBadInputShape,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			layers := make(map[string]*artificialLayer)

			for _, d := range tc.layerDescriptions {
				layers[d.UUID], _ = NewArtificialLayer(d, tc.rand)
			}

			for _, d := range tc.layerDescriptions {
				l := layers[d.UUID]
				if d.NextLayerUUID != "" {
					l.Next = layers[d.NextLayerUUID]
					layers[d.NextLayerUUID].Previous = l
				}

				layers[d.UUID] = l
			}

			var s *artificialLayer
			for _, d := range tc.layerDescriptions {
				l := layers[d.UUID]
				if l.Previous == nil {
					s = l
				}
			}

			prediction, err := s.Forwardprop(tc.input)
			if tc.expectedError != nil {
				if err != tc.expectedError {
					t.Errorf("expected error is %v, but got %v", tc.expectedError, err)
				}
			} else if err != nil {
				t.Errorf("expected error is %v, but got %v", nil, err)
			} else {
				for idx, ep := range tc.expectedPrediction {
					if math.Abs(prediction[idx]-ep) > 0.00001 {
						t.Errorf("expected prediction[%d] is %f+-0.00001, but got %f", idx, ep, prediction[idx])
					}
				}
			}
		})
	}
}

func TestBackwardprop_artificialLayer(t *testing.T) {
	learningRate := 0.1
	testCases := []struct {
		name              string
		rand              *rand.Rand
		layerDescriptions []ArtificialLayerDescriptor
		input, target     *matrix.Matrix
		expectedWeights   [][]float64
		expectedBiases    [][]float64
		expectedError     error
	}{
		{"Single layer already optimized", rand.New(rand.NewSource(0)), []ArtificialLayerDescriptor{
			{LayerDescriptor{"ARTIFICIAL_mdN6RA0rI0", "", Shape{2, 1, 1}, Shape{4, 1, 1}, &learningRate}, "ReLU",
				[]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}, []float64{0.01, 0.02, 0.03, 0.04},
			}},
			&matrix.Matrix{Values: []float64{0.5, 0.5}, Rows: 2, Columns: 1},
			&matrix.Matrix{Values: []float64{0.16, 0.37, 0.58, 0.79}, Rows: 4, Columns: 1}, [][]float64{
				{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8},
			}, [][]float64{
				{0.01, 0.02, 0.03, 0.04},
			}, nil,
		},
		{"Single layer not optimized", rand.New(rand.NewSource(0)), []ArtificialLayerDescriptor{
			{LayerDescriptor{"ARTIFICIAL_mdN6RA0rI0", "", Shape{2, 1, 1}, Shape{4, 1, 1}, &learningRate}, "ReLU",
				[]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}, []float64{0.01, 0.02, 0.03, 0.04},
			}},
			&matrix.Matrix{Values: []float64{0.5, 0.5}, Rows: 2, Columns: 1},
			&matrix.Matrix{Values: []float64{0.1, 0.7, 0.2, 0.9}, Rows: 4, Columns: 1}, [][]float64{
				{0.097, 0.197, 0.3165, 0.4165, 0.481, 0.581, 0.7055, 0.8055},
			}, [][]float64{
				{-0.05, 0.35, -0.35, 0.15},
			}, nil,
		},
		{"Dual layer not optimized", rand.New(rand.NewSource(0)), []ArtificialLayerDescriptor{
			{LayerDescriptor{"ARTIFICIAL_mdN6RA0rI0", "ARTIFICIAL_mdN6RA0rI1", Shape{2, 1, 1}, Shape{4, 1, 1}, &learningRate}, "ReLU",
				[]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}, []float64{0.01, 0.02, 0.03, 0.04},
			}, {LayerDescriptor{"ARTIFICIAL_mdN6RA0rI1", "", Shape{4, 1, 1}, Shape{4, 1, 1}, &learningRate}, "ReLU",
				[]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}, []float64{0.01, 0.02, 0.03, 0.04},
			}},
			&matrix.Matrix{Values: []float64{0.5, 0.5}, Rows: 2, Columns: 1},
			&matrix.Matrix{Values: []float64{0.1, 0.7, 0.2, 0.9}, Rows: 4, Columns: 1}, [][]float64{
				{0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85},
				{0.09216, 0.18187, 0.27158, 0.36129, 0.48944, 0.57558, 0.66172, 0.74786, 0.09344, 0.18483, 0.27622, 0.36761, 0.49232, 0.58224, 0.67216, 0.76208},
			}, [][]float64{
				{1.01, 1.02, 1.03, 1.04},
				{-0.48, -0.64, -0.38, -0.44},
			}, nil,
		},
		{"ErrNilTarget", rand.New(rand.NewSource(0)), []ArtificialLayerDescriptor{
			{LayerDescriptor{"ARTIFICIAL_mdN6RA0rI0", "", Shape{2, 1, 1}, Shape{4, 1, 1}, &learningRate}, "ReLU",
				[]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}, []float64{0.01, 0.02, 0.03, 0.04},
			}},
			&matrix.Matrix{Values: []float64{0.5, 0.5}, Rows: 2, Columns: 1}, nil, [][]float64{
				{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8},
			}, nil, ErrNilTarget,
		},
		{"ErrBadTargetShape", rand.New(rand.NewSource(0)), []ArtificialLayerDescriptor{
			{LayerDescriptor{"ARTIFICIAL_mdN6RA0rI0", "", Shape{2, 1, 1}, Shape{4, 1, 1}, &learningRate}, "ReLU",
				[]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}, []float64{0.01, 0.02, 0.03, 0.04},
			}},
			&matrix.Matrix{Values: []float64{0.5, 0.5}, Rows: 2, Columns: 1},
			&matrix.Matrix{Values: []float64{0.16, 0.37, 0.58}, Rows: 3, Columns: 1}, [][]float64{
				{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8},
			}, nil, ErrBadTargetShape,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			layers := make(map[string]*artificialLayer)

			for _, d := range tc.layerDescriptions {
				layers[d.UUID], _ = NewArtificialLayer(d, tc.rand)
			}

			var e *artificialLayer
			for _, d := range tc.layerDescriptions {
				l := layers[d.UUID]

				if d.NextLayerUUID != "" {
					l.Next = layers[d.NextLayerUUID]
					layers[d.NextLayerUUID].Previous = l
				} else {
					e = l
				}

				layers[d.UUID] = l
			}

			var s *artificialLayer
			for _, d := range tc.layerDescriptions {
				l := layers[d.UUID]
				if l.Previous == nil {
					s = l
				}
			}

			s.Forwardprop(tc.input)
			err := e.Backprop(tc.target)
			if tc.expectedError != nil {
				if err != tc.expectedError {
					t.Errorf("expected error is %v, but got %v", tc.expectedError, err)
				}
			} else if err != nil {
				t.Errorf("expected error is %v, but got %v", nil, err)
			} else {
				for idx, d := range tc.layerDescriptions {
					wvs := layers[d.UUID].weights.Values
					for wIdx, wv := range wvs {
						if math.Abs(wv-tc.expectedWeights[idx][wIdx]) > 0.00001 {
							t.Errorf("expected weights[%d][%d] is %f+-0.00001, but got %f", idx, wIdx, tc.expectedWeights[idx][wIdx], wv)
						}
					}

					bvs := layers[d.UUID].biases.Values
					for bIdx, bv := range bvs {
						if math.Abs(bv-tc.expectedBiases[idx][bIdx]) > 0.00001 {
							t.Errorf("expected biases[%d][%d] is %f+-0.00001, but got %f", idx, bIdx, tc.expectedBiases[idx][bIdx], bv)
						}
					}
				}
			}
		})
	}
}

func TestGetLayerDescription_artificialLayer(t *testing.T) {
	learningRate := 0.1
	testCases := []struct {
		name              string
		rand              *rand.Rand
		layerDescriptions []ArtificialLayerDescriptor
	}{
		{"Normal", rand.New(rand.NewSource(0)), []ArtificialLayerDescriptor{
			{LayerDescriptor{"ARTIFICIAL_mdN6RA0rI0", "ARTIFICIAL_mdN6RA0rI1", Shape{2, 1, 1}, Shape{4, 1, 1}, &learningRate}, "ReLU",
				[]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}, []float64{0.01, 0.02, 0.03, 0.04},
			}, {LayerDescriptor{"ARTIFICIAL_mdN6RA0rI1", "", Shape{4, 1, 1}, Shape{4, 1, 1}, &learningRate}, "ReLU",
				[]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}, []float64{0.01, 0.02, 0.03, 0.04},
			}},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			layers := make(map[string]*artificialLayer)

			for _, d := range tc.layerDescriptions {
				layers[d.UUID], _ = NewArtificialLayer(d, tc.rand)
			}

			for _, d := range tc.layerDescriptions {
				l := layers[d.UUID]
				if d.NextLayerUUID != "" {
					l.Next = layers[d.NextLayerUUID]
					layers[d.NextLayerUUID].Previous = l
				}

				layers[d.UUID] = l
			}

			for _, d := range tc.layerDescriptions {
				l := layers[d.UUID]
				if ld, ok := l.GetLayerDescription().(*ArtificialLayerDescriptor); !ok {
					t.Errorf("expected layer descriptor should be an ArtificialLayerDescriptor, but got %v", l.GetLayerDescription())
				} else {
					if ld.UUID != d.UUID {
						t.Errorf("expected UUID is %s, but got %s", d.UUID, ld.UUID)
					}

					if ld.NextLayerUUID != d.NextLayerUUID {
						t.Errorf("expected next layer UUID is %s, but got %s", d.NextLayerUUID, ld.NextLayerUUID)
					}

					if ld.InputShape.Rows != d.InputShape.Rows {
						t.Errorf("expected rows of the input shape is %d, but got %d", d.InputShape.Rows, ld.InputShape.Rows)
					}

					if ld.InputShape.Columns != d.InputShape.Columns {
						t.Errorf("expected columns of the input shape is %d, but got %d", d.InputShape.Columns, ld.InputShape.Columns)
					}

					if ld.InputShape.Depth != d.InputShape.Depth {
						t.Errorf("expected depth of the input shape is %d, but got %d", d.InputShape.Depth, ld.InputShape.Depth)
					}

					if ld.OutputShape.Rows != d.OutputShape.Rows {
						t.Errorf("expected rows of the output shape is %d, but got %d", d.OutputShape.Rows, ld.OutputShape.Rows)
					}

					if ld.OutputShape.Columns != d.OutputShape.Columns {
						t.Errorf("expected columns of the output shape is %d, but got %d", d.OutputShape.Columns, ld.OutputShape.Columns)
					}

					if ld.OutputShape.Depth != d.OutputShape.Depth {
						t.Errorf("expected depth of the output shape is %d, but got %d", d.OutputShape.Depth, ld.OutputShape.Depth)
					}

					if ld.ActivationFn != d.ActivationFn {
						t.Errorf("expected activation function is %s, but got %s", d.ActivationFn, ld.ActivationFn)
					}

					if len(ld.Weights) != len(d.Weights) {
						t.Errorf("expected length of the weights is %d, but got %d", len(d.Weights), len(ld.Weights))
					} else {
						for idx, w := range ld.Weights {
							if w != d.Weights[idx] {
								t.Errorf("expected weight[%d] is %f, but got %f", idx, d.Weights[idx], w)
							}
						}
					}

					if len(ld.Biases) != len(d.Biases) {
						t.Errorf("expected length of the biases is %d, but got %d", len(d.Biases), len(ld.Biases))
					} else {
						for idx, b := range ld.Biases {
							if b != d.Biases[idx] {
								t.Errorf("expected weight[%d] is %f, but got %f", idx, d.Biases[idx], b)
							}
						}
					}
				}
			}
		})
	}
}

func TestGetUUID_artificialLayer(t *testing.T) {
	learningRate := 0.1
	testCases := []struct {
		name              string
		rand              *rand.Rand
		layerDescriptions []ArtificialLayerDescriptor
	}{
		{"Normal", rand.New(rand.NewSource(0)), []ArtificialLayerDescriptor{
			{LayerDescriptor{"ARTIFICIAL_mdN6RA0rI0", "ARTIFICIAL_mdN6RA0rI1", Shape{2, 1, 1}, Shape{4, 1, 1}, &learningRate}, "ReLU",
				[]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}, []float64{0.01, 0.02, 0.03, 0.04},
			}, {LayerDescriptor{"ARTIFICIAL_mdN6RA0rI1", "", Shape{4, 1, 1}, Shape{4, 1, 1}, &learningRate}, "ReLU",
				[]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}, []float64{0.01, 0.02, 0.03, 0.04},
			}},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			layers := make(map[string]*artificialLayer)

			for _, d := range tc.layerDescriptions {
				layers[d.UUID], _ = NewArtificialLayer(d, tc.rand)
			}

			for _, d := range tc.layerDescriptions {
				l := layers[d.UUID]
				if d.NextLayerUUID != "" {
					l.Next = layers[d.NextLayerUUID]
					layers[d.NextLayerUUID].Previous = l
				}

				layers[d.UUID] = l
			}

			for _, d := range tc.layerDescriptions {
				l := layers[d.UUID]
				if l.GetUUID() != d.UUID {
					t.Errorf("expected UUID is %s, but got %s", d.UUID, l.GetUUID())
				}
			}
		})
	}
}
