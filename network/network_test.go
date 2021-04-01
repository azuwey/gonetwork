package network

import (
	"math"
	"math/rand"
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

func TestNew(t *testing.T) {
	rnd := rand.New(rand.NewSource(0))
	testCases := []struct {
		name          string
		rand          *rand.Rand
		model         *Model
		expectedError error
	}{
		{"Normal", rand.New(rand.NewSource(0)), &Model{0.1, []LayerDescriptor{
			{rnd.Intn(32) + 1, "", nil, nil},
			{rnd.Intn(32) + 1, "LogisticSigmoid", nil, nil},
			{rnd.Intn(32) + 1, "LogisticSigmoid", nil, nil},
		}}, nil},
		{"ErrLayerStructureLength", rand.New(rand.NewSource(0)), &Model{0.1, []LayerDescriptor{
			{rnd.Intn(32) + 1, "", nil, nil},
			{rnd.Intn(32) + 1, "", nil, nil},
		}}, ErrLayerStructureLength},
		{"ErrLearningRateRange <= 0", rand.New(rand.NewSource(0)), &Model{0, []LayerDescriptor{
			{rnd.Intn(32) + 1, "", nil, nil},
			{rnd.Intn(32) + 1, "LogisticSigmoid", nil, nil},
			{rnd.Intn(32) + 1, "LogisticSigmoid", nil, nil},
		}}, ErrLearningRateRange},
		{"ErrLearningRateRange > 1", rand.New(rand.NewSource(0)), &Model{1.1, []LayerDescriptor{
			{rnd.Intn(32) + 1, "", nil, nil},
			{rnd.Intn(32) + 1, "LogisticSigmoid", nil, nil},
			{rnd.Intn(32) + 1, "LogisticSigmoid", nil, nil},
		}}, ErrLearningRateRange},
		{"ErrNilRand", nil, &Model{0.1, []LayerDescriptor{
			{rnd.Intn(32) + 1, "", nil, nil},
			{rnd.Intn(32) + 1, "LogisticSigmoid", nil, nil},
			{rnd.Intn(32) + 1, "LogisticSigmoid", nil, nil},
		}}, ErrNilRand},
		{"ErrActivationFnNotExist", rand.New(rand.NewSource(0)), &Model{0.1, []LayerDescriptor{
			{rnd.Intn(32) + 1, "", nil, nil},
			{rnd.Intn(32) + 1, "", nil, nil},
			{rnd.Intn(32) + 1, "", nil, nil},
		}}, ErrActivationFnNotExist},
		{"matrix.ErrZeroRow", rand.New(rand.NewSource(0)), &Model{0.1, []LayerDescriptor{
			{rnd.Intn(32) + 1, "", nil, nil},
			{0, "", nil, nil},
			{rnd.Intn(32) + 1, "", nil, nil},
		}}, matrix.ErrZeroRow},
		{"matrix.ErrZeroCol", rand.New(rand.NewSource(0)), &Model{0.1, []LayerDescriptor{
			{0, "", nil, nil},
			{rnd.Intn(32) + 1, "", nil, nil},
			{rnd.Intn(32) + 1, "", nil, nil},
		}}, matrix.ErrZeroCol},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			n, err := New(tc.model, tc.rand)
			if tc.expectedError != nil {
				if err != tc.expectedError {
					t.Errorf("Expected error is %v, but got %v", tc.expectedError, err)
				}
			} else if err != nil {
				t.Errorf("Expected error is %v, but got %v", nil, err)
			} else if n == nil {
				t.Error("Network should not be nil")
			} else {
				if len(n.layers) != len(tc.model.Layers)-1 {
					t.Errorf("Expected length of layers is %d, but got %d", len(tc.model.Layers)-1, len(n.layers))
				}

				for idx, l := range n.layers {
					if l.activationFunction != ActivationFunctions[tc.model.Layers[idx+1].ActivationFunction] {
						t.Errorf("Expected activation function is %v, but got %v", tc.model.Layers[idx+1].ActivationFunction, l.activationFunction)
					}

					if l.weights == nil {
						t.Error("Weights should not be nil")
					}

					if l.weights.Rows != tc.model.Layers[idx+1].Nodes {
						t.Errorf("Expected rows of weights is %d, but got %d", tc.model.Layers[idx+1].Nodes, l.weights.Rows)
					}

					if l.weights.Columns != tc.model.Layers[idx].Nodes {
						t.Errorf("Expected columns of weights is %d, but got %d", tc.model.Layers[idx].Nodes, l.weights.Columns)
					}

					if l.biases == nil {
						t.Error("Biases should not be nil")
					}

					if l.biases.Rows != tc.model.Layers[idx+1].Nodes {
						t.Errorf("Expected rows of weights is %d, but got %d", tc.model.Layers[idx+1].Nodes, l.biases.Rows)
					}

					if l.biases.Columns != 1 {
						t.Errorf("Expected columns of weights is %d, but got %d", 1, l.biases.Columns)
					}
				}

				if n.learningRate != tc.model.LearningRate {
					t.Errorf("Expected learning rate is %f, but got %f", tc.model.LearningRate, n.learningRate)
				}

				if n.rand != tc.rand {
					t.Errorf("Expected randomizer is %v, but got %v", tc.rand, n.rand)
				}
			}
		})
	}
}

func TestCalculateLayerValues(t *testing.T) {
	testCases := []struct {
		name           string
		inputs         []float64
		expectedValues [][]float64
		expectedError  error
	}{
		{"Normal", []float64{0.5}, [][]float64{{0.500000}, {0.483787}, {0.322914}}, nil},
		{"matrix.ErrZeroRow", []float64{}, [][]float64{}, matrix.ErrZeroRow},
		{"matrix.ErrZeroRow", nil, [][]float64{}, matrix.ErrZeroRow},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			n, _ := New(&Model{0.1, []LayerDescriptor{
				{len(tc.inputs), "", nil, nil},
				{len(tc.inputs), "LogisticSigmoid", nil, nil},
				{len(tc.inputs), "LogisticSigmoid", nil, nil},
			}}, rand.New(rand.NewSource(0)))

			vals, err := n.calculateLayerValues(tc.inputs)
			if tc.expectedError != nil {
				if err != tc.expectedError {
					t.Errorf("Expected error is %v, but got %v", tc.expectedError, err)
				}
			} else if err != nil {
				t.Errorf("Expected error is %v, but got %v", nil, err)
			} else if n == nil {
				t.Error("Network should not be nil")
			} else {
				if len(vals) != len(tc.expectedValues) {
					t.Errorf("Expected length of layer values is %d, but got %d", len(tc.expectedValues), len(vals))
				}

				for lvIdx, lv := range vals {
					if len(lv.activated.Values) != len(tc.expectedValues[lvIdx]) {
						t.Errorf("Expected length of values is %d, but got %d", len(tc.expectedValues[lvIdx]), len(lv.activated.Values))
					}
					for vIdx, v := range lv.activated.Values {
						if !isFloatInThreshold(v, tc.expectedValues[lvIdx][vIdx], 0.03) {
							t.Errorf("Expected value of the layer is %f, but got %f", tc.expectedValues[lvIdx][vIdx], v)
						}
					}
				}
			}
		})
	}
}

func TestPredict(t *testing.T) {
	testCases := []struct {
		name            string
		inputs, targets []float64
		expectedError   error
	}{
		{"Normal", []float64{0.5}, []float64{0.322914}, nil},
		{"ErrNilInputSlice", nil, []float64{}, ErrNilInputSlice},
		{"matrix.ErrZeroRow", []float64{}, []float64{}, matrix.ErrZeroRow},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			n, _ := New(&Model{0.1, []LayerDescriptor{
				{len(tc.inputs), "", nil, nil},
				{len(tc.inputs), "LogisticSigmoid", nil, nil},
				{len(tc.inputs), "LogisticSigmoid", nil, nil},
			}}, rand.New(rand.NewSource(0)))

			vals, err := n.Predict(tc.inputs)
			if tc.expectedError != nil {
				if err != tc.expectedError {
					t.Errorf("Expected error is %v, but got %v", tc.expectedError, err)
				}
			} else if err != nil {
				t.Errorf("Expected error is %v, but got %v", nil, err)
			} else if n == nil {
				t.Error("Network should not be nil")
			} else {
				if len(vals) != len(tc.targets) {
					t.Errorf("Expected length of layer values is %d, but got %d", len(tc.targets), len(vals))
				}

				for idx, v := range vals {
					if !isFloatInThreshold(v, tc.targets[idx], 0.03) {
						t.Errorf("Expected value of the layer is %f, but got %f", tc.targets[idx], v)
					}
				}
			}
		})
	}
}

func TestTrain(t *testing.T) {
	type dataSet struct{ inputs, targets []float64 }
	testCases := []struct {
		name                   string
		learningData, testData []dataSet
		model                  *Model
		expectedError          error
	}{
		{
			"Normal", []dataSet{
				{[]float64{1, 0}, []float64{0}},
			}, []dataSet{
				{[]float64{1, 0}, []float64{0.590543}},
			}, &Model{0.1, []LayerDescriptor{
				{2, "", nil, nil},
				{2, "LogisticSigmoid", nil, nil},
				{1, "LogisticSigmoid", nil, nil},
			}}, nil,
		},
		{
			"ErrNilInputSlice", []dataSet{
				{nil, []float64{}},
			}, []dataSet{}, &Model{0.1, []LayerDescriptor{
				{1, "", nil, nil},
				{1, "LogisticSigmoid", nil, nil},
				{1, "LogisticSigmoid", nil, nil},
			}}, ErrNilInputSlice,
		},
		{
			"ErrNilTargetSlice", []dataSet{
				{[]float64{}, nil},
			}, []dataSet{}, &Model{0.1, []LayerDescriptor{
				{1, "", nil, nil},
				{1, "LogisticSigmoid", nil, nil},
				{1, "LogisticSigmoid", nil, nil},
			}}, ErrNilTargetSlice,
		},
		{
			"ErrBadTargetSlice", []dataSet{
				{[]float64{0.5}, []float64{0.5, 0.5}},
			}, []dataSet{}, &Model{0.1, []LayerDescriptor{
				{1, "", nil, nil},
				{1, "LogisticSigmoid", nil, nil},
				{1, "LogisticSigmoid", nil, nil},
			}}, ErrBadTargetSlice,
		},
		{
			"matrix.ErrZeroRow target matrix", []dataSet{
				{[]float64{0.5}, []float64{}},
			}, []dataSet{}, &Model{0.1, []LayerDescriptor{
				{1, "", nil, nil},
				{1, "LogisticSigmoid", nil, nil},
				{1, "LogisticSigmoid", nil, nil},
			}}, matrix.ErrZeroRow,
		},
		{
			"matrix.ErrZeroRow input matrix", []dataSet{
				{[]float64{}, []float64{0.5}},
			}, []dataSet{}, &Model{0.1, []LayerDescriptor{
				{1, "", nil, nil},
				{1, "LogisticSigmoid", nil, nil},
				{1, "LogisticSigmoid", nil, nil},
			}}, matrix.ErrZeroRow,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			r := rand.New(rand.NewSource(0))
			n, _ := New(tc.model, r)
			if n == nil {
				t.Error("Network should not be nil")
			}

			for _, ld := range tc.learningData {
				err := n.Train(ld.inputs, ld.targets)
				if tc.expectedError != nil {
					if err != tc.expectedError {
						t.Errorf("Expected error is %v, but got %v", tc.expectedError, err)
					}
				} else if err != nil {
					t.Errorf("Expected error is %v, but got %v", nil, err)
					t.FailNow()
				}
			}

			for _, td := range tc.testData {
				predictions, _ := n.Predict(td.inputs)
				for idx, p := range predictions {
					if !isFloatInThreshold(p, td.targets[idx], 0.03) {
						t.Errorf("Expected prediction is %f, but got %f", td.targets[idx], p)
					}
				}
			}
		})
	}
}

/*func TestTrain_long(t *testing.T) {
	// TODO: move this to an example
	type dataSet struct{ inputs, targets []float64 }
	testCases := []struct {
		name                   string
		epoch                  int
		learningData, testData []dataSet
		model                  *Model
		expectedError          error
	}{
		{
			"XOR", 30000, []dataSet{
				{[]float64{0, 0}, []float64{0}},
				{[]float64{0, 1}, []float64{1}},
				{[]float64{1, 0}, []float64{1}},
				{[]float64{1, 1}, []float64{0}},
			}, []dataSet{
				{[]float64{0, 0}, []float64{0}},
				{[]float64{0, 1}, []float64{1}},
				{[]float64{1, 0}, []float64{1}},
				{[]float64{1, 1}, []float64{0}},
			}, &Model{0.1, []LayerDescriptor{
				{2, "", nil, nil},
				{2, "TanH", nil, nil},
				{1, "LogisticSigmoid", nil, nil},
			}}, nil,
		},
		{
			"4-bit counter", 30000, []dataSet{
				{[]float64{0, 0, 0, 0}, []float64{0, 0, 0, 1}},
				{[]float64{0, 0, 0, 1}, []float64{0, 0, 1, 0}},
				{[]float64{0, 0, 1, 0}, []float64{0, 0, 1, 1}},
				{[]float64{0, 0, 1, 1}, []float64{0, 1, 0, 0}},
				{[]float64{0, 1, 0, 0}, []float64{0, 1, 0, 1}},
				{[]float64{0, 1, 0, 1}, []float64{0, 1, 1, 0}},
				{[]float64{0, 1, 1, 0}, []float64{0, 1, 1, 1}},
				{[]float64{0, 1, 1, 1}, []float64{1, 0, 0, 0}},
				{[]float64{1, 0, 0, 0}, []float64{1, 0, 0, 1}},
				{[]float64{1, 0, 0, 1}, []float64{1, 0, 1, 0}},
				{[]float64{1, 0, 1, 0}, []float64{1, 0, 1, 1}},
				{[]float64{1, 0, 1, 1}, []float64{1, 1, 0, 0}},
				{[]float64{1, 1, 0, 0}, []float64{1, 1, 0, 1}},
				{[]float64{1, 1, 1, 0}, []float64{1, 1, 1, 1}},
			}, []dataSet{
				{[]float64{1, 1, 0, 1}, []float64{1, 1, 1, 0}},
			}, &Model{0.3, []LayerDescriptor{
				{4, "", nil, nil},
				{16, "TanH", nil, nil},
				{4, "LogisticSigmoid", nil, nil},
			}}, nil,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			r := rand.New(rand.NewSource(0))
			n, _ := New(tc.model, r)
			if n == nil {
				t.Error("Network should not be nil")
			}

			for e := 0; e < tc.epoch; e++ {
				r.Shuffle(len(tc.learningData), func(i, j int) {
					tc.learningData[i], tc.learningData[j] = tc.learningData[j], tc.learningData[i]
				})

				for _, ld := range tc.learningData {
					err := n.Train(ld.inputs, ld.targets)
					if tc.expectedError != nil {
						if err != tc.expectedError {
							t.Errorf("Expected error is %v, but got %v", tc.expectedError, err)
						}
					} else if err != nil {
						t.Errorf("Expected error is %v, but got %v", nil, err)
						t.FailNow()
					}
				}
			}

			for _, td := range tc.testData {
				predictions, _ := n.Predict(td.inputs)
				for idx, p := range predictions {
					if !isFloatInThreshold(p, td.targets[idx]) {
						t.Errorf("Expected prediction is %f, but got %f", td.targets[idx], p)
					}
				}
			}
		})
	}
}*/
