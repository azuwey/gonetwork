package network

import (
	"math"
	"math/rand"
	"testing"

	"github.com/azuwey/gonetwork/matrix"
)

const float64EqualityThreshold = 0.05

func isFloatInThreshold(v float64, t float64) bool {
	if math.Abs(v-t) <= float64EqualityThreshold {
		return true
	} else {
		return false
	}
}

var dummyActivationFunction *ActivationFunction = &ActivationFunction{
	aFn: func(_ *matrix.Matrix) matrix.ApplyFn {
		return func(v float64, _, _ int, _ []float64) float64 {
			return v
		}
	},
	dFn: func(_ *matrix.Matrix) matrix.ApplyFn {
		return func(v float64, _, _ int, _ []float64) float64 {
			return 1
		}
	},
}

func TestNew(t *testing.T) {
	rnd := rand.New(rand.NewSource(0))
	testCases := []struct {
		name          string
		learningRate  float64
		rand          *rand.Rand
		layers        []Layer
		expectedError error
	}{
		{"Normal", 0.1, rand.New(rand.NewSource(0)), []Layer{{rnd.Intn(32) + 1, nil}, {rnd.Intn(32) + 1, dummyActivationFunction}, {rnd.Intn(32) + 1, dummyActivationFunction}}, nil},
		{"ErrLayerStructureLength", 0.1, rand.New(rand.NewSource(0)), []Layer{{rnd.Intn(32) + 1, nil}, {rnd.Intn(32) + 1, nil}}, ErrLayerStructureLength},
		{"ErrLearningRateRange < 0", -0, rand.New(rand.NewSource(0)), []Layer{{rnd.Intn(32) + 1, nil}, {rnd.Intn(32) + 1, nil}, {rnd.Intn(32) + 1, nil}}, ErrLearningRateRange},
		{"ErrLearningRateRange > 1", 1.1, rand.New(rand.NewSource(0)), []Layer{{rnd.Intn(32) + 1, nil}, {rnd.Intn(32) + 1, nil}, {rnd.Intn(32) + 1, nil}}, ErrLearningRateRange},
		{"ErrNilRand", 0.1, nil, []Layer{{1, nil}, {1, nil}, {1, nil}}, ErrNilRand},
		{"ErrNilActivationFn", 0.1, rand.New(rand.NewSource(0)), []Layer{{rnd.Intn(32) + 1, nil}, {rnd.Intn(32) + 1, nil}, {rnd.Intn(32) + 1, dummyActivationFunction}}, ErrNilActivationFn},
		{"matrix.ErrZeroRow", 0.1, rand.New(rand.NewSource(0)), []Layer{{rnd.Intn(32) + 1, nil}, {0, dummyActivationFunction}, {rnd.Intn(32) + 1, dummyActivationFunction}}, matrix.ErrZeroRow},
		{"matrix.ErrZeroCol", 0.1, rand.New(rand.NewSource(0)), []Layer{{0, nil}, {rnd.Intn(32) + 1, dummyActivationFunction}, {rnd.Intn(32) + 1, dummyActivationFunction}}, matrix.ErrZeroCol},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			n, err := New(tc.layers, tc.learningRate, tc.rand)
			if tc.expectedError != nil {
				if err != tc.expectedError {
					t.Errorf("Expected error is %v, but got %v", tc.expectedError, err)
				}
			} else if err != nil {
				t.Errorf("Expected error is %v, but got %v", nil, err)
			} else if n == nil {
				t.Error("Network should not be nil")
			} else {
				if len(n.layers) != len(tc.layers)-1 {
					t.Errorf("Expected length of layers is %d, but got %d", len(tc.layers)-1, len(n.layers))
				}

				for idx, l := range n.layers {
					if l.activationFunction != tc.layers[idx+1].ActivationFunction {
						t.Errorf("Expected activation function is %v, but got %v", tc.layers[idx+1].ActivationFunction, l.activationFunction)
					}

					if l.weights == nil {
						t.Error("Weights should not be nil")
					}

					if l.weights.Rows != tc.layers[idx+1].Nodes {
						t.Errorf("Expected rows of weights is %d, but got %d", tc.layers[idx+1].Nodes, l.weights.Rows)
					}

					if l.weights.Columns != tc.layers[idx].Nodes {
						t.Errorf("Expected columns of weights is %d, but got %d", tc.layers[idx].Nodes, l.weights.Columns)
					}

					if l.biases == nil {
						t.Error("Biases should not be nil")
					}

					if l.biases.Rows != tc.layers[idx+1].Nodes {
						t.Errorf("Expected rows of weights is %d, but got %d", tc.layers[idx+1].Nodes, l.biases.Rows)
					}

					if l.biases.Columns != 1 {
						t.Errorf("Expected columns of weights is %d, but got %d", 1, l.biases.Columns)
					}
				}

				if n.learningRate != tc.learningRate {
					t.Errorf("Expected learning rate is %f, but got %f", tc.learningRate, n.learningRate)
				}

				if n.rand != tc.rand {
					t.Errorf("Expected randomizer is %v, but got %v", tc.rand, n.rand)
				}
			}
		})
	}
}

func Test_calculateLayerValues(t *testing.T) {
	testCases := []struct {
		name           string
		inputs         []float64
		expectedValues [][]float64
		expectedError  error
	}{
		{"Normal", []float64{0.5}, [][]float64{{0.500000}, {-0.064874}, {-0.911547}}, nil},
		{"matrix.ErrZeroRow", []float64{}, [][]float64{}, matrix.ErrZeroRow},
		{"matrix.ErrZeroRow", nil, [][]float64{}, matrix.ErrZeroRow},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			n, _ := New([]Layer{
				{len(tc.inputs), nil},
				{len(tc.inputs), dummyActivationFunction},
				{len(tc.inputs), dummyActivationFunction},
			}, 0.1, rand.New(rand.NewSource(0)))

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
					if len(lv.Values) != len(tc.expectedValues[lvIdx]) {
						t.Errorf("Expected length of values is %d, but got %d", len(tc.expectedValues[lvIdx]), len(lv.Values))
					}
					for vIdx, v := range lv.Values {
						if !isFloatInThreshold(v, tc.expectedValues[lvIdx][vIdx]) {
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
		{"Normal", []float64{0.5}, []float64{-0.911547}, nil},
		{"ErrNilInputSlice", nil, []float64{}, ErrNilInputSlice},
		{"matrix.ErrZeroRow", []float64{}, []float64{}, matrix.ErrZeroRow},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			n, _ := New([]Layer{
				{len(tc.inputs), nil},
				{len(tc.inputs), dummyActivationFunction},
				{len(tc.inputs), dummyActivationFunction},
			}, 0.1, rand.New(rand.NewSource(0)))

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
					if !isFloatInThreshold(v, tc.targets[idx]) {
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
		learningRate           float64
		learningData, testData []dataSet
		layers                 []Layer
		expectedError          error
	}{
		{
			"Normal", 0.1, []dataSet{
				{[]float64{1, 0}, []float64{0}},
			}, []dataSet{
				{[]float64{1, 0}, []float64{-0.224604}},
			}, []Layer{
				{2, nil},
				{2, dummyActivationFunction},
				{1, dummyActivationFunction},
			}, nil,
		},
		{
			"ErrNilInputSlice", 0.1, []dataSet{
				{nil, []float64{}},
			}, []dataSet{}, []Layer{
				{1, nil},
				{1, dummyActivationFunction},
				{1, dummyActivationFunction},
			}, ErrNilInputSlice,
		},
		{
			"ErrNilTargetSlice", 0.1, []dataSet{
				{[]float64{}, nil},
			}, []dataSet{}, []Layer{
				{1, nil},
				{1, dummyActivationFunction},
				{1, dummyActivationFunction},
			}, ErrNilTargetSlice,
		},
		{
			"ErrBadTargetSlice", 0.1, []dataSet{
				{[]float64{0.5}, []float64{0.5, 0.5}},
			}, []dataSet{}, []Layer{
				{1, nil},
				{1, dummyActivationFunction},
				{1, dummyActivationFunction},
			}, ErrBadTargetSlice,
		},
		{
			"matrix.ErrZeroRow target matrix", 0.1, []dataSet{
				{[]float64{0.5}, []float64{}},
			}, []dataSet{}, []Layer{
				{1, nil},
				{1, dummyActivationFunction},
				{1, dummyActivationFunction},
			}, matrix.ErrZeroRow,
		},
		{
			"matrix.ErrZeroRow input matrix", 0.1, []dataSet{
				{[]float64{}, []float64{0.5}},
			}, []dataSet{}, []Layer{
				{1, nil},
				{1, dummyActivationFunction},
				{1, dummyActivationFunction},
			}, matrix.ErrZeroRow,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			r := rand.New(rand.NewSource(0))
			n, _ := New(tc.layers, tc.learningRate, r)
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
					if !isFloatInThreshold(p, td.targets[idx]) {
						t.Errorf("Expected prediction is %f, but got %f", td.targets[idx], p)
					}
				}
			}
		})
	}
}

func TestTrain_long(t *testing.T) {
	// TODO: move this to an example
	type dataSet struct{ inputs, targets []float64 }
	testCases := []struct {
		name                   string
		learningRate           float64
		epoch                  int
		learningData, testData []dataSet
		layers                 []Layer
		expectedError          error
	}{
		{
			"XOR", 0.1, 30000, []dataSet{
				{[]float64{0, 0}, []float64{0}},
				{[]float64{0, 1}, []float64{1}},
				{[]float64{1, 0}, []float64{1}},
				{[]float64{1, 1}, []float64{0}},
			}, []dataSet{
				{[]float64{0, 0}, []float64{0}},
				{[]float64{0, 1}, []float64{1}},
				{[]float64{1, 0}, []float64{1}},
				{[]float64{1, 1}, []float64{0}},
			}, []Layer{
				{2, nil},
				{2, TanH},
				{1, LogisticSigmoid},
			}, nil,
		},
		{
			"4-bit counter", 0.1, 30000, []dataSet{
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
			}, []Layer{
				{4, nil},
				{16, TanH},
				{4, LogisticSigmoid},
			}, nil,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			r := rand.New(rand.NewSource(0))
			n, _ := New(tc.layers, tc.learningRate, r)
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
}
