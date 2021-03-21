package network

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"testing"
)

const float64EqualityThreshold = 0.000001

func TestNew(t *testing.T) {
	testCases := []struct {
		inputs, targets    []float64
		activationFunction *ActivationFunction
		rnd                *rand.Rand
		expectedError      error
	}{
		{[]float64{0, 0}, []float64{0.165942}, LogisticSigmoid, rand.New(rand.NewSource(0)), nil},
		{[]float64{0, 0}, []float64{0.165942}, LogisticSigmoid, nil, nil},
	}

	for tcIndex, tcValue := range testCases {
		tcValue, tcIndex := tcValue, tcIndex // capture range variables
		t.Run(fmt.Sprintf("[%d] %+v", tcIndex, tcValue), func(t *testing.T) {
			t.Parallel()
		})
	}
}

func TestPredict(t *testing.T) {
	testCases := []struct {
		inputs, targets []float64
		layers          []Layer
		expectedError   error
	}{
		{[]float64{0, 0}, []float64{0.657095}, []Layer{{2, nil}, {2, LogisticSigmoid}, {1, LogisticSigmoid}}, nil},
		{[]float64{0, 0}, []float64{0.680450}, []Layer{{2, nil}, {2, TanH}, {1, TanH}}, nil},
		{[]float64{0, 0}, []float64{0.794339}, []Layer{{2, nil}, {2, ReLU}, {1, ReLU}}, nil},
		{[]float64{0, 0}, []float64{0.800759}, []Layer{{2, nil}, {2, LeakyReLU}, {1, LeakyReLU}}, nil},
		{[]float64{0, 0}, []float64{0.189601, 0.810399}, []Layer{{2, nil}, {20, LogisticSigmoid}, {2, Softmax}}, nil},
		{[]float64{0, 0}, []float64{0.139501, 0.860499}, []Layer{{2, nil}, {20, TanH}, {2, Softmax}}, nil},
		{[]float64{0, 0}, []float64{0.097127, 0.902873}, []Layer{{2, nil}, {20, ReLU}, {2, Softmax}}, nil},
		{[]float64{0, 0}, []float64{0.102186, 0.897814}, []Layer{{2, nil}, {20, LeakyReLU}, {2, Softmax}}, nil},
	}

	for tcIndex, tcValue := range testCases {
		tcValue, tcIndex := tcValue, tcIndex // capture range variables
		t.Run(fmt.Sprintf("[%d] %+v", tcIndex, tcValue), func(t *testing.T) {
			t.Parallel()

			r := rand.New(rand.NewSource(0))
			n, _ := New(tcValue.layers, 0.1, r)
			prediction, err := n.Predict(tcValue.inputs)

			if tcValue.expectedError != nil {
				if !errors.Is(err, tcValue.expectedError) {
					t.Logf("Err should be %v but it's %v", tcValue.expectedError, err)
					t.Fail()
				}
				return
			}

			if tcValue.targets != nil {
				if len(prediction) != len(tcValue.targets) {
					t.Logf("Lenght of the prediction should be %d but it's %d", len(tcValue.targets), len(prediction))
					t.Fail()
				}

				for i, v := range prediction {
					if math.Abs(v-tcValue.targets[i]) > float64EqualityThreshold {
						t.Logf("Value of the prediction should be %f but it's %f", tcValue.targets[i], v)
						t.Fail()
					}
				}
			} else if prediction != nil {
				t.Logf("Prediction should be nil, but it's %+v", prediction)
				t.Fail()
			}
		})
	}
}

func TestTrain(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode.")
	}

	type dataSet struct{ inputs, targets []float64 }
	testCases := []struct {
		name                   string
		learningRate           float64
		epocs                  int
		expectedError          error
		layerStructure         []Layer
		learningData, testData []dataSet
	}{
		{
			"XOR", 0.01, 500000, nil, []Layer{
				{2, nil}, {2, LeakyReLU}, {1, LeakyReLU},
			}, []dataSet{
				{[]float64{0, 0}, []float64{0}},
				{[]float64{0, 1}, []float64{1}},
				{[]float64{1, 0}, []float64{1}},
				{[]float64{1, 1}, []float64{0}},
			}, []dataSet{
				{[]float64{0, 0}, []float64{0}},
				{[]float64{0, 1}, []float64{1}},
				{[]float64{1, 0}, []float64{1}},
				{[]float64{1, 1}, []float64{0}},
			},
		},
	}

	for _, tcValue := range testCases {
		tcValue := tcValue // capture range variables
		t.Run(tcValue.name, func(t *testing.T) {
			t.Parallel()

			r := rand.New(rand.NewSource(0))
			n, _ := New(tcValue.layerStructure, tcValue.learningRate, r)

			for i := 0; i < tcValue.epocs; i++ {
				traingingDataIndex := rand.Intn(len(tcValue.learningData))
				if err := n.Train(tcValue.learningData[traingingDataIndex].inputs, tcValue.learningData[traingingDataIndex].targets); !errors.Is(err, nil) {
					if tcValue.expectedError == nil {
						t.Logf("Err should be %v but it's %v", tcValue.expectedError, err)
						t.Fail()
					} else {
						return
					}
				}
			}

			for _, d := range tcValue.testData {
				guesses, _ := n.Predict(d.inputs)

				for _, target := range d.targets {
					if target == 0 {
						for i, v := range guesses {
							if math.Abs(v-d.targets[i]) > float64EqualityThreshold {
								t.Logf("Value of the prediction should be %f but it's %f, targets: %v, inputs: %v", d.targets[i], v, d.targets, d.inputs)
								t.Fail()
							}
						}
					}
				}
			}
		})
	}
}
