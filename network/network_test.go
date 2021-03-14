package network

import (
	"errors"
	"math"
	"math/rand"
	"testing"

	"github.com/azuwey/gonetwork/matrix"
)

func TestNew(t *testing.T) {
	// TODO
}

func TestFeedForward(t *testing.T) {
	inputs, _ := matrix.New(2, 1, []float64{0, 1})
	network, _ := New(2, 2, 1, 0.1)
	guess, _ := network.FeedForward(inputs, func(v float64, _, _ int) float64 {
		return 1 / (1 + math.Exp(-v))
	})

	t.Log(guess.Raw())
	t.Fail()
}

func TestTrain_xor(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode.")
	}

	trainingDatas := []struct {
		inputs  []float64
		targets []float64
	}{
		{[]float64{0, 0}, []float64{0}},
		{[]float64{0, 1}, []float64{1}},
		{[]float64{1, 0}, []float64{1}},
		{[]float64{1, 1}, []float64{0}},
	}

	network, _ := New(2, 2, 1, 0.1)
	sigmoid := func(v float64, _, _ int) float64 {
		return 1 / (1 + math.Exp(-v))
	}

	derivativeSigmoid := func(v float64, r, c int) float64 {
		return sigmoid(v, r, c) * (1 - sigmoid(v, r, c))
	}

	for i := 0; i < 500000; i++ {
		traingingDataIndex := rand.Intn(len(trainingDatas))
		inputs, _ := matrix.New(2, 1, trainingDatas[traingingDataIndex].inputs)
		targets, _ := matrix.New(1, 1, trainingDatas[traingingDataIndex].targets)
		if err := network.Train(inputs, targets, sigmoid, derivativeSigmoid); !errors.Is(err, nil) {
			t.Log(err)
			t.Fail()
		}
	}

	for _, trainingData := range trainingDatas {
		inputs, _ := matrix.New(2, 1, trainingData.inputs)
		guesses, _ := network.FeedForward(inputs, sigmoid)

		for _, target := range trainingData.targets {
			if target == 0 {
				for _, guess := range guesses.Raw() {
					if guess >= 0.02 {
						t.Logf("Guessing target failed for %v, guess expected to be <0.02, but got %f", trainingData.inputs, guess)
						t.Fail()
					}
				}
			}

			if target == 1 {
				for _, guess := range guesses.Raw() {
					if guess <= 0.98 {
						t.Logf("Guessing target failed for %v, guess expected to be >0.98, but got %f", trainingData.inputs, guess)
						t.Fail()
					}
				}
			}
		}
	}
}
