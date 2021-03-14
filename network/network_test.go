package network

import (
	"errors"
	"math"
	"testing"

	"github.com/azuwey/gonetwork/matrix"
)

func TestNew(t *testing.T) {
	// TODO
}

func TestFeedForward(t *testing.T) {
	inputs, _ := matrix.New(2, 1, []float64{0, 1})
	network, _ := New(2, 2, 1)
	guess, _ := network.FeedForward(inputs, func(v float64, _, _ int) float64 {
		return 1 / (1 + math.Exp(-v))
	})

	t.Log(guess.Raw())
	t.Fail()
}

func TestTrain(t *testing.T) {
	inputs, _ := matrix.New(2, 1, []float64{0, 1})
	targets, _ := matrix.New(2, 1, []float64{1, 0})
	network, _ := New(2, 2, 2)
	guess, err := network.Train(inputs, targets, func(v float64, _, _ int) float64 {
		return 1 / (1 + math.Exp(-v))
	})

	if !errors.Is(err, nil) {
		t.Log(err)
		t.Fail()
	} else {
		t.Log(guess.Raw())
		t.Fail()
	}
}
