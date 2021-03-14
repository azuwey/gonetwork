package network

import (
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
