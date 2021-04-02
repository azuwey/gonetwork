package network

import (
	"math/rand"
	"testing"
)

func BenchmarkNew(b *testing.B) {
	model := &Model{0.1, []LayerDescriptor{
		{1, "", nil, nil},
		{1, "LogisticSigmoid", nil, nil},
		{1, "LogisticSigmoid", nil, nil},
	}}
	rnd := rand.New(rand.NewSource(0))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		New(model, rnd)
	}
}

func BenchmarkCalculateLayerValues(b *testing.B) {
	model := &Model{0.1, []LayerDescriptor{
		{1, "", nil, nil},
		{1, "LogisticSigmoid", nil, nil},
		{1, "LogisticSigmoid", nil, nil},
	}}
	rnd := rand.New(rand.NewSource(0))
	n, _ := New(model, rnd)
	inputs := []float64{rnd.Float64()}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		n.calculateLayerValues(inputs)
	}
}

func BenchmarkPredict(b *testing.B) {
	model := &Model{0.1, []LayerDescriptor{
		{1, "", nil, nil},
		{1, "LogisticSigmoid", nil, nil},
		{1, "LogisticSigmoid", nil, nil},
	}}
	rnd := rand.New(rand.NewSource(0))
	n, _ := New(model, rnd)
	inputs := []float64{rnd.Float64()}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		n.Predict(inputs)
	}
}

func BenchmarkTrain(b *testing.B) {
	model := &Model{0.1, []LayerDescriptor{
		{1, "", nil, nil},
		{1, "LogisticSigmoid", nil, nil},
		{1, "LogisticSigmoid", nil, nil},
	}}
	rnd := rand.New(rand.NewSource(0))
	n, _ := New(model, rnd)
	inputs := []float64{rnd.Float64()}
	target := []float64{rnd.Float64()}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		n.Train(inputs, target)
	}
}
