package network

import (
	"math/rand"
	"testing"

	"github.com/azuwey/gonetwork/matrix"
)

func BenchmarkLogisticSigmoidActivation(b *testing.B) {
	rnd := rand.New(rand.NewSource(0))
	inputs := []float64{rnd.Float64(), rnd.Float64()}
	m, _ := matrix.New(len(inputs), 1, inputs)
	aFn := LogisticSigmoid.aFn(m)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Apply(aFn, m)
	}
}

func BenchmarkLogisticSigmoidDeactivation(b *testing.B) {
	rnd := rand.New(rand.NewSource(0))
	inputs := []float64{rnd.Float64(), rnd.Float64()}
	m, _ := matrix.New(len(inputs), 1, inputs)
	dFn := LogisticSigmoid.dFn(m)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Apply(dFn, m)
	}
}

func BenchmarkTanHActivation(b *testing.B) {
	rnd := rand.New(rand.NewSource(0))
	inputs := []float64{rnd.Float64(), rnd.Float64()}
	m, _ := matrix.New(len(inputs), 1, inputs)
	aFn := TanH.aFn(m)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Apply(aFn, m)
	}
}

func BenchmarkTanHDeactivation(b *testing.B) {
	rnd := rand.New(rand.NewSource(0))
	inputs := []float64{rnd.Float64(), rnd.Float64()}
	m, _ := matrix.New(len(inputs), 1, inputs)
	dFn := TanH.dFn(m)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Apply(dFn, m)
	}
}

func BenchmarkReLUActivation(b *testing.B) {
	rnd := rand.New(rand.NewSource(0))
	inputs := []float64{rnd.Float64(), rnd.Float64()}
	m, _ := matrix.New(len(inputs), 1, inputs)
	aFn := ReLU.aFn(m)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Apply(aFn, m)
	}
}

func BenchmarkReLUDeactivation(b *testing.B) {
	rnd := rand.New(rand.NewSource(0))
	inputs := []float64{rnd.Float64(), rnd.Float64()}
	m, _ := matrix.New(len(inputs), 1, inputs)
	dFn := ReLU.dFn(m)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Apply(dFn, m)
	}
}

func BenchmarkLeakyReLUActivation(b *testing.B) {
	rnd := rand.New(rand.NewSource(0))
	inputs := []float64{rnd.Float64(), rnd.Float64()}
	m, _ := matrix.New(len(inputs), 1, inputs)
	aFn := LeakyReLU.aFn(m)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Apply(aFn, m)
	}
}

func BenchmarkLeakyReLUDeactivation(b *testing.B) {
	rnd := rand.New(rand.NewSource(0))
	inputs := []float64{rnd.Float64(), rnd.Float64()}
	m, _ := matrix.New(len(inputs), 1, inputs)
	dFn := LeakyReLU.dFn(m)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Apply(dFn, m)
	}
}

func BenchmarkSoftmaxActivation(b *testing.B) {
	rnd := rand.New(rand.NewSource(0))
	inputs := []float64{rnd.Float64(), rnd.Float64()}
	m, _ := matrix.New(len(inputs), 1, inputs)
	aFn := Softmax.aFn(m)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Apply(aFn, m)
	}
}

func BenchmarkSoftmaxDeactivation(b *testing.B) {
	rnd := rand.New(rand.NewSource(0))
	inputs := []float64{rnd.Float64(), rnd.Float64()}
	m, _ := matrix.New(len(inputs), 1, inputs)
	dFn := Softmax.dFn(m)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Apply(dFn, m)
	}
}

func BenchmarkStableSoftmaxActivation(b *testing.B) {
	rnd := rand.New(rand.NewSource(0))
	inputs := []float64{rnd.Float64(), rnd.Float64()}
	m, _ := matrix.New(len(inputs), 1, inputs)
	aFn := StableSoftmax.aFn(m)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Apply(aFn, m)
	}
}

func BenchmarkStableSoftmaxDeactivation(b *testing.B) {
	rnd := rand.New(rand.NewSource(0))
	inputs := []float64{rnd.Float64(), rnd.Float64()}
	m, _ := matrix.New(len(inputs), 1, inputs)
	dFn := StableSoftmax.dFn(m)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Apply(dFn, m)
	}
}
