package activationfn

import (
	"math/rand"
	"testing"

	"github.com/azuwey/gonetwork/matrix"
)

func BenchmarkLogisticSigmoidActivation(b *testing.B) {
	rnd := rand.New(rand.NewSource(0))
	inputs := []float64{rnd.Float64(), rnd.Float64()}
	m, _ := matrix.New(len(inputs), 1, inputs)
	aFn := logisticSigmoid.ActivationFn(m)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Apply(aFn, m)
	}
}

func BenchmarkLogisticSigmoidDeactivation(b *testing.B) {
	rnd := rand.New(rand.NewSource(0))
	inputs := []float64{rnd.Float64(), rnd.Float64()}
	m, _ := matrix.New(len(inputs), 1, inputs)
	dFn := logisticSigmoid.DeactivationFn(m)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Apply(dFn, m)
	}
}

func BenchmarkTanHActivation(b *testing.B) {
	rnd := rand.New(rand.NewSource(0))
	inputs := []float64{rnd.Float64(), rnd.Float64()}
	m, _ := matrix.New(len(inputs), 1, inputs)
	aFn := tanH.ActivationFn(m)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Apply(aFn, m)
	}
}

func BenchmarkTanHDeactivation(b *testing.B) {
	rnd := rand.New(rand.NewSource(0))
	inputs := []float64{rnd.Float64(), rnd.Float64()}
	m, _ := matrix.New(len(inputs), 1, inputs)
	dFn := tanH.DeactivationFn(m)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Apply(dFn, m)
	}
}

func BenchmarkReLUActivation(b *testing.B) {
	rnd := rand.New(rand.NewSource(0))
	inputs := []float64{rnd.Float64(), rnd.Float64()}
	m, _ := matrix.New(len(inputs), 1, inputs)
	aFn := reLU.ActivationFn(m)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Apply(aFn, m)
	}
}

func BenchmarkReLUDeactivation(b *testing.B) {
	rnd := rand.New(rand.NewSource(0))
	inputs := []float64{rnd.Float64(), rnd.Float64()}
	m, _ := matrix.New(len(inputs), 1, inputs)
	dFn := reLU.DeactivationFn(m)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Apply(dFn, m)
	}
}

func BenchmarkLeakyReLUActivation(b *testing.B) {
	rnd := rand.New(rand.NewSource(0))
	inputs := []float64{rnd.Float64(), rnd.Float64()}
	m, _ := matrix.New(len(inputs), 1, inputs)
	aFn := leakyReLU.ActivationFn(m)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Apply(aFn, m)
	}
}

func BenchmarkLeakyReLUDeactivation(b *testing.B) {
	rnd := rand.New(rand.NewSource(0))
	inputs := []float64{rnd.Float64(), rnd.Float64()}
	m, _ := matrix.New(len(inputs), 1, inputs)
	dFn := leakyReLU.DeactivationFn(m)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Apply(dFn, m)
	}
}

func BenchmarkSoftmaxActivation(b *testing.B) {
	rnd := rand.New(rand.NewSource(0))
	inputs := []float64{rnd.Float64(), rnd.Float64()}
	m, _ := matrix.New(len(inputs), 1, inputs)
	aFn := softmax.ActivationFn(m)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Apply(aFn, m)
	}
}

func BenchmarkSoftmaxDeactivation(b *testing.B) {
	rnd := rand.New(rand.NewSource(0))
	inputs := []float64{rnd.Float64(), rnd.Float64()}
	m, _ := matrix.New(len(inputs), 1, inputs)
	dFn := softmax.DeactivationFn(m)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Apply(dFn, m)
	}
}

func BenchmarkStableSoftmaxActivation(b *testing.B) {
	rnd := rand.New(rand.NewSource(0))
	inputs := []float64{rnd.Float64(), rnd.Float64()}
	m, _ := matrix.New(len(inputs), 1, inputs)
	aFn := stableSoftmax.ActivationFn(m)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Apply(aFn, m)
	}
}

func BenchmarkStableSoftmaxDeactivation(b *testing.B) {
	rnd := rand.New(rand.NewSource(0))
	inputs := []float64{rnd.Float64(), rnd.Float64()}
	m, _ := matrix.New(len(inputs), 1, inputs)
	dFn := stableSoftmax.DeactivationFn(m)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Apply(dFn, m)
	}
}
