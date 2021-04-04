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
	aFn := LogisticSigmoid.ActivationFn(m)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Apply(aFn, m)
	}
}

func BenchmarkLogisticSigmoidDeactivation(b *testing.B) {
	rnd := rand.New(rand.NewSource(0))
	inputs := []float64{rnd.Float64(), rnd.Float64()}
	m, _ := matrix.New(len(inputs), 1, inputs)
	dFn := LogisticSigmoid.DeactivationFn(m)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Apply(dFn, m)
	}
}

func BenchmarkTanHActivation(b *testing.B) {
	rnd := rand.New(rand.NewSource(0))
	inputs := []float64{rnd.Float64(), rnd.Float64()}
	m, _ := matrix.New(len(inputs), 1, inputs)
	aFn := TanH.ActivationFn(m)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Apply(aFn, m)
	}
}

func BenchmarkTanHDeactivation(b *testing.B) {
	rnd := rand.New(rand.NewSource(0))
	inputs := []float64{rnd.Float64(), rnd.Float64()}
	m, _ := matrix.New(len(inputs), 1, inputs)
	dFn := TanH.DeactivationFn(m)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Apply(dFn, m)
	}
}

func BenchmarkReLUActivation(b *testing.B) {
	rnd := rand.New(rand.NewSource(0))
	inputs := []float64{rnd.Float64(), rnd.Float64()}
	m, _ := matrix.New(len(inputs), 1, inputs)
	aFn := ReLU.ActivationFn(m)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Apply(aFn, m)
	}
}

func BenchmarkReLUDeactivation(b *testing.B) {
	rnd := rand.New(rand.NewSource(0))
	inputs := []float64{rnd.Float64(), rnd.Float64()}
	m, _ := matrix.New(len(inputs), 1, inputs)
	dFn := ReLU.DeactivationFn(m)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Apply(dFn, m)
	}
}

func BenchmarkLeakyReLUActivation(b *testing.B) {
	rnd := rand.New(rand.NewSource(0))
	inputs := []float64{rnd.Float64(), rnd.Float64()}
	m, _ := matrix.New(len(inputs), 1, inputs)
	aFn := LeakyReLU.ActivationFn(m)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Apply(aFn, m)
	}
}

func BenchmarkLeakyReLUDeactivation(b *testing.B) {
	rnd := rand.New(rand.NewSource(0))
	inputs := []float64{rnd.Float64(), rnd.Float64()}
	m, _ := matrix.New(len(inputs), 1, inputs)
	dFn := LeakyReLU.DeactivationFn(m)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Apply(dFn, m)
	}
}

func BenchmarkSoftmaxActivation(b *testing.B) {
	rnd := rand.New(rand.NewSource(0))
	inputs := []float64{rnd.Float64(), rnd.Float64()}
	m, _ := matrix.New(len(inputs), 1, inputs)
	aFn := Softmax.ActivationFn(m)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Apply(aFn, m)
	}
}

func BenchmarkSoftmaxDeactivation(b *testing.B) {
	rnd := rand.New(rand.NewSource(0))
	inputs := []float64{rnd.Float64(), rnd.Float64()}
	m, _ := matrix.New(len(inputs), 1, inputs)
	dFn := Softmax.DeactivationFn(m)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Apply(dFn, m)
	}
}

func BenchmarkStableSoftmaxActivation(b *testing.B) {
	rnd := rand.New(rand.NewSource(0))
	inputs := []float64{rnd.Float64(), rnd.Float64()}
	m, _ := matrix.New(len(inputs), 1, inputs)
	aFn := StableSoftmax.ActivationFn(m)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Apply(aFn, m)
	}
}

func BenchmarkStableSoftmaxDeactivation(b *testing.B) {
	rnd := rand.New(rand.NewSource(0))
	inputs := []float64{rnd.Float64(), rnd.Float64()}
	m, _ := matrix.New(len(inputs), 1, inputs)
	dFn := StableSoftmax.DeactivationFn(m)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Apply(dFn, m)
	}
}
