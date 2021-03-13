package matrix

import (
	"math/rand"
	"testing"
)

func BenchmarkNew(b *testing.B) {
	mVals := make([]float64, b.N)

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		New(b.N, 1, mVals)
	}
}

func BenchmarkAdd(b *testing.B) {
	mVals := make([]float64, b.N)
	aVals := make([]float64, b.N)

	for i := 0; i < b.N; i++ {
		mVals[i] = rand.Float64()
		aVals[i] = rand.Float64()
	}

	m := &Matrix{mVals, b.N, 1}
	a := &Matrix{aVals, b.N, 1}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		m.Add(m, a)
	}
}

func BenchmarkApply(b *testing.B) {
	mVals := make([]float64, b.N)

	for i := 0; i < b.N; i++ {
		mVals[i] = rand.Float64()
	}

	m := &Matrix{mVals, b.N, 1}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		m.Apply(func(v float64, r, c int) float64 {
			return v
		}, m)
	}
}

func BenchmarkAt(b *testing.B) {
	mVals := make([]float64, b.N)

	for i := 0; i < b.N; i++ {
		mVals[i] = rand.Float64()
	}

	m := &Matrix{mVals, b.N, 1}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		m.At(i, 1)
	}
}

func BenchmarkDimensions(b *testing.B) {
	mVals := make([]float64, b.N)

	for i := 0; i < b.N; i++ {
		mVals[i] = rand.Float64()
	}

	m := &Matrix{mVals, b.N, 1}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		m.Dimensions()
	}
}

func BenchmarkMatrixProduct(b *testing.B) {
	mVals := make([]float64, b.N)
	aVals := make([]float64, b.N)

	for i := 0; i < b.N; i++ {
		mVals[i] = rand.Float64()
		aVals[i] = rand.Float64()
	}

	m := &Matrix{mVals, 1, b.N}
	a := &Matrix{aVals, b.N, 1}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		m.MatrixProduct(m, a)
	}
}

func BenchmarkMultiply(b *testing.B) {
	mVals := make([]float64, b.N)
	aVals := make([]float64, b.N)

	for i := 0; i < b.N; i++ {
		mVals[i] = rand.Float64()
		aVals[i] = rand.Float64()
	}

	m := &Matrix{mVals, b.N, 1}
	a := &Matrix{aVals, b.N, 1}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		m.Multiply(m, a)
	}
}

func BenchmarkRaw(b *testing.B) {
	mVals := make([]float64, b.N)

	for i := 0; i < b.N; i++ {
		mVals[i] = rand.Float64()
	}

	m := &Matrix{mVals, b.N, 1}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		m.Raw()
	}
}

func BenchmarkScale(b *testing.B) {
	mVals := make([]float64, b.N)

	for i := 0; i < b.N; i++ {
		mVals[i] = rand.Float64()
	}

	m := &Matrix{mVals, b.N, 1}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		m.Scale(2, m)
	}
}

func BenchmarkTranspose(b *testing.B) {
	mVals := make([]float64, b.N*2)

	for i := 0; i < b.N; i++ {
		mVals[i] = rand.Float64()
	}

	m := &Matrix{mVals, b.N, 2}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		m.Transpose(m)
	}
}

func BenchmarkValues(b *testing.B) {
	mVals := make([]float64, b.N)

	for i := 0; i < b.N; i++ {
		mVals[i] = rand.Float64()
	}

	m := &Matrix{mVals, b.N, 1}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		m.Values()
	}
}
