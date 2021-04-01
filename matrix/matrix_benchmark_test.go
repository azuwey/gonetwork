package matrix

import (
	"math/rand"
	"testing"
)

func BenchmarkNew(b *testing.B) {
	r := rand.New(rand.NewSource(0))
	vals := []float64{r.Float64()}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		New(b.N, 1, vals)
	}
}

func BenchmarkNew_zeros(b *testing.B) {
	vals := make([]float64, 1)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		New(b.N, 1, vals)
	}
}

func BenchmarkCopy(b *testing.B) {
	r := rand.New(rand.NewSource(0))
	vals := []float64{r.Float64()}
	mat := &Matrix{vals, 1, 1}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Copy(mat)
	}
}

func BenchmarkAdd(b *testing.B) {
	r := rand.New(rand.NewSource(0))
	aVals := []float64{r.Float64()}
	bVals := []float64{r.Float64()}
	aMat := &Matrix{aVals, 1, 1}
	bMat := &Matrix{bVals, 1, 1}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		aMat.Add(aMat, bMat)
	}
}

func BenchmarkApply(b *testing.B) {
	r := rand.New(rand.NewSource(0))
	vals := []float64{r.Float64()}
	mat := &Matrix{vals, 1, 1}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mat.Apply(func(v float64, _, _ int, _ []float64) float64 {
			return v
		}, mat)
	}
}

func BenchmarkAt(b *testing.B) {
	r := rand.New(rand.NewSource(0))
	vals := []float64{r.Float64()}
	mat := &Matrix{vals, 1, 1}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mat.At(i, 1)
	}
}

func BenchmarkMultiply(b *testing.B) {
	r := rand.New(rand.NewSource(0))
	aVals := []float64{r.Float64()}
	bVals := []float64{r.Float64()}
	aMat := &Matrix{aVals, 1, 1}
	bMat := &Matrix{bVals, 1, 1}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		aMat.Multiply(aMat, bMat)
	}
}

func BenchmarkProduct(b *testing.B) {
	r := rand.New(rand.NewSource(0))
	aVals := []float64{r.Float64()}
	bVals := []float64{r.Float64()}
	aMat := &Matrix{aVals, 1, 1}
	bMat := &Matrix{bVals, 1, 1}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		aMat.Product(aMat, bMat)
	}
}

func BenchmarkScale(b *testing.B) {
	r := rand.New(rand.NewSource(0))
	vals := []float64{r.Float64()}
	mat := &Matrix{vals, 1, 1}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mat.Scale(2, mat)
	}
}

func BenchmarkSubtract(b *testing.B) {
	r := rand.New(rand.NewSource(0))
	aVals := []float64{r.Float64()}
	bVals := []float64{r.Float64()}
	aMat := &Matrix{aVals, 1, 1}
	bMat := &Matrix{bVals, 1, 1}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		aMat.Subtract(aMat, bMat)
	}
}

func BenchmarkTranspose(b *testing.B) {
	r := rand.New(rand.NewSource(0))
	vals := []float64{r.Float64()}
	mat := &Matrix{vals, 1, 1}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mat.Transpose(mat)
	}
}
