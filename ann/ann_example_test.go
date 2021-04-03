package ann_test

import (
	"fmt"
	"math/rand"

	network "github.com/azuwey/gonetwork/ann"
)

func ExampleNew() {
	// Create a 3 layer neural network (1 input, 1, hidden and 1 output layer), each layer with 1 nodes. The learning rate is 0.1.
	network.New(&network.Model{
		0.1, []network.LayerDescriptor{
			{1, "", nil, nil},
			{1, "LogisticSigmoid", nil, nil},
			{1, "LogisticSigmoid", nil, nil},
		},
	}, rand.New(rand.NewSource(0)))
}

func Example_predict_basic() {
	n, _ := network.New(&network.Model{
		0.1, []network.LayerDescriptor{
			{1, "", nil, nil},
			{1, "LogisticSigmoid", nil, nil},
			{1, "LogisticSigmoid", nil, nil},
		},
	}, rand.New(rand.NewSource(0)))

	predictions, _ := n.Predict([]float64{0.5})
	fmt.Println(predictions)
	// Output:
	// [0.4228985494909996]
}

func Example_predict_softmax() {
	n, _ := network.New(&network.Model{
		0.1, []network.LayerDescriptor{
			{1, "", nil, nil},
			{2, "LogisticSigmoid", nil, nil},
			{3, "Softmax", nil, nil},
		},
	}, rand.New(rand.NewSource(0)))

	predictions, _ := n.Predict([]float64{0.5})
	fmt.Println(predictions)

	sum := 0.0
	for _, p := range predictions {
		sum += p
	}

	fmt.Println(sum)
	// Output:
	// [0.3540425427265819 0.30588563177780553 0.34007182549561255]
	// 1
}

func Example_predict_stable_softmax() {
	n, _ := network.New(&network.Model{
		0.1, []network.LayerDescriptor{
			{3, "", nil, nil},
			{2, "LogisticSigmoid", nil, nil},
			{3, "StableSoftmax", nil, nil},
		},
	}, rand.New(rand.NewSource(0)))

	predictions, _ := n.Predict([]float64{1000, 2000, 3000})
	fmt.Println(predictions)

	sum := 0.0
	for _, p := range predictions {
		sum += p
	}

	fmt.Println(sum)
	// Output:
	// [0.15858466051288309 0.6492064646066636 0.19220887488045335]
	// 1
}

func Example_train_xor() {
	r := rand.New(rand.NewSource(0))
	n, _ := network.New(&network.Model{
		0.1, []network.LayerDescriptor{
			{2, "", nil, nil},
			{16, "TanH", nil, nil},
			{1, "LogisticSigmoid", nil, nil},
		},
	}, r)

	type dataSet struct{ inputs, targets []float64 }
	tesingData := []dataSet{
		{[]float64{0, 0}, []float64{0}},
		{[]float64{0, 1}, []float64{1}},
		{[]float64{1, 0}, []float64{1}},
		{[]float64{1, 1}, []float64{0}},
	}

	fmt.Println("Before training")
	for _, td := range tesingData {
		predictions, _ := n.Predict(td.inputs)
		fmt.Printf("Predictions: %v want: %v\n", predictions, td.targets)
	}

	learningData := []dataSet{
		{[]float64{0, 0}, []float64{0}},
		{[]float64{0, 1}, []float64{1}},
		{[]float64{1, 0}, []float64{1}},
		{[]float64{1, 1}, []float64{0}},
	}

	for e := 0; e < 3000; e++ {
		r.Shuffle(len(learningData), func(i, j int) {
			learningData[i], learningData[j] = learningData[j], learningData[i]
		})
		for _, ld := range learningData {
			n.Train(ld.inputs, ld.targets)
		}
	}

	fmt.Println("After training")
	for _, td := range tesingData {
		predictions, _ := n.Predict(td.inputs)
		fmt.Printf("Predictions: %v want: %v\n", predictions, td.targets)
	}
	// Output:
	// Before training
	// Predictions: [0.5] want: [0]
	// Predictions: [0.4202532649697775] want: [1]
	// Predictions: [0.273708193940601] want: [1]
	// Predictions: [0.24367486863911147] want: [0]
	// After training
	// Predictions: [0.005732007088584658] want: [0]
	// Predictions: [0.9881189978231344] want: [1]
	// Predictions: [0.9900294286745404] want: [1]
	// Predictions: [0.01471144916529516] want: [0]
}

func Example_train_4_bit_counter() {
	r := rand.New(rand.NewSource(0))
	n, _ := network.New(&network.Model{
		0.3, []network.LayerDescriptor{
			{4, "", nil, nil},
			{16, "TanH", nil, nil},
			{4, "LogisticSigmoid", nil, nil},
		},
	}, r)

	type dataSet struct{ inputs, targets []float64 }
	tesingData := []dataSet{
		{[]float64{1, 1, 0, 1}, []float64{1, 1, 1, 0}},
	}

	fmt.Println("Before training")
	for _, td := range tesingData {
		predictions, _ := n.Predict(td.inputs)
		fmt.Printf("Predictions: %v want: %v\n", predictions, td.targets)
	}

	learningData := []dataSet{
		{[]float64{0, 0, 0, 0}, []float64{0, 0, 0, 1}},
		{[]float64{0, 0, 0, 1}, []float64{0, 0, 1, 0}},
		{[]float64{0, 0, 1, 0}, []float64{0, 0, 1, 1}},
		{[]float64{0, 0, 1, 1}, []float64{0, 1, 0, 0}},
		{[]float64{0, 1, 0, 0}, []float64{0, 1, 0, 1}},
		{[]float64{0, 1, 0, 1}, []float64{0, 1, 1, 0}},
		{[]float64{0, 1, 1, 0}, []float64{0, 1, 1, 1}},
		{[]float64{0, 1, 1, 1}, []float64{1, 0, 0, 0}},
		{[]float64{1, 0, 0, 0}, []float64{1, 0, 0, 1}},
		{[]float64{1, 0, 0, 1}, []float64{1, 0, 1, 0}},
		{[]float64{1, 0, 1, 0}, []float64{1, 0, 1, 1}},
		{[]float64{1, 0, 1, 1}, []float64{1, 1, 0, 0}},
		{[]float64{1, 1, 0, 0}, []float64{1, 1, 0, 1}},
		{[]float64{1, 1, 1, 0}, []float64{1, 1, 1, 1}},
	}

	for e := 0; e < 3000; e++ {
		r.Shuffle(len(learningData), func(i, j int) {
			learningData[i], learningData[j] = learningData[j], learningData[i]
		})
		for _, ld := range learningData {
			n.Train(ld.inputs, ld.targets)
		}
	}

	fmt.Println("After training")
	for _, td := range tesingData {
		predictions, _ := n.Predict(td.inputs)
		fmt.Printf("Predictions: %v want: %v\n", predictions, td.targets)
	}
	// Output:
	// Before training
	// Predictions: [0.31960827874944575 0.15624517612075375 0.22486376541093606 0.33754839385451846] want: [1 1 1 0]
	// After training
	// Predictions: [0.9975696941940078 0.9801913790665919 0.9985607706133298 0.0024022594962865643] want: [1 1 1 0]
}
