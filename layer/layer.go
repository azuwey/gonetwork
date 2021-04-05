package layer

import (
	"github.com/azuwey/gonetwork/matrix"
)

/* type Shape struct {
	Rows    int `json:"rows"`
	Columns int `json:"columns"`
	Depth   int `json:"depth"`
}

type Layer struct {
	Type         string `json:"type"`
	ActivationFn string `json:"activationFn"`
}

type Network struct {
	LearningRate string  `json:"learningRate"`
	Layers       []Layer `json:"layers"`
}*/

/**
 * Shape - Convolutional layer
 * output needs to be flattened
 *
 * RGBA image [[255, 255, 255, 1], [255, 255, 255, 1], [255, 255, 255, 1], [255, 255, 255, 1]] 2x2 px
 * input: [255, 255, 255, 1, 255, 255, 255, 1, 255, 255, 255, 1, 255, 255, 255, 1]
 * Rows: 2
 * Columns: 2
 * Depth: 4
**/

/**
 * Shape - Fully connected layer
 *
 * input: [0, 1, 1, 0]
 * Rows: 1
 * Columns: 4
 * Depth: 1
**/

/**
 * Shape - Pool (max or avg)
 *
 * input: [0, 1, 1, 0]
 * Rows: 2
 * Columns: 2
 * Depth: 1
**/

type Shape struct {
	Rows    int `json:"rows"`
	Columns int `json:"columns"`
	Depth   int `json:"depth"`
}

type LayerDescriptor struct {
	UUID              string `json:"uuid"`
	NextLayerUUID     string `json:"nextLayerUUID"`
	PreviousLayerUUID string `json:"previousLayerUUID"`
	InputShape        Shape  `json:"inputShape"`
	OutputShape       Shape  `json:"outputShape"`

	LearningRate *float64
}

type Layer interface {
	// Forwardprop performs forwardpropagation for the current layer
	Forwardprop(*matrix.Matrix) ([]float64, error)

	// Backprop performs backpropagation for the current layer
	Backprop(target *matrix.Matrix) error
}

type layer struct {
	UUID           string
	InputShape     Shape
	OutputShape    Shape
	Previous, Next Layer

	learningRate           *float64
	activated, deactivated *matrix.Matrix
}
