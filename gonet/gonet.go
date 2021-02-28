package gonet

import (
	"fmt"
	"math"
	"math/rand"
)

// Base struct for ANN layer
type Layer struct {
	Neurons    uint
	Weigths    [][]float64
	Bias       float64
	Activation func(*[][]float64)
	Output     [][]float64
}

// Base struct for ANN model
type Model struct {
	Input  uint
	Layers []*Layer
}

// Adds a layer to a model
func (model *Model) Add(layer *Layer) {
	// Check if the new layer would be the first
	l := len(model.Layers)
	if l == 0 {
		// Initialize the weigths randomly
		layer.Weigths = initRand(model.Input, layer.Neurons)
	} else {
		// Get the outputsize of the previous layer
		layer.Weigths = initRand(model.Layers[len(model.Layers)-1].Neurons, layer.Neurons)
	}
	// Initialize bias
	layer.Bias = 0.5 - rand.Float64()
	// Append the new layer to the slice of layers in model
	model.Layers = append(model.Layers, layer)
}

// Performs a forward step with one input sample
func (model *Model) Forward(input [][]float64) {
	for i, l := range model.Layers {
		if i == 0 {
			l.Output, _ = matmult(&input, &l.Weigths)
		} else {
			l.Output, _ = matmult(&model.Layers[i-1].Output, &l.Weigths)
		}
		addBias(&l.Output, l.Bias)
		l.Activation(&l.Output)
	}
}

// Add a bias to the output of a layer
func addBias(mat *[][]float64, b float64) {
	n := len(*mat)
	m := len((*mat)[0])

	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			(*mat)[i][j] += b
		}
	}
}

// Hyperbolic tangent activation function
func Tanh(mat *[][]float64) {
	n := len(*mat)
	m := len((*mat)[0])

	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			(*mat)[i][j] = math.Tanh((*mat)[i][j])
		}
	}
}

// Fucntion for performing matrix multiplication
func matmult(A *[][]float64, B *[][]float64) ([][]float64, error) {
	a := len((*A)[0])
	b := len(*B)
	if a != b {
		err := fmt.Errorf("Cannot multiply matrices. Wrong dimensions %d and %d", a, b)
		return [][]float64{{}}, err
	}
	n := len(*A)
	m := len((*B)[0])

	mul := make([][]float64, n)
	for i := range mul {
		mul[i] = make([]float64, m)
	}

	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			for k := 0; k < a; k++ {
				mul[i][j] += (*A)[i][k] * (*B)[k][j]
			}
		}
	}
	return mul, nil
}

func initRand(n uint, m uint) [][]float64 {
	mat := make([][]float64, n)
	for i := range mat {
		row := make([]float64, m)
		for j := range row {
			row[j] = 0.5 - rand.Float64()
		}
		mat[i] = row
	}
	return mat
}
