package gonet

import "math"

// Base struct for activation function
type Activation struct {
	Activ    func(*[][]float64, *[][]float64)
	ActivDer func(*[][]float64, *[][]float64)
}

// Initializes an activation function including its derivative
func (activ *Activation) Init(activation string) {
	switch activation {
	case "tanh":
		activ.Activ = tanh
		activ.ActivDer = dTanh
	case "relu":
		activ.Activ = relu
		activ.ActivDer = dRelu
	case "sigmoid":
		activ.Activ = sigmoid
		activ.ActivDer = dSigmoid
	case "step":
		activ.Activ = step
	case "linear":
		activ.Activ = identity
		// Room for more
	}
}

// Hyperbolic tangent activation function
func tanh(mat *[][]float64, out *[][]float64) {
	n := len(*mat)
	m := len((*mat)[0])

	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			(*out)[i][j] = math.Tanh((*mat)[i][j])
		}
	}
}

// Derivative of hyperbolic tangent activation function
func dTanh(mat *[][]float64, out *[][]float64) {
	n := len(*mat)
	m := len((*mat)[0])

	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			(*out)[i][j] = 1 - math.Pow(math.Tanh((*mat)[i][j]), 2)
		}
	}
}

// Rectified linear unit activation function
func relu(mat *[][]float64, out *[][]float64) {
	n := len(*mat)
	m := len((*mat)[0])

	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			(*out)[i][j] = math.Max(0, (*mat)[i][j])
		}
	}
}

// Derivative of rectified linear unit activation function
func dRelu(mat *[][]float64, out *[][]float64) {
	n := len(*mat)
	m := len((*mat)[0])

	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			if (*mat)[i][j] > 0 {
				(*out)[i][j] = 1
			} else {
				(*out)[i][j] = 0
			}
		}
	}
}

// Sigmoid activation function
func sigmoid(mat *[][]float64, out *[][]float64) {
	n := len(*mat)
	m := len((*mat)[0])

	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			(*out)[i][j] = 1 / (1 + math.Exp(-(*mat)[i][j]))
		}
	}
}

// Derivative of sigmoid activation function
func dSigmoid(mat *[][]float64, out *[][]float64) {
	n := len(*mat)
	m := len((*mat)[0])

	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			sig := 1 / (1 + math.Exp(-(*mat)[i][j]))
			(*out)[i][j] = sig * (1 - sig)
		}
	}
}

// Linear activation function
func identity(mat *[][]float64, out *[][]float64) {
	n := len(*mat)
	m := len((*mat)[0])

	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			(*out)[i][j] = (*mat)[i][j]
		}
	}
}

// Step activation function
func step(mat *[][]float64, out *[][]float64) {
	n := len(*mat)
	m := len((*mat)[0])

	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			if (*mat)[i][j] < 0 {
				(*out)[i][j] = -1
			} else if (*mat)[i][j] >= 0 {
				(*out)[i][j] = 1
			}
		}
	}
}
