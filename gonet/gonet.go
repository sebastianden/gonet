package gonet

import (
	"fmt"
	"math"
	"math/rand"
)

// TODO: We need a Tensor type for [][]float64

// Base struct for ANN layer
type Layer struct {
	Neurons    uint
	Weigths    [][]float64
	Activation *Activation
	Delta      [][]float64
	Net        [][]float64
	Output     [][]float64
}

// Base struct for ANN model
type Model struct {
	Input     uint
	Layers    []*Layer
	Loss      *Loss
	Optimizer *Optimizer
}

// Base struct for Optimizer
type Optimizer struct {
	// TODO: Eta could be a function made in a closure returning a variable learning rate
	Eta float64
}

// Base struct for loss function
type Loss struct {
	Loss    func(*[][]float64, *[][]float64) [][]float64
	LossDer func(*[][]float64, *[][]float64, *[][]float64)
}

// Initializes a model
func (model *Model) Init(input uint) {
	model.Input = input
}

// Initializes a layer
func (layer *Layer) Init(neurons uint, activation string) {
	// Initialize the net vector (output before activtion function)
	layer.Net = initZero(neurons, 1)
	// Initialize the output vector
	layer.Output = initZero(neurons, 1)
	// Initialize the delta vector
	layer.Delta = initZero(neurons, 1)
	// Initialize number of neurons
	layer.Neurons = neurons
	// Initialize activation function
	layer.Activation = new(Activation)
	layer.Activation.Init(activation)
}

// Initializes an optimizer
func (optimizer *Optimizer) Init(opt string, eta float64) {
	switch opt {
	case "sgd":
		optimizer.Eta = eta
		// Room for more
	}
}

// Initializes an activation function including its derivative
func (loss *Loss) Init(loss_func string) {
	switch loss_func {
	case "mse":
		loss.Loss = mse
		loss.LossDer = dMse
		// Room for more
	}
}

// Adds a layer to a model
func (model *Model) Add(layer *Layer) {
	// Check if the new layer would be the first
	l := len(model.Layers)
	if l == 0 {
		// Initialize the weigths randomly
		// TODO: Better initialization (e.g. Glorot)
		layer.Weigths = initRand(layer.Neurons, model.Input+1)
	} else {
		// Get the output size of the previous layer
		layer.Weigths = initRand(layer.Neurons, model.Layers[len(model.Layers)-1].Neurons+1)
	}
	// Append the new layer to the slice of layers in model
	model.Layers = append(model.Layers, layer)
}

// Adds an optimizer to the model
func (model *Model) Compile(loss string, optimizer string, eta float64) {
	model.Loss = new(Loss)
	model.Loss.Init(loss)
	model.Optimizer = new(Optimizer)
	model.Optimizer.Init(optimizer, eta)
}

// Performs a forward step with one input sample
func (model *Model) forward(input [][]float64) error {
	if len(input) != int(model.Input) {
		err1 := fmt.Errorf("Input of wrong size %d. Must be %d.", len(input), int(model.Input))
		return err1
	}
	for i, l := range model.Layers {
		// Add the bias to the input of the current layer
		var layerInput [][]float64
		if i == 0 {
			layerInput = input
		} else {
			layerInput = model.Layers[i-1].Output
		}
		inputBias := initZero(uint(len(layerInput)), uint(len(layerInput[0])))
		copy(inputBias, layerInput)
		inputBias = append(inputBias, []float64{1.0})
		// net_j = w_ji * x_i
		var err2 error
		l.Net, err2 = matMult(&l.Weigths, &inputBias)
		if err2 != nil {
			return err2
		}
		// out_j = f(net_j)
		l.Activation.Activ(&l.Net, &l.Output)

	}
	return nil
}

// Performs a backpropagation step
// TODO needs error handling
func (model *Model) backward(input [][]float64, target [][]float64) error {
	// Traverse the layers backwards
	for i := len(model.Layers) - 1; i >= 0; i-- {
		// Calculate the error signal delta
		// If it's the last layer use the loss function to calculate error
		// else use error of anterior layer
		if i == len(model.Layers)-1 {
			// delta_out =
			model.Loss.LossDer(&model.Layers[i].Output, &target, &model.Layers[i].Delta)
		} else {
			activationDer := initZero(uint(len(model.Layers[i].Net)), uint(len(model.Layers[i].Net[0])))
			// Transpose the weight matrix and remove the bias weights
			wT := transpose(&model.Layers[i+1].Weigths)
			wTnoBias := wT[:len(wT)-1]
			// Calculate error signals of current layer
			var err error
			model.Layers[i].Delta, err = matMult(&wTnoBias, &model.Layers[i+1].Delta)
			if err != nil {
				return err
			}
			model.Layers[i].Activation.ActivDer(&model.Layers[i].Net, &activationDer)

			var err2 error
			model.Layers[i].Delta, err2 = elementMult(&model.Layers[i].Delta, &activationDer)
			if err2 != nil {
				fmt.Println(err2)
				return err
			}

		}
		// if its the first layer use the input as output of previous
		var layerInput [][]float64
		if i == 0 {
			layerInput = input
		} else {
			layerInput = model.Layers[i-1].Output
		}
		// Add bias to output of previous layer, Deep Copy needed there must be a better Way to do this
		inputBias := transpose(&layerInput)
		inputBias[0] = append(inputBias[0], 1.0)

		deltaW, err := matMult(&model.Layers[i].Delta, &inputBias)
		if err != nil {
			return err
		}
		// Perform weight update
		weightUpdate(&model.Layers[i].Weigths, &deltaW, &model.Optimizer.Eta)
	}
	return nil
}

// Function for training a model
// TODO Missing the whole batch part
func (model *Model) Fit(in [][]float64, t []float64, epochs uint) {
	for epoch := 0; epoch < int(epochs); epoch++ {
		fmt.Println("\nEpoch", epoch)
		loss := 0.0
		for i := 0; i < len(in); i++ {

			sample := [][]float64{in[i]}
			sample = transpose(&sample)
			sample_target := [][]float64{{t[i]}}

			model.forward(sample)

			loss += model.Loss.Loss(&model.Layers[len(model.Layers)-1].Output, &sample_target)[0][0]
			model.backward(sample, sample_target)
		}
		fmt.Printf("Loss Epoch %d: %f", epoch, loss/float64(len(in)))
	}
}

// Initialize a matrix with values from -0.5 to 0.5
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

// Initialize a matrix with zeros
func initZero(n uint, m uint) [][]float64 {
	mat := make([][]float64, n)
	for i := range mat {
		mat[i] = make([]float64, m)
	}
	return mat
}

// Function for performing weight update
func weightUpdate(w *[][]float64, deltaW *[][]float64, eta *float64) {
	n := len(*w)
	m := len((*w)[0])

	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			(*w)[i][j] += *eta * (*deltaW)[i][j]
		}
	}
}

// Function for performing matrix multiplication
func matMult(A *[][]float64, B *[][]float64) ([][]float64, error) {
	a := len((*A)[0])
	b := len(*B)
	if a != b {
		err := fmt.Errorf("Cannot multiply matrices. Wrong dimensions %d and %d", a, b)
		return [][]float64{{0}}, err
	}
	n := len(*A)
	m := len((*B)[0])

	mul := initZero(uint(n), uint(m))

	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			for k := 0; k < a; k++ {
				mul[i][j] += (*A)[i][k] * (*B)[k][j]
			}
		}
	}
	return mul, nil
}

// Transpose a Tensor
func transpose(A *[][]float64) [][]float64 {
	n := len(*A)
	m := len((*A)[0])

	T := initZero(uint(m), uint(n))

	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			T[j][i] = (*A)[i][j]
		}
	}
	return T
}

// Fucntion for performing elementwise multiplication / Hadamard product
func elementMult(A *[][]float64, B *[][]float64) ([][]float64, error) {
	a := len(*A)
	b := len(*B)
	if a != b {
		err := fmt.Errorf("Cannot perform elementwise multiplication. Wrong dimensions %d and %d", a, b)
		return [][]float64{{0}}, err
	}
	n := len(*A)
	m := len((*A)[0])

	mul := initZero(uint(n), uint(m))

	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			mul[i][j] = (*A)[i][j] * (*B)[i][j]
		}
	}
	return mul, nil
}

// Mean Squared Error Loss function
func mse(output *[][]float64, target *[][]float64) [][]float64 {
	n := len(*output)
	m := len((*output)[0])

	delta := initZero(uint(n), uint(m))

	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			delta[i][j] = math.Pow((*target)[i][j]-(*output)[i][j], 2)
		}
	}
	return delta
}

// Derivative of Mean Squared Error Loss function
func dMse(output *[][]float64, target *[][]float64, delta *[][]float64) {
	n := len(*output)
	m := len((*output)[0])

	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			(*delta)[i][j] = 2 * ((*target)[i][j] - (*output)[i][j])
		}
	}
}
