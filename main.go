package main

import (
	"fmt"
	"math/rand"

	"./gonet"
)

func main() {
	// Set the random seed
	rand.Seed(42)
	// Build the model
	model := new(gonet.Model)
	model.Input = 2
	// Add the first layer
	layer1 := new(gonet.Layer)
	layer1.Neurons = 3
	layer1.Activation = gonet.Tanh
	layer1.DActivation = gonet.DTanh
	model.Add(layer1)
	// Add the first layer
	layer2 := new(gonet.Layer)
	layer2.Neurons = 4
	layer2.Activation = gonet.Tanh
	layer2.DActivation = gonet.DTanh
	model.Add(layer2)
	// Add final layer
	layer3 := new(gonet.Layer)
	layer3.Neurons = 1
	layer3.Activation = gonet.Step
	model.Add(layer3)

	// Add an optimizer
	sgd := new(gonet.Optimizer)
	sgd.Eta = 0.05
	sgd.Loss = gonet.MSE
	model.Optimizer = sgd

	// Sample input
	in := [][]float64{
		{1.2637462, 0.52276885},
		{-1.58363913, -1.87520749},
		{-2.64189511, 0.9501662},
		{-0.69569954, -0.95261718},
		{0.0363519, -2.76182951},
		{-2.69148696, 1.53246326},
		{-0.78016167, 0.82118981},
		{-1.44299129, -1.16452886},
		{-0.29751061, 1.39199187},
		{0.43316508, 0.5014854},
	}

	// Sample target
	t := []float64{1, -1, 1, -1, -1, 1, 1, -1, -1, 1}

	for epoch := 0; epoch < 100; epoch++ {
		fmt.Println("\nEpoch", epoch)
		loss := 0.0
		for i := 0; i < len(in); i++ {

			sample := [][]float64{in[i]}
			sample = gonet.Transpose(&sample)
			sample_target := [][]float64{{t[i]}}

			model.Forward(sample)

			fmt.Println("Target", t[i])
			fmt.Println("Model Output:", model.Layers[len(model.Layers)-1].Output)
			loss += gonet.MSE(&model.Layers[len(model.Layers)-1].Output, &sample_target)[0][0]
			model.Backward(sample, sample_target)
		}
		fmt.Printf("Loss Epoch %d: %f", epoch, loss)

	}
	model.Forward(in)
}
