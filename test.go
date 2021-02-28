package main

import (
	"fmt"
	"gonet"
	"math/rand"
)

func main() {
	// Set the random seed
	rand.Seed(42)
	// Build the model
	model := new(gonet.Model)
	model.Input = 2
	// Add the first layer
	layer1 := new(gonet.Layer)
	layer1.Neurons = 4
	layer1.Activation = gonet.Tanh
	model.Add(layer1)
	// Add the second layer
	layer2 := new(gonet.Layer)
	layer2.Neurons = 3
	layer2.Activation = gonet.Tanh
	model.Add(layer2)

	// Sample input
	in := [][]float64{
		{1, 2},
	}

	model.Forward(in)

	fmt.Println(model.Layers[0].Bias)
	fmt.Println(model.Layers[0].Output)

}
