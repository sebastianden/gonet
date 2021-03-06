package main

import (
	"math/rand"

	"./gonet"
)

func main() {
	// Set the random seed
	rand.Seed(42)
	// Build the model
	model := new(gonet.Model)
	model.Init(2)
	// Add the first layer
	layer1 := new(gonet.Layer)
	layer1.Init(3, "tanh")
	model.Add(layer1)
	// Add the first layer
	layer2 := new(gonet.Layer)
	layer2.Init(4, "tanh")
	model.Add(layer2)
	// Add final layer
	layer3 := new(gonet.Layer)
	layer3.Init(1, "step")
	model.Add(layer3)

	// Add an optimizer
	model.Compile("mse", "sgd", 0.05)

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

	model.Fit(in, t, 100)

}
