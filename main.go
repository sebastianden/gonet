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
	model.Init(2)
	// Add the first layer
	layer1 := new(gonet.Layer)
	layer1.Init(50, "tanh")
	model.Add(layer1)
	// Add the first layer
	layer2 := new(gonet.Layer)
	layer2.Init(5, "tanh")
	model.Add(layer2)
	// Add final layer
	layer3 := new(gonet.Layer)
	layer3.Init(1, "step")
	model.Add(layer3)

	// Add an optimizer
	model.Compile("mse", "sgd", 0.01)

	// Import data
	X, y := gonet.LoadData("data/data100.csv")

	yPred := model.Predict(X)
	fmt.Println(yPred)
	fmt.Println(y)

	model.Fit(X, y, 100)

	yPred = model.Predict(X)
	fmt.Println(yPred)
	fmt.Println(y)

}
