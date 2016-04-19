package main

import (
	"fmt"
	"github.com/markcheno/go-mep"
)

func main() {

	m := mep.New(100, 50)
	m.ReadTrainingData("testdata/simple.txt", " ")

	gens := 0
	for gens < 100 {
		m.StartSteadyState()
		m.PrintIndividual(0)
		gens++
		if m.Pop[0].Fitness < 0.01 {
			break
		}
	}
	fmt.Printf("solution found in %d runs\n", gens)
	m.PrintIndividual(0)
}
