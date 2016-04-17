package main

import (
	"github.com/markcheno/go-mep"
)

func main() {

	mep := mep.New(100, 50)
	mep.ReadTrainingData("testdata/building1.txt", " ")
	gens := 0
	for gens < 50 {
		mep.StartSteadyState()
		mep.PrintIndividual(0)
		gens++
	}
}
