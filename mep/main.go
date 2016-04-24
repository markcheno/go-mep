package main

import (
	"fmt"
	"github.com/markcheno/go-mep"
	"math/rand"
	"time"
)

func main() {

	rand.Seed(time.Now().UTC().UnixNano())

	//testData := mep.NewAckley(50, 5)
	testData := mep.NewPi(50, 1)
	//testData := mep.ReadTrainingData("testdata/simple.txt", true, " ")
	m := mep.New(testData, mep.TotalErrorFF, 100, 50)
	m.SetOper("cos", true)
	m.SetConst(5, -1, 1)
	//m.SetOper("exp", true)
	start := time.Now()

	gens := 0
	for gens < 1000 {
		m.Search()
		m.PrintPop(0)
		gens++
		if m.BestFitness() < 0.01 {
			break
		}
	}
	elapsed := time.Since(start)
	fmt.Printf("Elapsed time: %s\n", elapsed)
	fmt.Printf("solution found in %d runs\n", gens)
	m.PrintPop(0)

}
