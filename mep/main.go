package main

import (
	"fmt"
	"github.com/markcheno/go-mep"
	"math/rand"
	"time"
)

func main() {

	rand.Seed(time.Now().UTC().UnixNano())

	//testData := mep.ReadTrainingData("testdata/simple2.txt", true, " ")
	testData := mep.NewPi(50, 1)
	m := mep.New(testData, mep.TotalErrorFF)
	//m.SetProb(0.2, 0.8)
	//m.SetOper("div", false)
	m.SetOper("cos", true)
	m.SetConst(10, -1, 1)
	//m.SetPop(100, 20)

	gens, elapsed := m.Solve(1000, 0.001, false)

	fmt.Printf("Elapsed time: %s\n", elapsed)
	fmt.Printf("Solution found in %d generations:\n", gens)
	m.PrintBest()
}
