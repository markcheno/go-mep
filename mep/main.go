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
	//m := mep.New(testData, mep.TotalErrorFF)

	m := mep.New(mep.NewPythagorean(50), mep.TotalErrorFF)
	m.SetOper("div", false)
	m.SetOper("sub", false)
	m.SetOper("sqrt", true)

	//m := mep.New(mep.NewPiTest(50, 2), mep.TotalErrorFF)
	//m.SetConst(5, -1, 1)

	//m := mep.New(mep.NewQuarticPoly(50), mep.TotalErrorFF)

	//m := mep.New(mep.NewRastigrinF1(50, 5), mep.TotalErrorFF)
	//m.SetProb(0.2, 0.8)
	//m.SetConst(5, -4, 4)
	//m.SetOper("sin", true)
	//m.SetPop(200, 50)

	//m := mep.New(mep.NewDejongF1(100), mep.TotalErrorFF)

	//m := mep.New(mep.NewSchwefel(100), mep.TotalErrorFF)
	//m.SetOper("sin", true)
	//m.SetOper("sqrt", true)
	//m.SetOper("abs", true)
	//m.SetConst(5, -1, 1)
	//m.SetPop(200, 50)

	//m := mep.New(mep.NewSequenceInduction(100), mep.TotalErrorFF)

	//m := mep.New(mep.NewDropwave(100), mep.TotalErrorFF)
	//m.SetConst(5, -1, 1)
	//m.SetPop(200, 50)

	//m := mep.New(mep.NewMichalewicz(100), mep.TotalErrorFF)
	//m.SetOper("sin", true)
	//m.SetConst(5, -1, 1)

	//m := mep.New(mep.NewSchafferF6(50), mep.TotalErrorFF)
	//m.SetOper("sin", true)
	//m.SetOper("sqrt", true)
	//m.SetConst(5, -1, 1)

	//m := mep.New(mep.NewSixHump(250), mep.TotalErrorFF)

	//m := mep.New(mep.NewSimpleConstantRegression1(100), mep.TotalErrorFF)
	//m.SetConst(10, -5, 5)

	//m := mep.New(mep.NewSimpleConstantRegression2(100), mep.TotalErrorFF)
	//m.SetConst(10, -5, 5)

	//m := mep.New(mep.NewSimpleConstantRegression3(100), mep.TotalErrorFF)
	//m.SetConst(10, -5, 5)

	gens, elapsed := m.Solve(2000, 0.1, true)
	fmt.Printf("Elapsed time: %s\n", elapsed)
	fmt.Printf("Solution after %d generations:\n", gens)
	m.PrintBest()
}
