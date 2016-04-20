package main

import (
	"fmt"
	"github.com/markcheno/go-mep"
	"math"
	"math/rand"
	"time"
)

func genQuarticPoly(npoints int) (mep.Labels, mep.TrainingData, mep.Target) {

	labels := mep.Labels{"x"}
	training := make(mep.TrainingData, npoints)
	target := make(mep.Target, npoints)

	for i := 0; i < npoints; i++ {
		x := rand.Float64()
		if rand.Float64() < 0.5 {
			x = x * -1
		}
		result := math.Pow(x, 4) + math.Pow(x, 3) + math.Pow(x, 2) + x
		training[i] = []float64{x}
		target[i] = result
	}

	return labels, training, target
}

func main() {

	rand.Seed(time.Now().UTC().UnixNano())

	m := mep.New(100, 50)
	labels, training, target := genQuarticPoly(30)
	m.Setup(labels, training, target)
	//m.Read("testdata/simple.txt", true, " ")
	start := time.Now()

	gens := 0
	for gens < 1000 {
		m.Run()
		m.Print(0)
		gens++
		if m.Pop[0].Fitness < 0.1 {
			break
		}
	}
	elapsed := time.Since(start)
	fmt.Printf("Elapsed time: %s\n", elapsed)
	fmt.Printf("solution found in %d runs\n", gens)
	m.Print(0)
}
