package mep

import (
	"fmt"
	"math"
	"math/rand"
)

type testData struct {
	xmin float64
	xmax float64
	eval func(terms []float64) float64
}

func (t *testData) generate(numTraining, numVariables int) TrainingData {
	td := TrainingData{}
	td.Labels = make([]string, numVariables)
	for i := 0; i < numVariables; i++ {
		td.Labels[i] = fmt.Sprintf("x%d", i)
	}
	td.Train = make([][]float64, numTraining)
	td.Target = make([]float64, numTraining)
	for i := 0; i < numTraining; i++ {
		td.Train[i] = make([]float64, numVariables)
		for j := 0; j < numVariables; j++ {
			td.Train[i][j] = rand.Float64()*(t.xmax-t.xmin) + t.xmin
		}
		td.Target[i] = t.eval(td.Train[i])
	}
	return td
}

// NewAckley -
func NewAckley(numTraining, numVariables int) TrainingData {
	testdata := testData{
		xmin: -32,
		xmax: 32,
		eval: func(terms []float64) float64 {
			n := float64(len(terms))
			a := 20.0
			b := 0.2
			c := 2.0 * math.Pi
			s1 := 0.0
			s2 := 0.0
			for i := 0; i < len(terms); i++ {
				s1 = math.Pow(s1+terms[i], 2)
				s2 = s2 + math.Cos(c*terms[i])
			}
			return -a*math.Exp(-b*math.Sqrt(1.0/n*s1)) - math.Exp(1.0/n*s2) + a + math.Exp(1.0)
		},
	}
	return testdata.generate(numTraining, numVariables)
}

// NewRosenbrock -
func NewRosenbrock(numTraining, numVariables int) TrainingData {
	testdata := testData{
		xmin: -32,
		xmax: 32,
		eval: func(terms []float64) float64 {
			sum := 0.0
			for i := 0; i < len(terms)-1; i++ {
				sum = sum + 100*math.Pow((math.Pow(terms[i], 2)-terms[i+1]), 2) + math.Pow(terms[i]-1, 2)
			}
			return sum
		},
	}
	return testdata.generate(numTraining, numVariables)
}

// NewPiTest -
func NewPiTest(numTraining int) TrainingData {
	testdata := testData{
		xmin: 0,
		xmax: 0,
		eval: func(terms []float64) float64 {
			return math.Pi
		},
	}
	return testdata.generate(numTraining, 1)
}

// NewRastigrinF1 -
func NewRastigrinF1(numTraining, numVariables int) TrainingData {
	testdata := testData{
		xmin: -5.12,
		xmax: 5.12,
		eval: func(terms []float64) float64 {
			n := float64(len(terms))
			s := 0.0
			for i := 0; i < len(terms); i++ {
				s = s + (math.Pow(terms[i], 2) - 10*math.Cos(2*math.Pi*terms[i]))
			}
			return 10*n + s
		},
	}
	return testdata.generate(numTraining, numVariables)
}

// NewQuarticPoly -
func NewQuarticPoly(numTraining int) TrainingData {
	testdata := testData{
		xmin: -1,
		xmax: 1,
		eval: func(terms []float64) float64 {
			return math.Pow(terms[0], 4) + math.Pow(terms[0], 3) + math.Pow(terms[0], 2) + terms[0]
		},
	}
	return testdata.generate(numTraining, 1)
}

// NewPythagorean -
func NewPythagorean(numTraining int) TrainingData {
	testdata := testData{
		xmin: 5,
		xmax: 50,
		eval: func(terms []float64) float64 {
			return math.Sqrt((terms[0] * terms[0]) + (terms[1] * terms[1]))
		},
	}
	return testdata.generate(numTraining, 2)
}

// NewDejongF1 -
func NewDejongF1(numTraining int) TrainingData {
	testdata := testData{
		xmin: -5.12,
		xmax: 5.12,
		eval: func(terms []float64) float64 {
			return math.Pow(terms[0], 2) + math.Pow(terms[1], 2)
		},
	}
	return testdata.generate(numTraining, 2)
}

// NewSchwefel -
func NewSchwefel(numTraining int) TrainingData {
	testdata := testData{
		xmin: -1,
		xmax: 1,
		eval: func(terms []float64) float64 {
			return -terms[0]*math.Sin(math.Sqrt(math.Abs(terms[0]))) - terms[1]*math.Sin(math.Sqrt(math.Abs(terms[1])))
		},
	}
	return testdata.generate(numTraining, 2)
}

// NewSequenceInduction -
func NewSequenceInduction(numTraining int) TrainingData {
	testdata := testData{
		xmin: 5,
		xmax: 50,
		eval: func(terms []float64) float64 {
			return ((5.0 * math.Pow(terms[0], 4)) + (4.0 * math.Pow(terms[0], 3)) + (3.0 * math.Pow(terms[0], 2)) + (2.0 * terms[0]) + 1.0)
		},
	}
	return testdata.generate(numTraining, 1)
}

// NewDropwave -
func NewDropwave(numTraining int) TrainingData {
	testdata := testData{
		xmin: -5.12,
		xmax: 5.12,
		eval: func(terms []float64) float64 {
			return -(1.0 + math.Cos(12*math.Sqrt(terms[0]*terms[0]+terms[1]*terms[1]))) / (0.5*(terms[0]*terms[0]+terms[1]*terms[1]) + 2)
		},
	}
	return testdata.generate(numTraining, 2)
}

// NewMichalewicz -
func NewMichalewicz(numTraining int) TrainingData {
	testdata := testData{
		xmin: -5.12,
		xmax: 5.12,
		eval: func(terms []float64) float64 {
			return -math.Sin(terms[0])*math.Pow(math.Sin(terms[0]*terms[0]/math.Pi), 2) - math.Sin(terms[1])*math.Pow(math.Sin(terms[1]*terms[1]/math.Pi), 2)
		},
	}
	return testdata.generate(numTraining, 2)
}

// NewSchafferF6 -
func NewSchafferF6(numTraining int) TrainingData {
	testdata := testData{
		xmin: -5.12,
		xmax: 5.12,
		eval: func(terms []float64) float64 {
			return 0.5 + (math.Pow(math.Sin(math.Sqrt(terms[0]*terms[0]+terms[1]*terms[1])), 2)-0.5)/math.Pow(1+0.001*(terms[0]*terms[0]+terms[1]*terms[1]), 2)
		},
	}
	return testdata.generate(numTraining, 2)
}

// NewSixHump -
func NewSixHump(numTraining int) TrainingData {
	testdata := testData{
		xmin: -1,
		xmax: 1,
		eval: func(terms []float64) float64 {
			return (4.0-2.1*terms[0]*terms[0]+math.Pow(terms[0], 4)/3)*terms[0]*terms[0] + terms[0]*terms[1] + (-4+4*terms[1]*terms[1])*terms[1]*terms[1]
		},
	}
	return testdata.generate(numTraining, 2)
}

// NewSimpleConstantRegression1 -
func NewSimpleConstantRegression1(numTraining int) TrainingData {
	testdata := testData{
		xmin: 1,
		xmax: 20,
		eval: func(terms []float64) float64 {
			return (math.Pow(terms[0], 3) - 0.3*math.Pow(terms[0], 2) - 0.4*terms[0] - 0.6)
		},
	}
	return testdata.generate(numTraining, 1)
}

// NewSimpleConstantRegression2 -
func NewSimpleConstantRegression2(numTraining int) TrainingData {
	testdata := testData{
		xmin: 1,
		xmax: 20,
		eval: func(terms []float64) float64 {
			return terms[0]*terms[0] + math.Pi
		},
	}
	return testdata.generate(numTraining, 1)
}

// NewSimpleConstantRegression3 -
func NewSimpleConstantRegression3(numTraining int) TrainingData {
	testdata := testData{
		xmin: 1,
		xmax: 20,
		eval: func(terms []float64) float64 {
			return (math.E * terms[0] * terms[0]) + (math.Pi * terms[0])
		},
	}
	return testdata.generate(numTraining, 1)
}
