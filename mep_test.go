package mep

import (
	"fmt"
	"path/filepath"
	"reflect"
	"runtime"
	"testing"
)

func ok(t *testing.T, err error) {
	if err != nil {
		_, file, line, _ := runtime.Caller(1)
		fmt.Printf("%s:%d: unexpected error: %s\n", filepath.Base(file), line, err.Error())
		t.FailNow()
	}
}

func equals(t *testing.T, exp, act interface{}) {
	if !reflect.DeepEqual(exp, act) {
		_, file, line, _ := runtime.Caller(1)
		fmt.Printf("%s:%d:\n\texp: %#v\n\tact: %#v\n", filepath.Base(file), line, exp, act)
		t.FailNow()
	}
}

func TestNew(t *testing.T) {
	codeSize := 50
	popSize := 100
	mep := New(popSize, codeSize)
	equals(t, mep.PopSize, popSize)
	equals(t, mep.CodeLength, codeSize)
	equals(t, mep.PopSize, popSize)
	equals(t, mep.MutationProbability, 0.1)
	equals(t, mep.CrossoverProbability, 0.9)
	equals(t, mep.VariablesProbability, 0.4)
	equals(t, mep.OperatorsProbability, 0.5)
	equals(t, mep.ConstantsProbability, 1-mep.VariablesProbability-mep.OperatorsProbability)
	equals(t, mep.ConstantsNum, 3)
	equals(t, mep.ConstantsMin, -1.0)
	equals(t, mep.ConstantsMax, 1.0)
	equals(t, len(mep.Pop), mep.PopSize)
}

func TestReadTrainingData(t *testing.T) {
	codeSize := 50
	popSize := 100
	mep := New(popSize, codeSize)
	mep.ReadTrainingData("mep/testdata/building1.txt", " ")
	equals(t, mep.numVariables, 14)
	equals(t, mep.numTrainingData, 2104)
	equals(t, len(mep.trainingData), mep.numTrainingData)
	equals(t, len(mep.trainingData[0]), mep.numVariables)
	equals(t, len(mep.target), mep.numTrainingData)
	equals(t, mep.target[0], 0.343433)
	equals(t, mep.target[len(mep.target)-1], 0.626552)
	equals(t, mep.trainingData[0][0], 1.0)
	equals(t, mep.trainingData[len(mep.trainingData)-1][0], 0.0)
	equals(t, len(mep.evalMatrix), mep.CodeLength)
	equals(t, len(mep.evalMatrix[0]), mep.numTrainingData)
}

func TestRandomChromosome(t *testing.T) {

	codeSize := 50
	popSize := 100
	mep := New(popSize, codeSize)
	mep.ReadTrainingData("mep/testdata/building1.txt", " ")
	c := mep.randomChromosome()

	// make sure constants are between mep.ConstantsMin, mep.ConstantsMax, len==mep.ConstantsNum
	equals(t, len(c.Constants), mep.ConstantsNum)
	for i := 1; i < mep.ConstantsNum; i++ {
		equals(t, c.Constants[i] <= mep.ConstantsMax, true)
		equals(t, c.Constants[i] >= mep.ConstantsMin, true)
	}

	program := c.Program
	nonzeroOp := false
	nonzeroAdr1 := false
	nonzeroAdr2 := false
	for i := 1; i < mep.CodeLength; i++ {
		// variables are indexed from 0: 0,1,2,...
		// constants are indexed from num_variables
		// operators are -1, -2, -3...
		if i == 1 {
			// make sure first op is variable or constant
			equals(t, program[i].op >= 0, true)
		} else {
			// make sure other ops are operator or variable or constant
			equals(t, program[i].op >= -len(operators), true)
			equals(t, program[i].op < (mep.numVariables+mep.ConstantsNum), true)
		}
		equals(t, program[i].adr1 >= 0, true)
		equals(t, program[i].adr2 >= 0, true)
		equals(t, program[i].adr1 < mep.CodeLength, true)
		equals(t, program[i].adr2 < mep.CodeLength, true)

		// make sure we have non zero data
		if program[i].op != 0 {
			nonzeroOp = true
		}
		if program[i].adr1 != 0 {
			nonzeroAdr1 = true
		}
		if program[i].adr2 != 0 {
			nonzeroAdr2 = true
		}
	}
	equals(t, nonzeroOp, true)
	equals(t, nonzeroAdr1, true)
	equals(t, nonzeroAdr2, true)
}

func TestComputeEvalMatrix(t *testing.T) {
	codeSize := 50
	popSize := 100
	mep := New(popSize, codeSize)
	mep.ReadTrainingData("mep/testdata/building1.txt", " ")
	mep.StartSteadyState()

	nonzero1 := false
	for i := 0; i < mep.CodeLength; i++ {
		if mep.evalMatrix[i][0] != 0.0 {
			nonzero1 = true
		}
	}
	equals(t, nonzero1, true)

	nonzero2 := false
	for k := 0; k < mep.numTrainingData; k++ {
		if mep.evalMatrix[0][k] != 0.0 {
			nonzero2 = true
		}
	}
	equals(t, nonzero2, true)
	//fmt.Println(mep.evalMatrix[0][0])
}

func TestFitness(t *testing.T) {
	codeSize := 50
	popSize := 100
	mep := New(popSize, codeSize)
	mep.ReadTrainingData("mep/testdata/simple.txt", " ")
	mep.StartSteadyState()
	equals(t, mep.Pop[0].Fitness == 0.0, true)
}
