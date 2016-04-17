package mep

import (
	"bufio"
	"fmt"
	"io/ioutil"
	"math"
	"math/rand"
	"sort"
	"strconv"
	"strings"
)

// Based on: Multi Expression Programming - Mihai Oltean  (mihai.oltean@gmail.com)

var operators = []string{"+", "-", "*", "/"}

type instruction struct {
	// either a variable, operator or constant
	// variables are indexed from 0: 0,1,2,...
	// constants are indexed from num_variables
	// operators are -1, -2, -3...
	op int
	// index to arguments
	adr1 int
	adr2 int
}
type program []instruction
type constants []float64

type chromosome struct {
	// the program - a string of genes
	prog program
	// an array of constants
	consts constants
	// the fitness (or the error)
	// for regression is computed as sum of abs differences between target and obtained
	// for classification is computed as the number of incorrectly classified data
	fitness float64
	// the index of the best expression in chromosome
	bestIndex int
}

type population []chromosome

// sort interface for population
func (slice population) Len() int {
	return len(slice)
}

func (slice population) Less(i, j int) bool {
	return slice[i].fitness < slice[j].fitness
}

func (slice population) Swap(i, j int) {
	slice[i], slice[j] = slice[j], slice[i]
}

// ProblemType -
type ProblemType int

const (
	// REGRESSION -
	REGRESSION ProblemType = iota
	// CLASSIFICATION -
	CLASSIFICATION
)

// Target -
type Target []float64

// TrainingData -
type TrainingData [][]float64

// Mep -
type Mep struct {
	PopSize              int
	CodeLength           int
	MutationProbability  float64
	CrossoverProbability float64
	VariablesProbability float64
	OperatorsProbability float64
	ConstantsProbability float64
	ConstantsNum         int
	ConstantsMin         float64
	ConstantsMax         float64
	ProblemType          ProblemType
	trainingData         TrainingData
	target               Target
	pop                  population
	numVariables         int
	numTrainingData      int
	evalMatrix           [][]float64
}

// New -
func New(popSize, codeLen int) Mep {
	m := Mep{}

	// mandatory
	if (popSize % 2) != 0 {
		panic("invalid popSize, must be an even number")
	}
	m.PopSize = popSize
	if codeLen < 4 {
		panic("invalid codeLen, should be >= 4")
	}
	m.CodeLength = codeLen

	// defaults
	m.MutationProbability = 0.1
	m.CrossoverProbability = 0.9
	// sum of variables_prob + operators_prob + constants_prob MUST BE 1 !
	m.VariablesProbability = 0.4
	m.OperatorsProbability = 0.5
	m.ConstantsProbability = 1 - m.VariablesProbability - m.OperatorsProbability
	m.ConstantsNum = 3
	m.ConstantsMin = -1
	m.ConstantsMax = 1
	m.ProblemType = REGRESSION

	// initialize population
	m.pop = make(population, m.PopSize)
	for i := 0; i < m.PopSize; i++ {
		m.pop[i] = m.newChromosome()
	}

	return m
}

// ReadTrainingData -
func (m *Mep) ReadTrainingData(filename, sep string) error {

	content, err := ioutil.ReadFile(filename)
	if err != nil {
		return err
	}
	lines := strings.Split(string(content), "\n")
	m.trainingData = make(TrainingData, len(lines))
	m.target = make(Target, len(lines))

	for i, line := range lines {

		var floats []float64
		r := strings.NewReader(line)
		scanner := bufio.NewScanner(r)
		scanner.Split(bufio.ScanWords)
		for scanner.Scan() {
			x, _ := strconv.ParseFloat(scanner.Text(), 64)
			floats = append(floats, x)
		}
		if len(floats) > 1 {
			m.trainingData[i] = floats[0 : len(floats)-1]
			m.target[i] = floats[len(floats)-1]
		}
	}

	m.numVariables = len(m.trainingData[0])
	m.numTrainingData = len(m.trainingData)

	m.evalMatrix = make([][]float64, m.CodeLength)
	for i := 0; i < m.CodeLength; i++ {
		m.evalMatrix[i] = make([]float64, m.numTrainingData)
	}

	return nil
}

// StartSteadyState -
func (m *Mep) StartSteadyState() {

	// a steady state approach:
	// we work with 1 population
	// newly created individuals will replace the worst existing ones (only if they are better)

	offspring1 := m.newChromosome()
	offspring2 := m.newChromosome()

	// initialize
	for i := 0; i < m.PopSize; i++ {
		m.pop[i] = m.randomChromosome()
		if m.ProblemType == REGRESSION {
			m.fitnessRegression(m.pop[i])
		} else {
			m.fitnessClassification(m.pop[i])
		}
	}

	// sort ascendingly by fitness
	sort.Sort(m.pop)

	for k := 0; k < m.PopSize; k += 2 {

		// choose the parents using binary tournament
		r1 := m.tournamentSelection(2)
		r2 := m.tournamentSelection(2)

		// crossover
		p := rand.Float64()
		if p < m.CrossoverProbability {
			m.oneCutPointCrossover(&m.pop[r1], &m.pop[r2], &offspring1, &offspring2)
		} else { // no crossover so the offspring are a copy of the parents
			m.copyIndividual(&offspring1, &m.pop[r1])
			m.copyIndividual(&offspring2, &m.pop[r2])
		}

		// mutate the result and compute fitness
		m.mutation(&offspring1)

		if m.ProblemType == REGRESSION {
			m.fitnessRegression(offspring1)
		} else {
			m.fitnessClassification(offspring1)
		}
		// mutate the other offspring and compute fitness
		m.mutation(&offspring2)
		if m.ProblemType == REGRESSION {
			m.fitnessRegression(offspring2)
		} else {
			m.fitnessClassification(offspring2)
		}

		// replace the worst in the population
		if offspring1.fitness < m.pop[m.PopSize-1].fitness {
			m.copyIndividual(&m.pop[m.PopSize-1], &offspring1)
			sort.Sort(m.pop)
		}
		if offspring2.fitness < m.pop[m.PopSize-1].fitness {
			m.copyIndividual(&m.pop[m.PopSize-1], &offspring2)
			sort.Sort(m.pop)
		}
	}
}

// PrintIndividual -
func (m *Mep) PrintIndividual(popIndex int) {

	fmt.Printf("The chromosome is:\n")
	fmt.Println("constants =")
	fmt.Println(m.pop[popIndex].consts)

	for i := 0; i < m.CodeLength; i++ {
		c := m.pop[popIndex]
		if c.prog[i].op < 0 {
			fmt.Printf("%d: %s %d %d\n", i, operators[int(math.Abs(float64(c.prog[i].op)))-1], c.prog[i].adr1, c.prog[i].adr2)
		} else {
			if m.pop[popIndex].prog[i].op < m.numVariables {
				fmt.Printf("%d: inputs[%d]\n", i, m.pop[popIndex].prog[i].op)
			} else {
				fmt.Printf("%d: constants[%d]\n", i, m.pop[popIndex].prog[i].op-m.numVariables)
			}
		}
	}

	fmt.Printf("best index = %d\n", m.pop[popIndex].bestIndex)
	fmt.Printf("fitness = %f\n", m.pop[popIndex].fitness)
}

func (m *Mep) newChromosome() chromosome {
	c := chromosome{}
	c.prog = make(program, m.CodeLength)
	if m.ConstantsNum > 0 {
		c.consts = make(constants, m.ConstantsNum)
	}
	return c
}

func (m *Mep) randomChromosome() chromosome {

	a := m.newChromosome()

	// generate constants first
	for c := 0; c < m.ConstantsNum; c++ {
		a.consts[c] = rand.Float64()*(m.ConstantsMax-m.ConstantsMin) + m.ConstantsMin
		//TODO check this - a.constants[c] = math.Rand() / double(RAND_MAX) * (params.constantsMax - params.constantsMin) + params.constantsMin
	}

	// on the first position we can have only a variable or a constant
	sum := m.VariablesProbability + m.ConstantsProbability
	p := rand.Float64() * sum
	//TODO check this - double p = rand() / (double)RAND_MAX * sum;

	if p <= m.VariablesProbability {
		a.prog[0].op = rand.Intn(m.numVariables)
	} else {
		a.prog[0].op = m.numVariables + rand.Intn(m.ConstantsNum)
	}
	// for all other genes we put either an operator, variable or constant
	for i := 1; i < m.CodeLength; i++ {
		p := rand.Float64()

		if p <= m.OperatorsProbability {
			a.prog[i].op = -(rand.Intn(len(operators))) - 1 // an operator
		} else {
			if p <= m.OperatorsProbability+m.VariablesProbability {
				a.prog[i].op = rand.Intn(m.numVariables) // a variable
			} else {
				a.prog[i].op = m.numVariables + rand.Intn(m.ConstantsNum) // index of a constant
			}
		}
		//fmt.Printf("op=%d\n", a.prog[i].op)
		a.prog[i].adr1 = rand.Int() % i
		a.prog[i].adr2 = rand.Int() % i
	}
	return a
}

func (m *Mep) fitnessRegression(c chromosome) {

	c.fitness = 1e+308
	c.bestIndex = -1

	m.computeEvalMatrix(c)

	for i := 0; i < m.CodeLength; i++ { // read the chromosome from top to down
		sumOfErrors := 0.0
		for k := 0; k < m.numTrainingData; k++ {
			sumOfErrors += math.Abs(m.evalMatrix[i][k] - m.target[k]) // difference between obtained and expected
		}
		if c.fitness > sumOfErrors {
			c.fitness = sumOfErrors
			c.bestIndex = i
		}
	}
}

func (m *Mep) fitnessClassification(c chromosome) {

	c.fitness = 1e+308
	c.bestIndex = -1

	m.computeEvalMatrix(c)

	for i := 0; i < m.CodeLength; i++ { // read the chromosome from top to down

		var countIncorrectClassified float64

		for k := 0; k < m.numTrainingData; k++ {
			if m.evalMatrix[i][k] < 0 { // the program tells me that this data is in class 0
				countIncorrectClassified += m.target[k]
			} else { // the program tells me that this data is in class 1
				countIncorrectClassified += math.Abs(1 - m.target[k]) // difference between obtained and expected
			}
			if c.fitness > countIncorrectClassified {
				c.fitness = countIncorrectClassified
				c.bestIndex = i
			}
		}
	}
}

func (m *Mep) computeEvalMatrix(c chromosome) {

	// we keep intermediate values in a matrix because when an error occurs (like division by 0) we mutate that gene into a variables.
	// in such case it is faster to have all intermediate results until current gene, so that we don't have to recompute them again.

	var isErrorCase bool // division by zero, other errors

	for i := 0; i < m.CodeLength; i++ { // read the chromosome from top to down

		isErrorCase = false
		switch c.prog[i].op {
		case -1: // +
			for k := 0; k < m.numTrainingData; k++ {
				m.evalMatrix[i][k] = m.evalMatrix[c.prog[i].adr1][k] + m.evalMatrix[c.prog[i].adr2][k]
			}
		case -2: // -
			for k := 0; k < m.numTrainingData; k++ {
				m.evalMatrix[i][k] = m.evalMatrix[c.prog[i].adr1][k] - m.evalMatrix[c.prog[i].adr2][k]
			}
		case -3: // *
			for k := 0; k < m.numTrainingData; k++ {
				m.evalMatrix[i][k] = m.evalMatrix[c.prog[i].adr1][k] * m.evalMatrix[c.prog[i].adr2][k]
			}
		case -4: //  /
			for k := 0; k < m.numTrainingData; k++ {
				if math.Abs(m.evalMatrix[c.prog[i].adr2][k]) < 1e-6 { // a small constant
					isErrorCase = true
				}
			}
			if isErrorCase { // an division by zero error occured !!!
				c.prog[i].op = rand.Intn(m.numVariables) // the gene is mutated into a terminal
				for k := 0; k < m.numTrainingData; k++ {
					m.evalMatrix[i][k] = m.trainingData[k][c.prog[i].op]
				}

			} else { // normal execution....
				for k := 0; k < m.numTrainingData; k++ {
					m.evalMatrix[i][k] = m.evalMatrix[c.prog[i].adr1][k] / m.evalMatrix[c.prog[i].adr2][k]
				}
			}
		default: // a variable
			for k := 0; k < m.numTrainingData; k++ {
				if c.prog[i].op < m.numVariables {
					m.evalMatrix[i][k] = m.trainingData[k][c.prog[i].op]
				} else {
					m.evalMatrix[i][k] = c.consts[c.prog[i].op-m.numVariables]
				}
			}
		}
	}
}

func (m *Mep) tournamentSelection(tournamentSize int) int {

	p := rand.Intn(m.PopSize)
	for i := 1; i < tournamentSize; i++ {
		r := rand.Intn(m.PopSize)
		if m.pop[r].fitness < m.pop[p].fitness {
			p = r
		}
	}
	return p
}

func (m *Mep) oneCutPointCrossover(parent1, parent2, offspring1, offspring2 *chromosome) {

	cuttingPct := rand.Intn(m.CodeLength)

	for i := 0; i < cuttingPct; i++ {
		offspring1.prog[i] = parent1.prog[i]
		offspring2.prog[i] = parent2.prog[i]
	}
	for i := cuttingPct; i < m.CodeLength; i++ {
		offspring1.prog[i] = parent2.prog[i]
		offspring2.prog[i] = parent1.prog[i]
	}
	// now the constants
	if m.ConstantsNum > 0 {
		cuttingPct = rand.Intn(m.ConstantsNum)
		for i := 0; i < cuttingPct; i++ {
			offspring1.consts[i] = parent1.consts[i]
			offspring2.consts[i] = parent2.consts[i]
		}
		for i := cuttingPct; i < m.ConstantsNum; i++ {
			offspring1.consts[i] = parent2.consts[i]
			offspring2.consts[i] = parent1.consts[i]
		}
	}
}

func (m *Mep) uniformCrossover(parent1, parent2, offspring1, offspring2 *chromosome) {
	// code
	for i := 0; i < m.CodeLength; i++ {
		if (rand.Int() % 2) == 0 {
			offspring1.prog[i] = parent1.prog[i]
			offspring2.prog[i] = parent2.prog[i]
		} else {
			offspring1.prog[i] = parent2.prog[i]
			offspring2.prog[i] = parent1.prog[i]
		}
	}

	// constants
	for i := 0; i < m.ConstantsNum; i++ {
		if (rand.Int() % 2) == 0 {
			offspring1.consts[i] = parent1.consts[i]
			offspring2.consts[i] = parent2.consts[i]
		} else {
			offspring1.consts[i] = parent2.consts[i]
			offspring2.consts[i] = parent1.consts[i]
		}
	}
}

func (m *Mep) copyIndividual(source, dest *chromosome) {

	for i := 0; i < m.CodeLength; i++ {
		dest.prog[i] = source.prog[i]
	}

	for i := 0; i < m.ConstantsNum; i++ {
		dest.consts[i] = source.consts[i]
	}
	dest.fitness = source.fitness
	dest.bestIndex = source.bestIndex
}

func (m *Mep) mutation(aChromosome *chromosome) {

	// mutate each symbol with the given probability
	// first gene must be a variable or constant
	p := rand.Float64()
	if p < m.MutationProbability {
		sum := m.VariablesProbability + m.ConstantsProbability
		p = rand.Float64() * sum

		if p <= m.VariablesProbability {
			aChromosome.prog[0].op = rand.Intn(m.numVariables)
		} else {
			aChromosome.prog[0].op = m.numVariables + rand.Intn(m.ConstantsNum)
		}
	}
	// other genes
	for i := 1; i < m.CodeLength; i++ {
		p = rand.Float64() // mutate the operator
		if p < m.MutationProbability {

			// we mutate it, but we have to decide what we put here
			p = rand.Float64()

			if p <= m.OperatorsProbability {
				aChromosome.prog[i].op = -rand.Intn(len(operators)) - 1
			} else {
				if p <= m.OperatorsProbability+m.VariablesProbability {
					aChromosome.prog[i].op = rand.Intn(m.numVariables)
				} else {
					aChromosome.prog[i].op = m.numVariables + rand.Intn(m.ConstantsNum) // index of a constant
				}
			}
		}

		p = rand.Float64() // mutate the first address  (adr1)
		if p < m.MutationProbability {
			aChromosome.prog[i].adr1 = rand.Intn(i)
		}

		p = rand.Float64() // mutate the second address   (adr2)
		if p < m.MutationProbability {
			aChromosome.prog[i].adr2 = rand.Intn(i)
		}
	}

	// mutate the constants
	for c := 0; c < m.ConstantsNum; c++ {
		p = rand.Float64()
		if p < m.MutationProbability {
			aChromosome.consts[c] = rand.Float64()*(m.ConstantsMax-m.ConstantsMin) + m.ConstantsMin
		}
	}

}
