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
	Program program
	// an array of constants
	Constants constants
	// the fitness (or the error)
	// for regression is computed as sum of abs differences between target and obtained
	// for classification is computed as the number of incorrectly classified data
	Fitness float64
	// the index of the best expression in chromosome
	BestIndex int
}

type population []chromosome

// sort interface for population
func (slice population) Len() int {
	return len(slice)
}

func (slice population) Less(i, j int) bool {
	return slice[i].Fitness < slice[j].Fitness
}

func (slice population) Swap(i, j int) {
	slice[i], slice[j] = slice[j], slice[i]
}

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
	trainingData         TrainingData
	target               Target
	Pop                  population
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

	// TODO force this:
	// sum of variables_prob + operators_prob + constants_prob MUST BE 1 !
	m.VariablesProbability = 0.4
	m.OperatorsProbability = 0.5
	m.ConstantsProbability = 1 - m.VariablesProbability - m.OperatorsProbability
	m.ConstantsNum = 3
	m.ConstantsMin = -1
	m.ConstantsMax = 1

	// initialize population
	m.Pop = make(population, m.PopSize)

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

	m.randomPopulation()

	return nil
}

// StartSteadyState -
func (m *Mep) StartSteadyState() {

	offspring1 := m.randomChromosome()
	offspring2 := m.randomChromosome()

	for k := 0; k < m.PopSize; k += 2 {

		// binary tournament
		r1 := m.tournamentSelection(2)
		r2 := m.tournamentSelection(2)
		m.copyChromosome(&m.Pop[r1], &offspring1)
		m.copyChromosome(&m.Pop[r2], &offspring2)

		// crossover
		if rand.Float64() < m.CrossoverProbability {
			m.oneCutPointCrossover(&m.Pop[r1], &m.Pop[r2], &offspring1, &offspring2)
		}

		// mutatation
		m.mutation(&offspring1)
		m.fitness(&offspring1)

		m.mutation(&offspring2)
		m.fitness(&offspring2)

		// replace the worst in the population
		if offspring1.Fitness < m.Pop[m.PopSize-1].Fitness {
			m.copyChromosome(&offspring1, &m.Pop[m.PopSize-1])
		}
		if offspring2.Fitness < m.Pop[m.PopSize-1].Fitness {
			m.copyChromosome(&offspring2, &m.Pop[m.PopSize-1])
		}

		sort.Sort(m.Pop)
	}
}

// PrintIndividual -
func (m *Mep) PrintIndividual(popIndex int) {
	exp := m.GetExpression("", m.Pop[0], m.Pop[0].BestIndex)
	fmt.Printf("fitness = %f, exp=%s\n", m.Pop[popIndex].Fitness, exp)
}

// GetExpression -
func (m *Mep) GetExpression(exp string, individual chromosome, poz int) string {

	code := individual.Program
	op := code[poz].op
	adr1 := code[poz].adr1
	adr2 := code[poz].adr2

	if op == -1 { // +
		exp = m.GetExpression(exp, individual, adr1)
		exp += "+"
		exp = m.GetExpression(exp, individual, adr2)
	} else if op == -2 { // -
		exp = m.GetExpression(exp, individual, adr1)
		exp += "-"
		if code[adr2].op == -1 || code[adr2].op == -2 {
			exp += "("
		}
		exp = m.GetExpression(exp, individual, adr2)
		if code[adr2].op == -1 || code[adr2].op == -2 {
			exp += ")"
		}
	} else if op == -3 { // *
		if code[adr1].op == -1 || code[adr1].op == -2 {
			exp += "("
		}
		exp = m.GetExpression(exp, individual, adr1)
		if code[adr1].op == -1 || code[adr1].op == -2 {
			exp += ")"
		}
		exp += "*"
		if code[adr2].op == -1 || code[adr2].op == -2 {
			exp += "("
		}
		exp = m.GetExpression(exp, individual, adr2)
		if code[adr2].op == -1 || code[adr2].op == -2 {
			exp += ")"
		}
	} else if op == -4 { // /
		if code[adr1].op == -1 || code[adr1].op == -2 {
			exp += "("
		}
		exp = m.GetExpression(exp, individual, adr1)
		if code[adr1].op == -1 || code[adr1].op == -2 {
			exp += ")"
		}
		exp += "/"
		if code[adr2].op == -1 || code[adr2].op == -2 {
			exp += "()"
		}
		exp = m.GetExpression(exp, individual, adr2)
		if code[adr2].op == -1 || code[adr2].op == -2 {
			exp += ")"
		}
	} else if op < m.numVariables {
		exp += fmt.Sprintf("x%d", op)
	} else {
		exp += fmt.Sprintf("%f", individual.Constants[op-m.numVariables])
	}
	return exp
}

func (m *Mep) randomTerminal() int {
	var op int
	prob := rand.Float64() * (m.VariablesProbability + m.ConstantsProbability)
	if prob <= m.VariablesProbability {
		op = rand.Intn(m.numVariables)
	} else {
		op = m.numVariables + rand.Intn(m.ConstantsNum)
	}
	return op
}

func (m *Mep) randomAdr(index int) int {
	return rand.Intn(index)
}

func (m *Mep) randomCode(index int) int {
	var op int
	p := rand.Float64()
	if p <= m.OperatorsProbability {
		op = -(rand.Intn(len(operators)) + 1) // an operator
	} else {
		if p <= m.OperatorsProbability+m.VariablesProbability {
			op = rand.Intn(m.numVariables) // a variable
		} else {
			op = m.numVariables + rand.Intn(m.ConstantsNum) // index of a constant
		}
	}
	return op
}

func (m *Mep) randomConstant() float64 {
	return rand.Float64()*(m.ConstantsMax-m.ConstantsMin) + m.ConstantsMin
}

func (m *Mep) randomChromosome() chromosome {

	a := chromosome{}
	a.Program = make(program, m.CodeLength)
	if m.ConstantsNum > 0 {
		a.Constants = make(constants, m.ConstantsNum)
	}

	// generate constants first
	for c := 0; c < m.ConstantsNum; c++ {
		a.Constants[c] = m.randomConstant()
	}

	// on the first position we can have only a variable or a constant
	a.Program[0].op = m.randomTerminal()

	// for all other genes we put either an operator, variable or constant
	for i := 1; i < m.CodeLength; i++ {
		a.Program[i].op = m.randomCode(i)
		a.Program[i].adr1 = m.randomAdr(i)
		a.Program[i].adr2 = m.randomAdr(i)
	}

	m.fitness(&a)

	return a
}

func (m *Mep) randomPopulation() {

	for i := 0; i < m.PopSize; i++ {
		m.Pop[i] = m.randomChromosome()
	}

	// sort by fitness ascending
	sort.Sort(m.Pop)
}

func (m *Mep) fitness(c *chromosome) {

	c.Fitness = 1e+308
	c.BestIndex = -1

	m.computeEvalMatrix(c)

	for i := 0; i < m.CodeLength; i++ {
		sumOfErrors := 0.0
		for k := 0; k < m.numTrainingData; k++ {
			sumOfErrors += math.Abs(m.evalMatrix[i][k] - m.target[k]) // difference between obtained and expected
		}
		if c.Fitness > sumOfErrors {
			c.Fitness = sumOfErrors
			c.BestIndex = i
		}
	}
}

func (m *Mep) tournamentSelection(tournamentSize int) int {

	p := rand.Intn(m.PopSize)
	for i := 1; i < tournamentSize; i++ {
		r := rand.Intn(m.PopSize)
		if m.Pop[r].Fitness < m.Pop[p].Fitness {
			p = r
		}
	}
	return p
}

func (m *Mep) oneCutPointCrossover(parent1, parent2, offspring1, offspring2 *chromosome) {

	cuttingPoint := rand.Intn(m.CodeLength)
	for i := 0; i < cuttingPoint; i++ {
		offspring1.Program[i] = parent1.Program[i]
		offspring2.Program[i] = parent2.Program[i]
	}
	for i := cuttingPoint; i < m.CodeLength; i++ {
		offspring1.Program[i] = parent2.Program[i]
		offspring2.Program[i] = parent1.Program[i]
	}

	// now the constants
	if m.ConstantsNum > 0 {
		cuttingPoint = rand.Intn(m.ConstantsNum)
		for i := 0; i < cuttingPoint; i++ {
			offspring1.Constants[i] = parent1.Constants[i]
			offspring2.Constants[i] = parent2.Constants[i]
		}
		for i := cuttingPoint; i < m.ConstantsNum; i++ {
			offspring1.Constants[i] = parent1.Constants[i]
			offspring2.Constants[i] = parent2.Constants[i]
		}
	}
}

func (m *Mep) uniformCrossover(parent1, parent2, offspring1, offspring2 *chromosome) {

	// code
	for i := 0; i < m.CodeLength; i++ {
		if rand.Float64() < 0.5 {
			offspring1.Program[i] = parent1.Program[i]
			offspring2.Program[i] = parent2.Program[i]
		} else {
			offspring1.Program[i] = parent2.Program[i]
			offspring2.Program[i] = parent1.Program[i]
		}
	}

	// constants
	for i := 0; i < m.ConstantsNum; i++ {
		if (rand.Int() % 2) == 0 {
			offspring1.Constants[i] = parent1.Constants[i]
			offspring2.Constants[i] = parent2.Constants[i]
		} else {
			offspring1.Constants[i] = parent2.Constants[i]
			offspring2.Constants[i] = parent1.Constants[i]
		}
	}
}

func (m *Mep) mutation(aChromosome *chromosome) {

	// mutate each symbol with the given probability
	// first gene must be a variable or constant
	if rand.Float64() < m.MutationProbability {
		aChromosome.Program[0].op = m.randomTerminal()
	}

	for i := 1; i < m.CodeLength; i++ {

		if rand.Float64() < m.MutationProbability {
			aChromosome.Program[i].op = m.randomCode(i)
		}

		if rand.Float64() < m.MutationProbability {
			aChromosome.Program[i].adr1 = m.randomAdr(i)
		}

		if rand.Float64() < m.MutationProbability {
			aChromosome.Program[i].adr2 = m.randomAdr(i)
		}
	}

	// mutate the constants
	for c := 0; c < m.ConstantsNum; c++ {
		if rand.Float64() < m.MutationProbability {
			aChromosome.Constants[c] = m.randomConstant()
		}
	}
}

func (m *Mep) computeEvalMatrix(c *chromosome) {

	// we keep intermediate values in a matrix because when an error occurs (like division by 0) we mutate that gene into a variables.
	// in such case it is faster to have all intermediate results until current gene, so that we don't have to recompute them again.

	var isErrorCase bool // division by zero, other errors

	for i := 0; i < m.CodeLength; i++ { // read the chromosome from top to down

		isErrorCase = false
		switch c.Program[i].op {
		case -1: // +
			for k := 0; k < m.numTrainingData; k++ {
				m.evalMatrix[i][k] = m.evalMatrix[c.Program[i].adr1][k] + m.evalMatrix[c.Program[i].adr2][k]
			}
		case -2: // -
			for k := 0; k < m.numTrainingData; k++ {
				m.evalMatrix[i][k] = m.evalMatrix[c.Program[i].adr1][k] - m.evalMatrix[c.Program[i].adr2][k]
			}
		case -3: // *
			for k := 0; k < m.numTrainingData; k++ {
				m.evalMatrix[i][k] = m.evalMatrix[c.Program[i].adr1][k] * m.evalMatrix[c.Program[i].adr2][k]
			}
		case -4: //  /
			for k := 0; k < m.numTrainingData; k++ {
				if math.Abs(m.evalMatrix[c.Program[i].adr2][k]) < 1e-6 { // a small constant
					isErrorCase = true
				}
			}
			if isErrorCase { // an division by zero error occured !!!
				c.Program[i].op = rand.Intn(m.numVariables) // the gene is mutated into a terminal
				for k := 0; k < m.numTrainingData; k++ {
					m.evalMatrix[i][k] = m.trainingData[k][c.Program[i].op]
				}

			} else { // normal execution....
				for k := 0; k < m.numTrainingData; k++ {
					m.evalMatrix[i][k] = m.evalMatrix[c.Program[i].adr1][k] / m.evalMatrix[c.Program[i].adr2][k]
				}
			}
		default: // a variable
			for k := 0; k < m.numTrainingData; k++ {
				if c.Program[i].op < m.numVariables {
					m.evalMatrix[i][k] = m.trainingData[k][c.Program[i].op]
				} else {
					m.evalMatrix[i][k] = c.Constants[c.Program[i].op-m.numVariables]
				}
			}
		}
	}
}

func (m *Mep) copyChromosome(source, dest *chromosome) {

	for i := 0; i < m.CodeLength; i++ {
		dest.Program[i] = source.Program[i]
	}

	for i := 0; i < m.ConstantsNum; i++ {
		dest.Constants[i] = source.Constants[i]
	}
	dest.Fitness = source.Fitness
	dest.BestIndex = source.BestIndex
}
