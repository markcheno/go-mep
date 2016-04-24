package mep

import (
	"fmt"
	"io/ioutil"
	"math"
	"math/rand"
	"sort"
	"strconv"
	"strings"
)

// FitnessFunction -
type FitnessFunction func(signal, target []float64) float64

// TotalErrorFF -
func TotalErrorFF(signal, target []float64) float64 {
	total := 0.0
	for i := 0; i < len(signal); i++ {
		total += math.Abs(signal[i] - target[i])
	}
	return total
}

// MeanErrorFF -
func MeanErrorFF(signal, target []float64) float64 {
	mean := 0.0
	for i := 0; i < len(signal); i++ {
		mean += math.Abs(signal[i] - target[i])
	}
	return mean / float64(len(signal))
}

// TrainingData -
type TrainingData struct {
	Train  [][]float64
	Target []float64
	Labels []string
}

// ReadTrainingData -
func ReadTrainingData(filename string, header bool, sep string) TrainingData {

	var labels []string
	var train [][]float64
	var target []float64

	content, err := ioutil.ReadFile(filename)
	if err != nil {
		return TrainingData{}
	}
	lines := strings.Split(string(content), "\n")

	if header {
		labels = strings.Split(lines[0], sep)
		lines = append(lines[:0], lines[1:]...)
	}

	train = make([][]float64, len(lines))
	target = make([]float64, len(lines))

	for i := 0; i < len(lines); i++ {
		var floats []float64
		for _, f := range strings.Split(lines[i], sep) {
			x, _ := strconv.ParseFloat(f, 64)
			floats = append(floats, x)
		}
		train[i] = floats[0 : len(floats)-1]
		target[i] = floats[len(floats)-1]
	}

	if !header {
		for i := 0; i < len(train[0])+1; i++ {
			labels = append(labels, fmt.Sprintf("x%d", i))
		}
	}

	return TrainingData{train, target, labels}
}

type instruction struct {
	// either a variable, operator or constant
	// variables are indexed from 0: 0,1,2,...
	// constants are indexed from num_variables
	// operators are -1, -2, -3...
	op   int
	adr1 int
	adr2 int
}
type program []instruction
type constants []float64
type chromosome struct {
	program   program
	constants constants
	fitness   float64
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

//
type operator struct {
	op      int
	name    string
	enabled bool
}

// Mep -
type Mep struct {
	MutationProbability      float64
	CrossoverProbability     float64
	popSize                  int
	codeLength               int
	td                       TrainingData
	ff                       FitnessFunction
	variablesProbability     float64
	operatorsProbability     float64
	randConstantsProbability float64
	numRandConstants         int
	randConstantsMin         float64
	randConstantsMax         float64
	numVariables             int
	numTraining              int
	pop                      population
	results                  [][]float64
	operators                []operator
}

// New -
func New(td TrainingData, ff FitnessFunction, popSize, codeLen int) Mep {

	m := Mep{}
	m.ff = ff
	m.td = td

	m.numTraining = len(m.td.Train)
	m.numVariables = len(m.td.Train[0])
	if m.numTraining == 0 || m.numVariables == 0 {
		panic("Invalid data")
	}

	m.popSize = popSize
	m.codeLength = codeLen
	if (m.popSize % 2) != 0 {
		panic("invalid popSize, must be an even number")
	}

	if m.codeLength < 4 {
		panic("invalid codeLength, should be >= 4")
	}

	m.operators = []operator{
		{-1, "add", true},
		{-2, "sub", true},
		{-3, "mul", true},
		{-4, "div", true},
		{-5, "sin", false},
		{-6, "cos", false},
		{-7, "tan", false},
		{-8, "exp", false},
		{-9, "log", false},
		{-10, "sqrt", false},
		{-11, "abs", false},
	}

	// defaults
	m.MutationProbability = 0.1
	m.CrossoverProbability = 0.9

	m.variablesProbability = 0.5
	m.operatorsProbability = 0.5
	m.numRandConstants = 0
	m.randConstantsMin = -1
	m.randConstantsMax = 1
	m.randConstantsProbability = 0

	// pre-allocate results matrix
	m.results = make([][]float64, m.codeLength)
	for i := 0; i < m.codeLength; i++ {
		m.results[i] = make([]float64, m.numTraining)
	}

	// initialize population
	m.pop = make(population, m.popSize)
	m.randomPopulation()

	return m
}

// SetConst -
func (m *Mep) SetConst(num int, min, max float64) {
	m.numRandConstants = num
	m.randConstantsMax = max
	m.randConstantsMin = min
	m.variablesProbability = 0.5
	m.operatorsProbability = 0.4
	m.randConstantsProbability = 1 - m.variablesProbability - m.operatorsProbability
	// initialize population
	m.pop = make(population, m.popSize)
	m.randomPopulation()
}

// SetOper - enable/disable operator
func (m *Mep) SetOper(operName string, state bool) {
	found := false
	var index int
	for index = 0; index < len(m.operators); index++ {
		if m.operators[index].name == operName {
			found = true
			break
		}
	}
	if !found {
		return
	}
	if state && !m.operators[index].enabled {
		m.operators[index].enabled = true
	} else if !state && m.operators[index].enabled {
		m.operators[index].enabled = false
	}
}

// Search -
func (m *Mep) Search() {

	if (m.variablesProbability + m.operatorsProbability + m.randConstantsProbability) != 1.0 {
		panic("probabilities must sum to 1.0")
	}

	offspring1 := m.randomChromosome()
	offspring2 := m.randomChromosome()

	for k := 0; k < m.popSize; k += 2 {

		// binary tournament
		r1 := m.tournamentSelection(2)
		r2 := m.tournamentSelection(2)
		m.copyChromosome(&m.pop[r1], &offspring1)
		m.copyChromosome(&m.pop[r2], &offspring2)

		// crossover
		if rand.Float64() < m.CrossoverProbability {
			m.oneCutPointCrossover(&m.pop[r1], &m.pop[r2], &offspring1, &offspring2)
		}

		// mutatation
		m.mutation(&offspring1)
		m.eval(&offspring1)

		m.mutation(&offspring2)
		m.eval(&offspring2)

		// replace the worst in the population
		if offspring1.fitness < m.pop[m.popSize-1].fitness {
			m.copyChromosome(&offspring1, &m.pop[m.popSize-1])
		}
		if offspring2.fitness < m.pop[m.popSize-1].fitness {
			m.copyChromosome(&offspring2, &m.pop[m.popSize-1])
		}

		sort.Sort(m.pop)
	}
}

// BestFitness -
func (m *Mep) BestFitness() float64 {
	return m.pop[0].fitness
}

// BestExpr -
func (m *Mep) BestExpr() string {
	return m.parse("", m.pop[0], m.pop[0].bestIndex)
}

// Best -
func (m *Mep) Best() (float64, string) {
	return m.pop[0].fitness, m.parse("", m.pop[0], m.pop[0].bestIndex)
}

// PrintBest -
func (m *Mep) PrintBest() {
	exp := m.parse("", m.pop[0], m.pop[0].bestIndex)
	fmt.Printf("fitness = %f, expr=%s\n", m.pop[0].fitness, exp)
}

func (m *Mep) parse(exp string, individual chromosome, poz int) string {

	code := individual.program
	op := code[poz].op
	adr1 := code[poz].adr1
	adr2 := code[poz].adr2

	if op == -1 { // +
		exp = m.parse(exp, individual, adr1)
		exp += "+"
		exp = m.parse(exp, individual, adr2)
	} else if op == -2 { // -
		exp = m.parse(exp, individual, adr1)
		exp += "-"
		if code[adr2].op == -1 || code[adr2].op == -2 {
			exp += "("
		}
		exp = m.parse(exp, individual, adr2)
		if code[adr2].op == -1 || code[adr2].op == -2 {
			exp += ")"
		}
	} else if op == -3 { // *
		if code[adr1].op == -1 || code[adr1].op == -2 {
			exp += "("
		}
		exp = m.parse(exp, individual, adr1)
		if code[adr1].op == -1 || code[adr1].op == -2 {
			exp += ")"
		}
		exp += "*"
		if code[adr2].op == -1 || code[adr2].op == -2 {
			exp += "("
		}
		exp = m.parse(exp, individual, adr2)
		if code[adr2].op == -1 || code[adr2].op == -2 {
			exp += ")"
		}
	} else if op == -4 { // /
		if code[adr1].op == -1 || code[adr1].op == -2 {
			exp += "("
		}
		exp = m.parse(exp, individual, adr1)
		if code[adr1].op == -1 || code[adr1].op == -2 {
			exp += ")"
		}
		exp += "/"
		if code[adr2].op == -1 || code[adr2].op == -2 {
			exp += "("
		}
		exp = m.parse(exp, individual, adr2)
		if code[adr2].op == -1 || code[adr2].op == -2 {
			exp += ")"
		}
	} else if op == -5 { // sin
		exp += "sin("
		exp = m.parse(exp, individual, adr1)
		exp += ")"
	} else if op == -6 { // cos
		exp += "cos("
		exp = m.parse(exp, individual, adr1)
		exp += ")"
	} else if op == -7 { // tan
		exp += "tan("
		exp = m.parse(exp, individual, adr1)
		exp += ")"
	} else if op == -8 { // exp
		exp += "exp("
		exp = m.parse(exp, individual, adr1)
		exp += ")"
	} else if op == -9 { // log
		exp += "log("
		exp = m.parse(exp, individual, adr1)
		exp += ")"
	} else if op == -10 { // sqrt
		exp += "sqrt("
		exp = m.parse(exp, individual, adr1)
		exp += ")"
	} else if op == -11 { // abs
		exp += "abs("
		exp = m.parse(exp, individual, adr1)
		exp += ")"
	} else if op < m.numVariables {
		exp += m.td.Labels[op]
	} else {
		exp += fmt.Sprintf("%f", individual.constants[op-m.numVariables])
	}
	return exp
}

func (m *Mep) randomTerminal() int {
	var op int
	prob := rand.Float64() * (m.variablesProbability + m.randConstantsProbability)
	if prob <= m.variablesProbability {
		op = rand.Intn(m.numVariables)
	} else {
		op = m.numVariables + rand.Intn(m.numRandConstants)
	}
	return op
}

func (m *Mep) randomAdr(index int) int {
	return rand.Intn(index)
}

func (m *Mep) randomCode(index int) int {
	var op int
	p := rand.Float64()
	if p <= m.operatorsProbability {

		n := rand.Intn(len(m.operators))
		for !m.operators[n].enabled {
			n = rand.Intn(len(m.operators))
		}
		op = m.operators[n].op // an operator

	} else {

		if p <= m.operatorsProbability+m.variablesProbability {
			op = rand.Intn(m.numVariables) // a variable
		} else {
			op = m.numVariables + rand.Intn(m.numRandConstants) // index of a constant
		}

	}
	return op
}

func (m *Mep) randomConstant() float64 {
	return rand.Float64()*(m.randConstantsMax-m.randConstantsMin) + m.randConstantsMin
}

func (m *Mep) randomChromosome() chromosome {

	a := chromosome{}
	a.program = make(program, m.codeLength)
	if m.numRandConstants > 0 {
		a.constants = make(constants, m.numRandConstants)
	}

	// generate constants first
	for c := 0; c < m.numRandConstants; c++ {
		a.constants[c] = m.randomConstant()
	}

	// on the first position we can have only a variable or a constant
	a.program[0].op = m.randomTerminal()

	// for all other genes we put either an operator, variable or constant
	for i := 1; i < m.codeLength; i++ {
		a.program[i].op = m.randomCode(i)
		a.program[i].adr1 = m.randomAdr(i)
		a.program[i].adr2 = m.randomAdr(i)
	}

	m.eval(&a)

	return a
}

func (m *Mep) randomPopulation() {

	for i := 0; i < m.popSize; i++ {
		m.pop[i] = m.randomChromosome()
	}

	// sort by fitness ascending
	sort.Sort(m.pop)
}

func (m *Mep) eval(c *chromosome) {

	c.fitness = 1e+308
	c.bestIndex = -1

	// we keep intermediate values in a matrix because when an error occurs (like division by 0) we mutate that gene into a variables.
	// in such case it is faster to have all intermediate results until current gene, so that we don't have to recompute them again.

	var isErrorCase bool // division by zero, other errors

	for i := 0; i < m.codeLength; i++ { // read the chromosome from top to down

		isErrorCase = false
		switch c.program[i].op {
		case -1: // +
			for k := 0; k < m.numTraining; k++ {
				m.results[i][k] = m.results[c.program[i].adr1][k] + m.results[c.program[i].adr2][k]
			}
		case -2: // -
			for k := 0; k < m.numTraining; k++ {
				m.results[i][k] = m.results[c.program[i].adr1][k] - m.results[c.program[i].adr2][k]
			}
		case -3: // *
			for k := 0; k < m.numTraining; k++ {
				m.results[i][k] = m.results[c.program[i].adr1][k] * m.results[c.program[i].adr2][k]
			}
		case -4: //  /
			for k := 0; k < m.numTraining; k++ {
				if math.Abs(m.results[c.program[i].adr2][k]) < 1e-6 { // a small constant
					isErrorCase = true
				}
			}
			if isErrorCase { // an division by zero error occured !!!
				c.program[i].op = rand.Intn(m.numVariables) // the gene is mutated into a terminal
				for k := 0; k < m.numTraining; k++ {
					m.results[i][k] = m.td.Train[k][c.program[i].op]
				}

			} else { // normal execution....
				for k := 0; k < m.numTraining; k++ {
					m.results[i][k] = m.results[c.program[i].adr1][k] / m.results[c.program[i].adr2][k]
				}
			}
		case -5: //  sin
			for k := 0; k < m.numTraining; k++ {
				m.results[i][k] = math.Sin(m.results[c.program[i].adr1][k])
			}
		case -6: //  cos
			for k := 0; k < m.numTraining; k++ {
				m.results[i][k] = math.Cos(m.results[c.program[i].adr1][k])
			}
		case -7: //  tan
			for k := 0; k < m.numTraining; k++ {
				m.results[i][k] = math.Tan(m.results[c.program[i].adr1][k])
			}
		case -8: //  exp
			for k := 0; k < m.numTraining; k++ {
				m.results[i][k] = math.Exp(m.results[c.program[i].adr1][k])
			}
		case -9: //  log
			for k := 0; k < m.numTraining; k++ {
				m.results[i][k] = math.Log(m.results[c.program[i].adr1][k])
			}
		case -10: //  sqrt
			for k := 0; k < m.numTraining; k++ {
				m.results[i][k] = math.Sqrt(m.results[c.program[i].adr1][k])
			}
		case -11: //  abs
			for k := 0; k < m.numTraining; k++ {
				m.results[i][k] = math.Abs(m.results[c.program[i].adr1][k])
			}
		default: // a variable
			for k := 0; k < m.numTraining; k++ {
				if c.program[i].op < m.numVariables {
					m.results[i][k] = m.td.Train[k][c.program[i].op]
				} else {
					m.results[i][k] = c.constants[c.program[i].op-m.numVariables]
				}
			}
		}

		fitness := m.ff(m.results[i], m.td.Target)
		if c.fitness > fitness {
			c.fitness = fitness
			c.bestIndex = i
		}
	}
}

func (m *Mep) tournamentSelection(tournamentSize int) int {

	p := rand.Intn(m.popSize)
	for i := 1; i < tournamentSize; i++ {
		r := rand.Intn(m.popSize)
		if m.pop[r].fitness < m.pop[p].fitness {
			p = r
		}
	}
	return p
}

func (m *Mep) oneCutPointCrossover(parent1, parent2, offspring1, offspring2 *chromosome) {

	cuttingPoint := rand.Intn(m.codeLength)
	for i := 0; i < cuttingPoint; i++ {
		offspring1.program[i] = parent1.program[i]
		offspring2.program[i] = parent2.program[i]
	}
	for i := cuttingPoint; i < m.codeLength; i++ {
		offspring1.program[i] = parent2.program[i]
		offspring2.program[i] = parent1.program[i]
	}

	// now the constants
	if m.numRandConstants > 0 {
		cuttingPoint = rand.Intn(m.numRandConstants)
		for i := 0; i < cuttingPoint; i++ {
			offspring1.constants[i] = parent1.constants[i]
			offspring2.constants[i] = parent2.constants[i]
		}
		for i := cuttingPoint; i < m.numRandConstants; i++ {
			offspring1.constants[i] = parent1.constants[i]
			offspring2.constants[i] = parent2.constants[i]
		}
	}
}

func (m *Mep) uniformCrossover(parent1, parent2, offspring1, offspring2 *chromosome) {

	// code
	for i := 0; i < m.codeLength; i++ {
		if rand.Float64() < 0.5 {
			offspring1.program[i] = parent1.program[i]
			offspring2.program[i] = parent2.program[i]
		} else {
			offspring1.program[i] = parent2.program[i]
			offspring2.program[i] = parent1.program[i]
		}
	}

	// constants
	for i := 0; i < m.numRandConstants; i++ {
		if (rand.Int() % 2) == 0 {
			offspring1.constants[i] = parent1.constants[i]
			offspring2.constants[i] = parent2.constants[i]
		} else {
			offspring1.constants[i] = parent2.constants[i]
			offspring2.constants[i] = parent1.constants[i]
		}
	}
}

func (m *Mep) mutation(aChromosome *chromosome) {

	// mutate each symbol with the given probability
	// first gene must be a variable or constant
	if rand.Float64() < m.MutationProbability {
		aChromosome.program[0].op = m.randomTerminal()
	}

	for i := 1; i < m.codeLength; i++ {

		if rand.Float64() < m.MutationProbability {
			aChromosome.program[i].op = m.randomCode(i)
		}

		if rand.Float64() < m.MutationProbability {
			aChromosome.program[i].adr1 = m.randomAdr(i)
		}

		if rand.Float64() < m.MutationProbability {
			aChromosome.program[i].adr2 = m.randomAdr(i)
		}
	}

	// mutate the constants
	for c := 0; c < m.numRandConstants; c++ {
		if rand.Float64() < m.MutationProbability {
			aChromosome.constants[c] = m.randomConstant()
		}
	}
}

func (m *Mep) copyChromosome(source, dest *chromosome) {

	for i := 0; i < m.codeLength; i++ {
		dest.program[i] = source.program[i]
	}

	for i := 0; i < m.numRandConstants; i++ {
		dest.constants[i] = source.constants[i]
	}
	dest.fitness = source.fitness
	dest.bestIndex = source.bestIndex
}
