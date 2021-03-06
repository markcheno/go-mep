package mep

import (
	"bufio"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"
)

// see: http://blog.loadimpact.com/random-thoughts-about-go
func intn(max int) int {
	return int((rand.Float64() * float64(max)) + 0.5)
}

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

// ClassificationFF -
func ClassificationFF(signal, target []float64) float64 {

	logLoss := 1e-15
	for i := 0; i < len(signal); i++ {
		logLoss += target[i]*math.Log(signal[i]) + (1.0-target[i])*math.Log(1.0-signal[i])
	}
	return -1.0 / float64(len(signal)) * logLoss
}

// TrainingData -
type TrainingData struct {
	Train  [][]float64
	Target []float64
	Labels []string
}

// ReadTrainingData - read trainging data from a file
func ReadTrainingData(filename string, header bool, sep string) TrainingData {

	td := TrainingData{}

	inFile, _ := os.Open(filename)
	defer inFile.Close()
	scanner := bufio.NewScanner(inFile)
	scanner.Split(bufio.ScanLines)

	if header {
		scanner.Scan()
		headerLine := scanner.Text()
		td.Labels = strings.Split(strings.Replace(headerLine, "\"", "", -1), sep)
	}

	for scanner.Scan() {
		line := scanner.Text()
		var floats []float64
		for _, f := range strings.Split(line, sep) {
			x, _ := strconv.ParseFloat(f, 64)
			floats = append(floats, x)
		}
		td.Train = append(td.Train, floats[0:len(floats)-1])
		td.Target = append(td.Target, floats[len(floats)-1])
	}

	if !header {
		for i := 0; i < len(td.Train[0])+1; i++ {
			td.Labels = append(td.Labels, fmt.Sprintf("x%d", i))
		}
	}

	return td
}

/*
func ReadTrainingData(filename string, header bool, sep string) TrainingData {

	td := TrainingData{}

	content, err := ioutil.ReadFile(filename)
	if err != nil {
		return td
	}
	lines := strings.Split(string(content), "\n")

	if header {
		td.Labels = strings.Split(strings.Replace(lines[0], "\"", "", -1), sep)
		lines = append(lines[:0], lines[1:]...)
	}
	fmt.Println(td.Labels)

	td.Train = make([][]float64, len(lines))
	td.Target = make([]float64, len(lines))

	for i := 0; i < len(lines); i++ {
		var floats []float64
		for _, f := range strings.Split(lines[i], sep) {
			x, _ := strconv.ParseFloat(f, 64)
			floats = append(floats, x)
		}
		td.Train[i] = floats[0 : len(floats)-1]
		td.Target[i] = floats[len(floats)-1]
	}

	if !header {
		for i := 0; i < len(td.Train[0])+1; i++ {
			td.Labels = append(td.Labels, fmt.Sprintf("x%d", i))
		}
	}

	return td
}
*/

type instruction struct {
	// either a variable, operator or constant
	// variables are indexed from 0: 0,1,2,...
	// constants are indexed from num_variables
	// operators are -1, -2, -3...
	op   int
	adr1 int
	adr2 int
	adr3 int
	adr4 int
}

type program []instruction

type constants []float64

type chromosome struct {
	program   program
	constants constants
	fitness   float64
	bestIndex int
}

type subPopulation []chromosome

type population []subPopulation

// sort interface for population
func (slice subPopulation) Len() int {
	return len(slice)
}

func (slice subPopulation) Less(i, j int) bool {
	return slice[i].fitness < slice[j].fitness
}

func (slice subPopulation) Swap(i, j int) {
	slice[i], slice[j] = slice[j], slice[i]
}

type operator struct {
	op      int
	name    string
	enabled bool
}

// CrossoverType - uniform or onecutpoint
type CrossoverType int

const (
	// OneCutPoint - crossover type
	OneCutPoint CrossoverType = iota
	// Uniform - crossover type
	Uniform
)

// Mep - primary class
type Mep struct {
	mutationProbability  float64
	crossoverProbability float64
	subPopSize           int
	numSubpopulation     int
	curSubpopulation     int
	bestPop              int
	codeLength           int
	td                   TrainingData
	ff                   FitnessFunction
	variablesProbability float64
	operatorsProbability float64
	constantsProbability float64
	numRandConstants     int
	randConstantsMin     float64
	randConstantsMax     float64
	fixedConstants       constants
	numConstants         int
	numVariables         int
	numTraining          int
	pop                  population
	results              [][][]float64
	operators            []operator
	crossoverType        CrossoverType
}

// New - create a new Multi-Expression population
func New(td TrainingData, ff FitnessFunction) *Mep {

	m := Mep{}
	m.ff = ff
	m.td = td

	m.numTraining = len(m.td.Train)
	m.numVariables = len(m.td.Train[0])
	fmt.Printf("numTraining=%d, numVariables=%d\n", m.numTraining, m.numVariables)
	if m.numTraining == 0 || m.numVariables == 0 {
		panic("Invalid data")
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
		{-12, "max", false},
		{-13, "min", false},
		{-14, "ifgtz", false},
		{-15, "ifltz", false},
		{-16, "ifgt", false},
		{-17, "iflt", false},
		{-18, "ifbgt", false},
		{-19, "ifblt", false},
		{-20, "and", false},
		{-21, "or", false},
		{-22, "pow", false},
		{-23, "pow10", false},
		{-24, "log10", false},
		{-25, "log2", false},
		{-26, "floor", false},
		{-27, "ceil", false},
		{-28, "inv", false},
		{-29, "square", false},
	}

	// defaults
	m.subPopSize = 100
	m.numSubpopulation = 1
	m.codeLength = 50
	m.mutationProbability = 0.1
	m.crossoverProbability = 0.9
	m.variablesProbability = 0.5
	m.operatorsProbability = 0.5
	m.numRandConstants = 0
	m.randConstantsMin = -1
	m.randConstantsMax = 1
	m.constantsProbability = 0
	m.numConstants = 0
	m.crossoverType = OneCutPoint

	// initialize population
	m.randomPopulation()

	return &m
}

// SetPop - set population size and code length (resets population)
func (m *Mep) SetPop(popSize, numSubpopulation, codeLength int) {
	if (popSize % 2) != 0 {
		panic("invalid popSize, must be an even number")
	}

	if codeLength < 4 {
		panic("invalid codeLength, should be >= 4")
	}
	m.subPopSize = popSize
	m.numSubpopulation = numSubpopulation
	m.codeLength = codeLength
	// initialize population
	m.randomPopulation()
}

// SetConst - set fixed and random constants (resets population)
func (m *Mep) SetConst(fixed []float64, numRand int, minRand, maxRand float64) {

	m.numConstants = numRand + len(fixed)

	if m.numConstants > 0 {
		m.variablesProbability = 0.5
		m.operatorsProbability = 0.4
		m.constantsProbability = 1 - m.variablesProbability - m.operatorsProbability

	} else {
		m.variablesProbability = 0.5
		m.operatorsProbability = 0.5
		m.constantsProbability = 0.0
	}

	if len(fixed) > 0 {
		m.fixedConstants = make(constants, len(fixed))
		copy(m.fixedConstants, fixed)
	}

	if numRand > 0 {
		m.numRandConstants = numRand
		m.randConstantsMax = maxRand
		m.randConstantsMin = minRand
	} else {
		m.numRandConstants = 0
		m.randConstantsMax = 0
		m.randConstantsMin = 0
	}
	// initialize population
	m.randomPopulation()
}

// SetOper - enable/disable operator
func (m *Mep) SetOper(operName string, state bool) {
	for index := 0; index < len(m.operators); index++ {
		if m.operators[index].name == operName {
			if state && !m.operators[index].enabled {
				m.operators[index].enabled = true
			} else if !state && m.operators[index].enabled {
				m.operators[index].enabled = false
			}
		}
	}
}

// Oper - list of enabled operators
func (m *Mep) Oper(all bool) []string {
	var operators []string
	for index := 0; index < len(m.operators); index++ {
		if all {
			operators = append(operators, m.operators[index].name)
		} else if m.operators[index].enabled {
			operators = append(operators, m.operators[index].name)
		}
	}
	return operators
}

// SetCrossover - crossover type and probability (valid range 0.0 - 1.0)
func (m *Mep) SetCrossover(crossoverType CrossoverType, crossoverProbability float64) {
	m.crossoverType = crossoverType
	m.crossoverProbability = crossoverProbability
	if m.crossoverProbability < 0.0 || m.crossoverProbability > 1.0 {
		panic("invalid crossoverProbability")
	}
}

// SetMutation - mutation probability (valid range 0.0 - 1.0)
func (m *Mep) SetMutation(mutationProbability float64) {
	m.mutationProbability = mutationProbability
	if m.mutationProbability < 0.0 || m.mutationProbability > 1.0 {
		panic("invalid mutationProbability")
	}
}

// SetProb - set mutation/crossover probability (valid range 0.0 - 1.0)
func (m *Mep) SetProb(mutationProbability, crossoverProbability float64) {
	m.mutationProbability = mutationProbability
	if m.mutationProbability < 0.0 || m.mutationProbability > 1.0 {
		panic("invalid mutationProbability")
	}
	m.crossoverProbability = crossoverProbability
	if m.crossoverProbability < 0.0 || m.crossoverProbability > 1.0 {
		panic("invalid crossoverProbability")
	}
}

// Evolve - one generation of population and sort for best fitness
func (m *Mep) Evolve() {

	if (m.variablesProbability + m.operatorsProbability + m.constantsProbability) != 1.0 {
		panic("probabilities must sum to 1.0")
	}

	for p := 0; p < m.numSubpopulation; p++ {

		offspring1 := m.randomChromosome(p)
		offspring2 := m.randomChromosome(p)

		for k := 0; k < m.subPopSize; k += 2 {

			// binary tournament
			r1 := m.tournamentSelection(p, 2)
			r2 := m.tournamentSelection(p, 2)
			m.copyChromosome(&m.pop[p][r1], &offspring1)
			m.copyChromosome(&m.pop[p][r2], &offspring2)
			// crossover
			if rand.Float64() < m.crossoverProbability {
				if m.crossoverType == OneCutPoint {
					m.oneCutPointCrossover(&m.pop[p][r1], &m.pop[p][r2], &offspring1, &offspring2)
				} else if m.crossoverType == Uniform {
					m.uniformCrossover(&m.pop[p][r1], &m.pop[p][r2], &offspring1, &offspring2)
				} else {
					panic("invalid crossover type")
				}
			}

			// mutatation
			m.mutation(&offspring1)
			m.eval(m.results[p], &offspring1)

			m.mutation(&offspring2)
			m.eval(m.results[p], &offspring2)

			// replace the worst in the population
			if offspring1.fitness < m.pop[p][m.subPopSize-1].fitness {
				m.copyChromosome(&offspring1, &m.pop[p][m.subPopSize-1])
			}
			if offspring2.fitness < m.pop[p][m.subPopSize-1].fitness {
				m.copyChromosome(&offspring2, &m.pop[p][m.subPopSize-1])
			}
		}

		// now copy one individual from one population to the next one.
		// the copied invidual will replace the worst in the next one (if is better)
		k := rand.Intn(m.subPopSize) // the individual to be copied

		// replace the worst in the next population (p + 1) - only if is better
		indexNextPop := (p + 1) % m.numSubpopulation // index of the next subpopulation (taken in circular order)

		if m.pop[p][k].fitness < m.pop[indexNextPop][m.subPopSize-1].fitness {
			m.copyChromosome(&m.pop[p][k], &m.pop[indexNextPop][m.subPopSize-1])
			sort.Sort(m.pop[indexNextPop])
		}

		sort.Sort(m.pop[p])

		if m.pop[p][0].fitness < m.pop[m.bestPop][0].fitness {
			m.bestPop = p
		}
	}
}

// Solve - Evolve until fitnessThreshold or numGens is reached. Returns generations and total time
func (m *Mep) Solve(numGens int, fitnessThreshold float64, showProgress bool) (int, time.Duration) {

	start := time.Now()
	gens := 0
	for gens < numGens {
		m.Evolve()
		if showProgress {
			m.PrintBest()
		}
		gens++
		if m.BestFitness() <= fitnessThreshold {
			break
		}
	}
	return gens, time.Since(start)
}

// BestFitness - return the best fitness of the population
func (m *Mep) BestFitness() float64 {
	return m.pop[m.bestPop][0].fitness
}

// BestExpr - return the best expression of the population
func (m *Mep) BestExpr() string {
	return m.parse("", m.pop[m.bestPop][0], m.pop[m.bestPop][0].bestIndex)
}

// Best - return the best fitness,expression of the population
func (m *Mep) Best() (float64, string) {
	return m.pop[m.bestPop][0].fitness, m.parse("", m.pop[m.bestPop][0], m.pop[m.bestPop][0].bestIndex)
}

// PrintBest - print the best member of the population
func (m *Mep) PrintBest() {
	exp := m.parse("", m.pop[m.bestPop][0], m.pop[m.bestPop][0].bestIndex)
	fmt.Printf("expr='%s' # fitness = %f\n", exp, m.pop[m.bestPop][0].fitness)
}

// PrintTestData - print the testdata
func (m *Mep) PrintTestData() {
	fmt.Println(strings.Join(m.td.Labels, ",") + ",target")
	for row := range m.td.Train {
		fmt.Print(strings.Replace(strings.Trim(fmt.Sprint(m.td.Train[row]), "[]"), " ", ",", -1))
		fmt.Println("," + fmt.Sprint(m.td.Target[row]))
	}
}

func (m *Mep) eval(results [][]float64, c *chromosome) {

	c.fitness = 1e+308
	c.bestIndex = -1

	// we keep intermediate values in a matrix because when an error occurs (like division by 0) we mutate that gene into a variables.
	// in such case it is faster to have all intermediate results until current gene, so that we don't have to recompute them again.

	for i := 0; i < m.codeLength; i++ { // read the chromosome from top to down

		isErrorCase := false
		switch c.program[i].op {
		case -1: // +
			for k := 0; k < m.numTraining; k++ {
				results[i][k] = results[c.program[i].adr1][k] + results[c.program[i].adr2][k]
			}
		case -2: // -
			for k := 0; k < m.numTraining; k++ {
				results[i][k] = results[c.program[i].adr1][k] - results[c.program[i].adr2][k]
			}
		case -3: // *
			for k := 0; k < m.numTraining; k++ {
				results[i][k] = results[c.program[i].adr1][k] * results[c.program[i].adr2][k]
			}
		case -4: //  /
			for k := 0; k < m.numTraining; k++ {
				if math.Abs(results[c.program[i].adr2][k]) < 1e-6 { // a small constant
					isErrorCase = true
				}
			}
			if isErrorCase { // an division by zero error occured !!!
				c.program[i].op = rand.Intn(m.numVariables) // the gene is mutated into a terminal
				for k := 0; k < m.numTraining; k++ {
					results[i][k] = m.td.Train[k][c.program[i].op]
				}
			} else { // normal execution....
				for k := 0; k < m.numTraining; k++ {
					results[i][k] = results[c.program[i].adr1][k] / results[c.program[i].adr2][k]
				}
			}
		case -5: //  sin
			for k := 0; k < m.numTraining; k++ {
				results[i][k] = math.Sin(results[c.program[i].adr1][k])
			}
		case -6: //  cos
			for k := 0; k < m.numTraining; k++ {
				results[i][k] = math.Cos(results[c.program[i].adr1][k])
			}
		case -7: //  tan
			for k := 0; k < m.numTraining; k++ {
				results[i][k] = math.Tan(results[c.program[i].adr1][k])
			}
		case -8: //  exp
			for k := 0; k < m.numTraining; k++ {
				results[i][k] = math.Exp(results[c.program[i].adr1][k])
			}
		case -9: //  log
			for k := 0; k < m.numTraining; k++ {
				results[i][k] = math.Log(results[c.program[i].adr1][k])
			}
		case -10: //  sqrt
			for k := 0; k < m.numTraining; k++ {
				results[i][k] = math.Sqrt(results[c.program[i].adr1][k])
			}
		case -11: //  abs
			for k := 0; k < m.numTraining; k++ {
				results[i][k] = math.Abs(results[c.program[i].adr1][k])
			}
		case -12: // max
			for k := 0; k < m.numTraining; k++ {
				if results[c.program[i].adr1][k] > results[c.program[i].adr2][k] {
					results[i][k] = results[c.program[i].adr1][k]
				} else {
					results[i][k] = results[c.program[i].adr2][k]
				}
			}
		case -13: // min
			for k := 0; k < m.numTraining; k++ {
				if results[c.program[i].adr1][k] < results[c.program[i].adr2][k] {
					results[i][k] = results[c.program[i].adr1][k]
				} else {
					results[i][k] = results[c.program[i].adr2][k]
				}
			}
		case -14: // ifgtz
			for k := 0; k < m.numTraining; k++ {
				if results[c.program[i].adr1][k] > 0.0 {
					results[i][k] = results[c.program[i].adr2][k]
				} else {
					results[i][k] = results[c.program[i].adr3][k]
				}
			}
		case -15: // ifltz
			for k := 0; k < m.numTraining; k++ {
				if results[c.program[i].adr1][k] < 0.0 {
					results[i][k] = results[c.program[i].adr2][k]
				} else {
					results[i][k] = results[c.program[i].adr3][k]
				}
			}
		case -16: // ifgt
			for k := 0; k < m.numTraining; k++ {
				if results[c.program[i].adr1][k] > results[c.program[i].adr2][k] {
					results[i][k] = results[c.program[i].adr3][k]
				} else {
					results[i][k] = results[c.program[i].adr4][k]
				}
			}
		case -17: // iflt
			for k := 0; k < m.numTraining; k++ {
				if results[c.program[i].adr1][k] < results[c.program[i].adr2][k] {
					results[i][k] = results[c.program[i].adr3][k]
				} else {
					results[i][k] = results[c.program[i].adr4][k]
				}
			}
		case -18: // ifbgt
			for k := 0; k < m.numTraining; k++ {
				if results[c.program[i].adr1][k] > results[c.program[i].adr2][k] {
					results[i][k] = 1.0
				} else {
					results[i][k] = 0.0
				}
			}
		case -19: // ifblt
			for k := 0; k < m.numTraining; k++ {
				if results[c.program[i].adr1][k] < results[c.program[i].adr2][k] {
					results[i][k] = 1.0
				} else {
					results[i][k] = 0.0
				}
			}
		case -20: // and
			for k := 0; k < m.numTraining; k++ {
				if results[c.program[i].adr1][k] > 0.0 && results[c.program[i].adr2][k] > 0.0 {
					results[i][k] = 1.0
				} else {
					results[i][k] = 0.0
				}
			}
		case -21: // or
			for k := 0; k < m.numTraining; k++ {
				if results[c.program[i].adr1][k] > 0.0 || results[c.program[i].adr2][k] > 0.0 {
					results[i][k] = 1.0
				} else {
					results[i][k] = 0.0
				}
			}
		case -22: // pow
			for k := 0; k < m.numTraining; k++ {
				results[i][k] = math.Pow(results[c.program[i].adr1][k], results[c.program[i].adr2][k])
			}
		case -23: // pow10
			for k := 0; k < m.numTraining; k++ {
				results[i][k] = math.Pow10(int(results[c.program[i].adr1][k]))
			}
		case -24: // log10
			for k := 0; k < m.numTraining; k++ {
				results[i][k] = math.Log10(results[c.program[i].adr1][k])
			}
		case -25: // log2
			for k := 0; k < m.numTraining; k++ {
				results[i][k] = math.Log2(results[c.program[i].adr1][k])
			}
		case -26: // floor
			for k := 0; k < m.numTraining; k++ {
				results[i][k] = math.Floor(results[c.program[i].adr1][k])
			}
		case -27: // ceil
			for k := 0; k < m.numTraining; k++ {
				results[i][k] = math.Ceil(results[c.program[i].adr1][k])
			}
		case -28: // inv
			for k := 0; k < m.numTraining; k++ {
				results[i][k] = 1.0 / results[c.program[i].adr1][k]
			}
		case -29: // square
			for k := 0; k < m.numTraining; k++ {
				results[i][k] = results[c.program[i].adr1][k] * results[c.program[i].adr1][k]
			}
		default: // a variable
			for k := 0; k < m.numTraining; k++ {
				if c.program[i].op < m.numVariables {
					results[i][k] = m.td.Train[k][c.program[i].op]
				} else {
					results[i][k] = c.constants[c.program[i].op-m.numVariables]
				}
			}
		}

		fitness := m.ff(results[i], m.td.Target)
		if c.fitness > fitness {
			c.fitness = fitness
			c.bestIndex = i
		}
	}
}

func (m *Mep) parse(exp string, individual chromosome, poz int) string {

	code := individual.program
	op := code[poz].op
	adr1 := code[poz].adr1
	adr2 := code[poz].adr2
	adr3 := code[poz].adr3
	adr4 := code[poz].adr4

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
	} else if op == -12 { // max
		exp += "max("
		exp = m.parse(exp, individual, adr1)
		exp += ","
		exp = m.parse(exp, individual, adr2)
		exp += ")"

	} else if op == -13 { // min
		exp += "min("
		exp = m.parse(exp, individual, adr1)
		exp += ","
		exp = m.parse(exp, individual, adr2)
		exp += ")"

	} else if op == -14 { // ifgtz
		exp += "iif("
		exp = m.parse(exp, individual, adr1)
		exp += ">0,"
		exp = m.parse(exp, individual, adr2)
		exp += ","
		exp = m.parse(exp, individual, adr3)
		exp += ")"

	} else if op == -15 { // ifltz
		exp += "iif("
		exp = m.parse(exp, individual, adr1)
		exp += "<0,"
		exp = m.parse(exp, individual, adr2)
		exp += ","
		exp = m.parse(exp, individual, adr3)
		exp += ")"

	} else if op == -16 { // ifgt
		exp += "iif("
		exp = m.parse(exp, individual, adr1)
		exp += ">"
		exp = m.parse(exp, individual, adr2)
		exp += ","
		exp = m.parse(exp, individual, adr3)
		exp += ","
		exp = m.parse(exp, individual, adr4)
		exp += ")"

	} else if op == -17 { // iflt
		exp += "iif("
		exp = m.parse(exp, individual, adr1)
		exp += "<"
		exp = m.parse(exp, individual, adr2)
		exp += ","
		exp = m.parse(exp, individual, adr3)
		exp += ","
		exp = m.parse(exp, individual, adr4)
		exp += ")"

	} else if op == -18 { // ifbgt
		exp += "iif("
		exp = m.parse(exp, individual, adr1)
		exp += ">"
		exp = m.parse(exp, individual, adr2)
		exp += ",1,0)"

	} else if op == -19 { // ifblt
		exp += "iif("
		exp = m.parse(exp, individual, adr1)
		exp += "<"
		exp = m.parse(exp, individual, adr2)
		exp += ",1,0)"

	} else if op == -20 { // and
		exp += "iif("
		exp = m.parse(exp, individual, adr1)
		exp += ">0 &&"
		exp = m.parse(exp, individual, adr2)
		exp += ">0,1,0)"

	} else if op == -21 { // or
		exp += "iif("
		exp = m.parse(exp, individual, adr1)
		exp += ">0 ||"
		exp = m.parse(exp, individual, adr2)
		exp += ">0,1,0)"

	} else if op == -22 { // pow
		exp += "pow("
		exp = m.parse(exp, individual, adr1)
		exp += ","
		exp = m.parse(exp, individual, adr2)
		exp += ")"

	} else if op == -23 { // pow10
		exp += "pow10("
		exp = m.parse(exp, individual, adr1)
		exp += ")"

	} else if op == -24 { // log10
		exp += "log10("
		exp = m.parse(exp, individual, adr1)
		exp += ")"

	} else if op == -25 { // log2
		exp += "log2("
		exp = m.parse(exp, individual, adr1)
		exp += ")"

	} else if op == -26 { // floor
		exp += "floor("
		exp = m.parse(exp, individual, adr1)
		exp += ")"

	} else if op == -27 { // ceil
		exp += "ceil("
		exp = m.parse(exp, individual, adr1)
		exp += ")"

	} else if op == -28 { // inv
		exp += "inv("
		exp = m.parse(exp, individual, adr1)
		exp += ")"

	} else if op == -29 { // square
		exp += "square("
		exp = m.parse(exp, individual, adr1)
		exp += ")"

	} else if op < m.numVariables {
		exp += m.td.Labels[op]

	} else {
		exp += fmt.Sprintf("(%f)", individual.constants[op-m.numVariables])
	}
	return exp
}

func (m *Mep) randomTerminal() int {
	var op int
	prob := rand.Float64() * (m.variablesProbability + m.constantsProbability)
	if prob <= m.variablesProbability {
		op = rand.Intn(m.numVariables)
	} else {
		op = m.numVariables + rand.Intn(m.numConstants)
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
			op = m.numVariables + rand.Intn(m.numConstants) // index of a constant
		}

	}
	return op
}

func (m *Mep) randomConstant() float64 {
	idx := rand.Intn(m.numConstants)
	if idx < len(m.fixedConstants) {
		return m.fixedConstants[idx]
	}
	return rand.Float64()*(m.randConstantsMax-m.randConstantsMin) + m.randConstantsMin
}

func (m *Mep) randomChromosome(subPop int) chromosome {

	a := chromosome{}
	a.program = make(program, m.codeLength)

	if m.numConstants > 0 {
		a.constants = make(constants, m.numConstants)
	}

	// generate random constants
	for c := 0; c < m.numConstants; c++ {
		a.constants[c] = m.randomConstant()
	}

	// on the first position we can have only a variable or a constant
	a.program[0].op = m.randomTerminal()

	// for all other genes we put either an operator, variable or constant
	for i := 1; i < m.codeLength; i++ {
		a.program[i].op = m.randomCode(i)
		a.program[i].adr1 = m.randomAdr(i)
		a.program[i].adr2 = m.randomAdr(i)
		a.program[i].adr3 = m.randomAdr(i)
		a.program[i].adr4 = m.randomAdr(i)
	}

	m.eval(m.results[subPop], &a)

	return a
}

func (m *Mep) randomPopulation() {

	// allocate results matrix

	m.results = make([][][]float64, m.numSubpopulation)
	for p := 0; p < m.numSubpopulation; p++ {
		m.results[p] = make([][]float64, m.codeLength)
		for i := 0; i < m.codeLength; i++ {
			m.results[p][i] = make([]float64, m.numTraining)
		}
	}

	// create new random population(s)
	m.pop = make(population, m.numSubpopulation)
	for p := 0; p < m.numSubpopulation; p++ {

		m.pop[p] = make(subPopulation, m.subPopSize)
		for i := 0; i < m.subPopSize; i++ {
			m.pop[p][i] = m.randomChromosome(p)
		}
		// sort by fitness ascending
		sort.Sort(m.pop[p])
	}

	// find the best individual
	m.bestPop = 0 // the index of the subpopulation containing the best invidual
	for p := 1; p < m.numSubpopulation; p++ {
		if m.pop[p][0].fitness < m.pop[m.bestPop][0].fitness {
			m.bestPop = p
		}
	}

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
	if m.numConstants > 0 {
		cuttingPoint = rand.Intn(m.numConstants)
		for i := 0; i < cuttingPoint; i++ {
			offspring1.constants[i] = parent1.constants[i]
			offspring2.constants[i] = parent2.constants[i]
		}
		for i := cuttingPoint; i < m.numConstants; i++ {
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
	for i := 0; i < m.numConstants; i++ {
		if (rand.Int() % 2) == 0 {
			offspring1.constants[i] = parent1.constants[i]
			offspring2.constants[i] = parent2.constants[i]
		} else {
			offspring1.constants[i] = parent2.constants[i]
			offspring2.constants[i] = parent1.constants[i]
		}
	}
}

func (m *Mep) tournamentSelection(subPop, tournamentSize int) int {

	p := rand.Intn(m.subPopSize)
	for i := 1; i < tournamentSize; i++ {
		r := rand.Intn(m.subPopSize)
		if m.pop[subPop][r].fitness < m.pop[subPop][p].fitness {
			p = r
		}
	}
	return p
}

func (m *Mep) mutation(aChromosome *chromosome) {

	// mutate each symbol with the given probability
	// first gene must be a variable or constant
	if rand.Float64() < m.mutationProbability {
		aChromosome.program[0].op = m.randomTerminal()
	}

	for i := 1; i < m.codeLength; i++ {

		if rand.Float64() < m.mutationProbability {
			aChromosome.program[i].op = m.randomCode(i)
		}

		if rand.Float64() < m.mutationProbability {
			aChromosome.program[i].adr1 = m.randomAdr(i)
		}

		if rand.Float64() < m.mutationProbability {
			aChromosome.program[i].adr2 = m.randomAdr(i)
		}

		if rand.Float64() < m.mutationProbability {
			aChromosome.program[i].adr3 = m.randomAdr(i)
		}

		if rand.Float64() < m.mutationProbability {
			aChromosome.program[i].adr4 = m.randomAdr(i)
		}
	}

	// mutate the constants
	for c := 0; c < m.numConstants; c++ {
		if rand.Float64() < m.mutationProbability {
			aChromosome.constants[c] = m.randomConstant()
		}
	}
}

func (m *Mep) copyChromosome(source, dest *chromosome) {

	for i := 0; i < m.codeLength; i++ {
		dest.program[i] = source.program[i]
	}

	for i := 0; i < m.numConstants; i++ {
		dest.constants[i] = source.constants[i]
	}
	dest.fitness = source.fitness
	dest.bestIndex = source.bestIndex
}
