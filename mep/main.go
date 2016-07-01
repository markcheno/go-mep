/*
Package is an implementation of the Multi Expression Programming algorithm

Copyright 2016 Mark Chenoweth
Licensed under terms of MIT license (see LICENSE)

Usage:
  mep -h | -help
  mep -v | -version
	mep -o | -oper
  mep [options] <filename>|testdata

Options:
  -h -help         			print help
  -v -version       		print version
	-o -oper              print operators
	-td                   print testdata
	-summary              print summary only
	-popsize=<subPopSize> sets sub-population size (default=100)
	-numpop=<numSubPop>   sets number of sub-populations (default=1)
	-code=<codeLen>       sets code length (default=50)
	-gens=<numGens>       sets number of generations to evolve
	-seed=<int>           sets random number seed (default=unixNano time)
	-fitness=<float>     	sets fitness threshold to stop evolving
	-mp=<mutationProb>    sets mutation probability
	-cp=<crossoverProb>		sets crossover probability
	-const=num,min,max		sets random constant parameters (-const=num,min,max[,(e|pi|<fixed>)])
	-enable=<op[,op]>     enables operators (comma separated list)
	-disable=<op[,op]>    disables operators (comma separated list)
*/
package main

import (
	"flag"
	"fmt"
	"github.com/markcheno/go-mep"
	"log"
	"math"
	"math/rand"
	"net/http"
	_ "net/http/pprof"
	"os"
	"strconv"
	"strings"
	"time"
)

const (
	version = "0.1"
)

type mepFlags struct {
	subPopSize           int
	numSubPops           int
	codeLen              int
	numGens              int
	seed                 int64
	fitnessThreshold     float64
	crossoverProbability float64
	mutationProbability  float64
	enable               string
	disable              string
	constants            string
	operators            bool
	version              bool
	td                   bool
	summary              bool
	regression           bool
}

func main() {

	go func() {
		log.Println(http.ListenAndServe("localhost:6060", nil))
	}()
	var flags mepFlags

	flag.IntVar(&flags.subPopSize, "popsize", 100, "sets sub-population size (default=100)")
	flag.IntVar(&flags.numSubPops, "numpop", 1, "sets number of sub-populations (default=1)")
	flag.IntVar(&flags.codeLen, "code", 50, "sets code length")
	flag.IntVar(&flags.numGens, "gens", 1000, "max number of generations to evolve")
	flag.Int64Var(&flags.seed, "seed", 0, "random seed (default unixNano time)")
	flag.Float64Var(&flags.fitnessThreshold, "fitness", 0.01, "fitness threshold")
	flag.Float64Var(&flags.mutationProbability, "mp", 0.1, "mutation probability")
	flag.Float64Var(&flags.crossoverProbability, "cp", 0.9, "crossover probability")
	flag.StringVar(&flags.enable, "enable", "", "list of operators to enable")
	flag.StringVar(&flags.disable, "disable", "", "list of operators to disable")
	flag.StringVar(&flags.constants, "const", "0,0,0", "constants: num,min,max[,(e|pi|<fixed>)]")
	flag.BoolVar(&flags.td, "td", false, "print testdata")
	flag.BoolVar(&flags.summary, "summary", false, "print summary only")
	flag.BoolVar(&flags.regression, "regression", true, "regression problem (classification=false)")
	flag.BoolVar(&flags.version, "v", false, "print version")
	flag.BoolVar(&flags.version, "version", false, "print version")
	flag.BoolVar(&flags.operators, "o", false, "print list of operators")
	flag.BoolVar(&flags.operators, "oper", false, "print list of operators")

	flag.Usage = func() {
		fmt.Println("Usage:")
		fmt.Println("  mep [options] <filename>|<testdata>")
		fmt.Println("Options:")
		flag.PrintDefaults()
		fmt.Println("Testdata:")
		fmt.Println("  pi")
		fmt.Println("  pythagorean")
		fmt.Println("  quarticpoly")
		fmt.Println("  rastigrin")
		fmt.Println("  dejong")
		fmt.Println("  schwefel")
		fmt.Println("  seqinduction")
		fmt.Println("  dropwave")
		fmt.Println("  michalewicz")
		fmt.Println("  schaffer")
		fmt.Println("  sixhump")
		fmt.Println("  simple1")
		fmt.Println("  simple2")
		fmt.Println("  simple3")
		fmt.Println("  kepler")
		fmt.Println("  booth")
	}

	flag.Parse()

	if flags.version {
		fmt.Println(version)
		os.Exit(0)
	}

	if flags.operators {
		m := *mep.New(mep.NewPiTest(1), mep.TotalErrorFF)
		fmt.Println(strings.Trim(strings.Join(m.Oper(true), ","), "[]"))
		os.Exit(0)
	}

	if flags.seed == 0 {
		rand.Seed(time.Now().UTC().UnixNano())
	} else {
		rand.Seed(flags.seed)
	}

	if len(flag.Args()) == 0 {
		flag.Usage()
		os.Exit(0)
	}

	filename := flag.Args()[0]

	var m *mep.Mep
	var td mep.TrainingData

	switch filename {
	case "pi":
		td = mep.NewPiTest(100)
	case "pythagorean":
		td = mep.NewPythagorean(100)
	case "quarticpoly":
		td = mep.NewQuarticPoly(100)
	case "rastigrin":
		td = mep.NewRastigrinF1(100)
	case "dejong":
		td = mep.NewDejongF1(100)
	case "schwefel":
		td = mep.NewSchwefel(100)
	case "seqinduction":
		td = mep.NewSequenceInduction(100)
	case "dropwave":
		td = mep.NewDropwave(100)
	case "michalewicz":
		td = mep.NewMichalewicz(100)
	case "schaffer":
		td = mep.NewSchafferF6(100)
	case "sixhump":
		td = mep.NewSixHump(100)
	case "simple1":
		td = mep.NewSimpleConstantRegression1(100)
	case "simple2":
		td = mep.NewSimpleConstantRegression2(100)
	case "simple3":
		td = mep.NewSimpleConstantRegression3(100)
	case "kepler":
		td = mep.NewKepler(100)
	case "booth":
		td = mep.NewBooth(50)
	default:
		td = mep.ReadTrainingData(filename, true, ",")
	}

	if flags.regression {
		m = mep.New(td, mep.TotalErrorFF)
	} else {
		m = mep.New(td, mep.ClassificationFF)
	}

	m.SetProb(flags.mutationProbability, flags.crossoverProbability)

	for _, op := range strings.Split(flags.enable, ",") {
		m.SetOper(op, true)
	}

	for _, op := range strings.Split(flags.disable, ",") {
		m.SetOper(op, false)
	}

	if flags.constants > "" {
		var fixed []float64
		tmp := strings.Split(flags.constants, ",")
		numRand, _ := strconv.ParseInt(tmp[0], 10, 64)
		minRand, _ := strconv.ParseFloat(tmp[1], 64)
		maxRand, _ := strconv.ParseFloat(tmp[2], 64)
		if len(tmp) > 3 {
			for c := 3; c < len(tmp); c++ {
				var tmp2 float64
				if tmp[c] == "e" {
					tmp2 = math.E
				} else if tmp[c] == "pi" {
					tmp2 = math.Pi
				} else {
					tmp2, _ = strconv.ParseFloat(tmp[c], 64)
				}
				fixed = append(fixed, tmp2)
			}
		}
		m.SetConst(fixed, int(numRand), maxRand, minRand)
	}

	m.SetPop(flags.subPopSize, flags.numSubPops, flags.codeLen)

	if flags.td {
		m.PrintTestData()
	}

	gens, elapsed := m.Solve(flags.numGens, flags.fitnessThreshold, !flags.summary)
	fmt.Printf("Elapsed time: %s\n", elapsed)
	fmt.Printf("Solution after %d generations:\n", gens)
	m.PrintBest()
	//m.PrintTestData()
}
