package main

import (
  "github.com/markcheno/go-mep"
)

func main() {
 
	params mep.Params

	params.popSize = 100 // the number of individuals in population  (must be an even number!)
	params.codeLength = 50
	params.numGenerations = 100	// the number of generations
	params.mutationProbability = 0.1 // mutation probability
	params.crossoverProbability = 0.9 // crossover probability
	params.variablesProbability = 0.4
	params.operatorsProbability = 0.5
	params.constantsProbability = 1 - params.variablesProbability - params.operatorsProbability // sum of variables_prob + operators_prob + constants_prob MUST BE 1 !
	params.numConstants = 3 // use 3 constants from -1 ... +1 interval
	params.constantsMin = -1
	params.constantsMax = 1
	params.problemType = 0 //0 - regression, 1 - classification; DONT FORGET TO SET IT
	params.classificationThreshold = 0 // only for classification problems

  trainingData,target := mep.ReadTrainingData("data/building.txt")

	mep.StartSteadyState(params, trainingData, target)
}