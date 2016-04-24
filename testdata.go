package mep

import (
	"fmt"
	"math"
	"math/rand"
)

// NewQuarticPoly -
func NewQuarticPoly(npoints int) TrainingData {

	labels := []string{"x"}
	training := make([][]float64, npoints)
	target := make([]float64, npoints)

	for i := 0; i < npoints; i++ {
		x := rand.Float64()
		if rand.Float64() < 0.5 {
			x = x * -1
		}
		result := math.Pow(x, 4) + math.Pow(x, 3) + math.Pow(x, 2) + x
		training[i] = []float64{x}
		target[i] = result
	}

	return TrainingData{training, target, labels}
}

// NewAckley -
func NewAckley(numTraining, numVariables int) TrainingData {

	t := TrainingData{}
	const xmin = -32.0
	const xmax = 32.0
	t.Labels = make([]string, numVariables)
	t.Train = make([][]float64, numTraining)
	t.Target = make([]float64, numTraining)
	for i := 0; i < numVariables; i++ {
		t.Labels[i] = fmt.Sprintf("x%d", i)
	}

	for i := 0; i < numTraining; i++ {
		t.Train[i] = make([]float64, numVariables)
		for j := 0; j < numVariables; j++ {
			t.Train[i][j] = rand.Float64()*(xmax-xmin) + xmin
		}
		const a = 20.0
		const b = 0.2
		const c = 2.0 * math.Pi
		s1 := 0.0
		s2 := 0.0
		for j := 0; j < numVariables; j++ {
			s1 = math.Pow(s1+t.Train[i][j], 2)
			s2 = s2 + math.Cos(c*t.Train[i][j])
		}
		t.Target[i] = -a*math.Exp(-b*math.Sqrt(1.0/float64(numTraining)*s1)) - math.Exp(1.0/float64(numTraining)*s2) + a + math.Exp(1.0)
	}

	return t
}

// NewPi -
func NewPi(numTraining, numVariables int) TrainingData {
	t := TrainingData{}
	const xmin = 1.0
	const xmax = 100.0
	t.Labels = make([]string, numVariables)
	t.Train = make([][]float64, numTraining)
	t.Target = make([]float64, numTraining)
	for i := 0; i < numVariables; i++ {
		t.Labels[i] = fmt.Sprintf("x%d", i)
	}

	for i := 0; i < numTraining; i++ {
		t.Train[i] = make([]float64, numVariables)
		for j := 0; j < numVariables; j++ {
			t.Train[i][j] = rand.Float64()*(xmax-xmin) + xmin
		}
		t.Target[i] = math.Pi
	}

	return t
}

// NewRastrigin -
func NewRastrigin(numTraining, numVariables int) TrainingData {

	t := TrainingData{}
	const xmin = -5.0
	const xmax = 5.0
	t.Labels = make([]string, numVariables)
	t.Train = make([][]float64, numTraining)
	t.Target = make([]float64, numTraining)
	for i := 0; i < numVariables; i++ {
		t.Labels[i] = fmt.Sprintf("x%d", i)
	}

	n := float64(numTraining)
	s := 0.0
	for i := 0; i < numTraining; i++ {
		t.Train[i] = make([]float64, numVariables)
		for j := 0; j < numVariables; j++ {
			t.Train[i][j] = rand.Float64()*(xmax-xmin) + xmin
			s = s + (math.Pow(t.Train[i][j], 2) - 10*math.Cos(2*math.Pi*t.Train[i][j]))
			s = 10*n + s
			t.Target[i] = s
		}
	}
	return t
}

/*
func (t *Test)  TestFunction(int num_training_data, int num_variables, float xmin, float xmax) {
    t.xmin = xmin;
    t.xmax = xmax;
    labels = new String[num_variables];
    for (int i=0; i<num_variables; i++ ) {
      labels[i] = new String("x"+i);
    }
    train = new float[num_training_data][num_variables];
    target = new float[num_training_data];
    for (int i=0; i<num_training_data; i++ ) {
      for (int j=0; j<num_variables; j++ ) {
        //train[i][j] = random(xmin,xmax);
        train[i][j] = rnd.nextFloat()*(xmax-xmin)+xmin;
      }
      target[i] = evaluate(train[i]);
    }
  }
}

abstract class TestFunction extends TrainingData {
  float xmin = 0.0f;
  float xmax = 0.0f;


  abstract String name();
  abstract float evaluate(float[] terms);
}

class Ackley extends TestFunction {
  Ackley() {
    super(50, 5, -32.0f, 32.0f);
  }
  Ackley(int num_training_data, int num_variables) {
    super(num_training_data, num_variables, -32.0f, 32.0f);
  }
  float evaluate(float[] x) {
    float n = x.length;
    float a = 20.0f;
    float b = 0.2f;
    float c = 2f*(float)Math.PI;
    float s1 = 0.0f;
    float s2 = 0.0f;
    for (int i=0; i<x.length; i++) {
      s1 = (float)Math.pow(s1+x[i], 2 );
      s2 = s2+(float)Math.cos(c*x[i]);
    }
    return -a*(float)Math.exp(-b*(float)Math.sqrt(1/n*s1))-(float)Math.exp(1/n*s2)+a+(float)Math.exp(1.0);
  }
  String name() {
    return "Ackley";
  }
}

class Rosenbrock extends TestFunction {
  Rosenbrock() {
    super(50, 5, -30.0f, 30.0f);
  }
  Rosenbrock(int num_training_data, int num_variables) {
    super(num_training_data, num_variables, -30.0f, 30.0f);
  }
  float evaluate(float[] x) {
    float n = x.length;
    float sum = 0f;
    for (int i=0; i<x.length-1; i++) {
      sum = sum + 100 * (float)Math.pow(((float)Math.pow(x[i], 2) - x[i+1]), 2) + (float)Math.pow(x[i]-1, 2);
    }
    return sum;
  }
  String name() {
    return "Rosenbrock";
  }
}

class Rastrigin extends TestFunction {
  Rastrigin() {
    super(50, 5, -5.0f, 5.0f);
  }
  Rastrigin(int num_training_data, int num_variables) {
    super(num_training_data, num_variables, -5.0f, 5.0f);
  }
  float evaluate(float[] x) {
    float n = x.length;
    float s = 0f;
    for (int i=0; i<x.length; i++) {
      s = s + ( (float)Math.pow(x[i], 2) - 10 * (float)Math.cos( 2*Math.PI*x[i]));
    }
    return 10*n+s;
  }
  String name() {
    return "Rastrigin";
  }
}

class Schwefel extends TestFunction {
  Schwefel() {
    super(50, 5, -500.0f, 500.0f);
  }
  Schwefel(int num_training_data, int num_variables) {
    super(num_training_data, num_variables, -500.0f, 500.0f);
  }
  float evaluate(float[] x) {
    float n = x.length;
    float sum = 0f;
    for (int i=0; i<x.length; i++) {
      sum = sum - x[i]*(float)Math.sin((float)Math.sqrt((float)Math.abs(x[i])));
    }
    return 418.9829f * n + sum;
  }
  String name() {
    return "Schwefel";
  }
}

class Pythagorean extends TestFunction {
  Pythagorean() {
    super(20, 2, 5.0f, 50.0f);
  }
  Pythagorean(int num_training_data) {
    super(num_training_data, 2, 5.0f, 50.0f);
  }
  float evaluate(float[] x) {
    return (float)Math.sqrt( (x[0]*x[0]) + (x[1]*x[1]) );
  }
  String name() {
    return "Pythagorean";
  }
}

class SymbolicRegressionTD extends TestFunction {
  SymbolicRegressionTD() {
    super(50, 1, 5.0f, 50.0f);
  }
  SymbolicRegressionTD(int num_training_data) {
    super(num_training_data, 1, 5.0f, 50.0f);
  }
  float evaluate(float[] x) {
    return (float)(Math.pow(x[0], 4)+Math.pow(x[0], 3)+Math.pow(x[0], 2)+x[0]);
  }
  String name() {
    return "SymbolicRegression";
  }
}

class SequenceInduction extends TestFunction {
  SequenceInduction() {
    super(50, 1, 5.0f, 50.0f);
  }
  SequenceInduction(int num_training_data) {
    super(num_training_data, 1, 5.0f, 50.0f);
  }
  float evaluate(float[] x) {
    return (float)((5.0*Math.pow(x[0], 4)) + (4.0*Math.pow(x[0], 3)) + (3.0*Math.pow(x[0], 2)) + (2.0*x[0]) + 1.0);
  }
  String name() {
    return "SequenceInduction";
  }
}

class SimpleConstantRegression1 extends TestFunction {
  SimpleConstantRegression1() {
    super(100, 1, 1.0f, 20.0f);
  }
  SimpleConstantRegression1(int num_training_data) {
    super(num_training_data, 1, 1.0f, 20.0f);
  }
  float evaluate(float[] x) {
    return (float)(Math.pow(x[0],3)-0.3*Math.pow(x[0],2)-0.4*x[0]-0.6);
  }
  String name() {
    return "SimpleConstantRegression1";
  }
}

class SimpleConstantRegression2 extends TestFunction {
  SimpleConstantRegression2() {
    super(50, 1, 1.0f, 20.0f);
  }
  SimpleConstantRegression2(int num_training_data) {
    super(num_training_data, 1, 1.0f, 20.0f);
  }
  float evaluate(float[] x) {
    return (float)(x[0]*x[0]+Math.PI);
  }
  String name() {
    return "SimpleConstantRegression2";
  }
}

class SimpleConstantRegression3 extends TestFunction {
  SimpleConstantRegression3() {
    super(25, 1, 1.0f, 20.0f);
  }
  SimpleConstantRegression3(int num_training_data) {
    super(num_training_data, 1, 1.0f, 100.0f);
  }
  float evaluate(float[] x) {
    return (float)((Math.E*x[0]*x[0])+(Math.PI*x[0]));
  }
  String name() {
    return "SimpleConstantRegression3";
  }
}

class PiTest extends TestFunction {
  PiTest() {
    super(50, 1, 1.0f, 100.0f);
  }
  PiTest(int num_training_data) {
    super(num_training_data, 1, 1.0f, 100.0f);
  }
  float evaluate(float[] x) {
    return (float)Math.PI;
  }
  String name() {
    return "PiTest";
  }
}
*/

// TODO: integer constant test: y = 4*pow(x,4)+3*pow(x,3)+2*pow(x,2)+x

// TODO: real constant test: y = 4.251*pow(x,2)+log(x*x)+7.243*pow(e,x)

// TODO: real constant test: 2.718*x*x + 3.1416*x
