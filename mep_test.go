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
	mep := New(NewPi(50, 1), TotalErrorFF, popSize, codeSize)
	equals(t, mep.popSize, popSize)
	equals(t, mep.codeLength, codeSize)
	equals(t, mep.popSize, popSize)
	equals(t, mep.MutationProbability, 0.1)
	equals(t, mep.CrossoverProbability, 0.9)
	equals(t, len(mep.pop), mep.popSize)
}
