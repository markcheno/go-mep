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
