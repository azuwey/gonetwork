package matrix

import (
	"sync"
	"testing"
)

func TestNew(t *testing.T) {
	testCases := []struct {
		name          string
		rows, columns int
		values        []float64
		expectedError error
	}{
		{"Normal", 2, 2, []float64{0, 1, 2, 3}, nil},
		{"ErrNilValues", 2, 2, nil, ErrNilValues},
		{"ErrZeroRow", 0, 0, nil, ErrZeroRow},
		{"ErrZeroColumns", 1, 0, nil, ErrZeroColumns},
		{"ErrDataLength", 2, 2, []float64{}, ErrDataLength},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			m, err := New(tc.rows, tc.columns, tc.values)
			if tc.expectedError != nil {
				if err != tc.expectedError {
					t.Errorf("Expected error is %v, but got %v", tc.expectedError, err)
				}
			} else if err != nil {
				t.Errorf("Expected error is %v, but got %v", nil, err)
			} else if _, ok := m.(Matrix); !ok {
				t.Errorf("%v exptected to be a Matrix instance", m)
			}
		})
	}
}

func TestZero(t *testing.T) {
	testCases := []struct {
		name          string
		rows, columns int
		expectedError error
	}{
		{"Zero", 2, 2, nil},
		{"ErrZeroRow", 0, 0, ErrZeroRow},
		{"ErrZeroColumns", 1, 0, ErrZeroColumns},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			m, err := Zero(tc.rows, tc.columns)
			if tc.expectedError != nil {
				if err != tc.expectedError {
					t.Errorf("Expected error is %v, but got %v", tc.expectedError, err)
				}
			} else if err != nil {
				t.Errorf("Expected error is %v, but got %v", nil, err)
			} else if _, ok := m.(Matrix); !ok {
				t.Errorf("%v exptected to be a Matrix instance", m)
			}
		})
	}
}

func TestEmpty(t *testing.T) {
	t.Parallel()
	m := Empty()
	if _, ok := m.(Matrix); !ok {
		t.Errorf("%v exptected to be a Matrix instance", m)
	}
}

func TestCopy(t *testing.T) {
	t.Parallel()
	m, _ := Zero(10, 10)
	var w sync.WaitGroup
	for i := 0; i < 2; i++ {
		w.Add(1)
		go func() {
			n := m.Copy()
			defer w.Done()
			if _, ok := n.(Matrix); !ok {
				t.Errorf("%v exptected to be a Matrix instance", n)
			}
		}()
	}
	w.Wait()
}
