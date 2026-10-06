package broadcast

import (
	"slices"
	"testing"
)

func drain[T any](ch chan T) []T {
	var out []T
	for {
		select {
		case v, ok := <-ch:
			if !ok {
				return out
			}
			out = append(out, v)
		default:
			return out
		}
	}
}

func TestSendReachesEverySubscriberAndSkipsAFullOne(t *testing.T) {
	h := NewHub[int](1)
	a, b := h.Subscribe(), h.Subscribe()
	h.Send(1)
	h.Send(2) // both buffers are full: dropped, not blocking
	if got := drain(a); !slices.Equal(got, []int{1}) {
		t.Errorf("a = %v", got)
	}
	if got := drain(b); !slices.Equal(got, []int{1}) {
		t.Errorf("b = %v", got)
	}
}

func TestUnsubscribeClosesOnceAndStopsDelivery(t *testing.T) {
	h := NewHub[int](4)
	ch := h.Subscribe()
	h.Unsubscribe(ch)
	h.Unsubscribe(ch) // a second call must not panic on a closed channel
	h.Send(1)
	if _, ok := <-ch; ok {
		t.Error("channel still open after Unsubscribe")
	}
}

// A run's hub is closed when the run ends. Someone arriving after that must
// still read the final state they were seeded with, then the end.
func TestClosedHubGivesALateSubscriberItsSeedThenTheEnd(t *testing.T) {
	h := NewHub[string](2)
	h.Close()
	ch := h.Subscribe("final")
	if v, ok := <-ch; !ok || v != "final" {
		t.Fatalf("got %q, %v", v, ok)
	}
	if _, ok := <-ch; ok {
		t.Error("channel not closed")
	}
}

// EndSubscriptions ends the current followers but, unlike Close, leaves the
// hub serving the next job's.
func TestEndSubscriptionsLeavesTheHubOpen(t *testing.T) {
	h := NewHub[int](2)
	old := h.Subscribe()
	h.EndSubscriptions()
	if _, ok := <-old; ok {
		t.Error("old subscriber not ended")
	}
	next := h.Subscribe()
	h.Send(7)
	if got := drain(next); !slices.Equal(got, []int{7}) {
		t.Errorf("next = %v", got)
	}
}

func TestSubscribeLastReportsWhereThingsStand(t *testing.T) {
	h := NewHub[int](2)
	if _, _, ok := h.SubscribeLast(); ok {
		t.Error("a last value before anything was sent")
	}
	h.Send(3)
	_, last, ok := h.SubscribeLast()
	if !ok || last != 3 {
		t.Errorf("last = %d, %v", last, ok)
	}
}

func TestLogKeepsTheNewestLinesUpToItsCap(t *testing.T) {
	l := NewLog(3, 8)
	ch := l.Subscribe()
	for _, s := range []string{"a", "b", "c", "d"} {
		l.Append(s)
	}
	if got := l.Recent(0); !slices.Equal(got, []string{"b", "c", "d"}) {
		t.Errorf("Recent(0) = %v", got)
	}
	if got := l.Recent(2); !slices.Equal(got, []string{"c", "d"}) {
		t.Errorf("Recent(2) = %v", got)
	}
	if got := drain(ch); len(got) != 4 {
		t.Errorf("subscriber got %v, want every line", got)
	}
	l.Clear()
	if got := l.Recent(0); len(got) != 0 {
		t.Errorf("after Clear: %v", got)
	}
}
