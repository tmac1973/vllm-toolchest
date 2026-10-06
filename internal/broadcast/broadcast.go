// Package broadcast fans values out to subscribers: the live logs, progress
// and metrics the UI streams over SSE.
package broadcast

import "sync"

// Hub sends each value to every current subscriber. A subscriber that is not
// keeping up misses values rather than holding up the sender: what is sent
// is progress and log lines, and a stalled browser tab must not stall a
// download or a benchmark.
type Hub[T any] struct {
	mu      sync.Mutex
	size    int
	subs    map[chan T]struct{}
	last    T
	hasLast bool
	closed  bool
}

// NewHub returns a hub whose subscriber channels buffer size values.
func NewHub[T any](size int) *Hub[T] {
	return &Hub[T]{size: size, subs: map[chan T]struct{}{}}
}

// Subscribe returns a channel of the values sent from now on, preceded by
// seed. On a closed hub the channel holds seed and is already closed, so a
// late subscriber still reads the final state and then the end.
func (h *Hub[T]) Subscribe(seed ...T) chan T {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.subscribeLocked(seed)
}

// SubscribeLast is Subscribe that also returns the last value sent, if any,
// for a subscriber that joins midway and wants to show where things stand.
func (h *Hub[T]) SubscribeLast() (ch chan T, last T, ok bool) {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.subscribeLocked(nil), h.last, h.hasLast
}

func (h *Hub[T]) subscribeLocked(seed []T) chan T {
	ch := make(chan T, max(h.size, len(seed)))
	for _, v := range seed {
		ch <- v
	}
	if h.closed {
		close(ch)
	} else {
		h.subs[ch] = struct{}{}
	}
	return ch
}

// Unsubscribe removes ch and closes it. It is safe to call on a channel the
// hub has already closed.
func (h *Hub[T]) Unsubscribe(ch chan T) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if _, ok := h.subs[ch]; ok {
		delete(h.subs, ch)
		close(ch)
	}
}

// Send delivers v to every subscriber with room for it, and records it as
// the last value.
func (h *Hub[T]) Send(v T) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.last, h.hasLast = v, true
	for ch := range h.subs {
		select {
		case ch <- v:
		default:
		}
	}
}

// EndSubscriptions closes every current subscriber's channel, telling each
// that what it was following has ended. The hub stays open: later
// subscribers follow whatever comes next.
func (h *Hub[T]) EndSubscriptions() {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.endLocked()
}

// Close ends every subscription for good: later subscribers get a closed
// channel. For a hub that belongs to one run.
func (h *Hub[T]) Close() {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.endLocked()
	h.closed = true
}

func (h *Hub[T]) endLocked() {
	for ch := range h.subs {
		close(ch)
		delete(h.subs, ch)
	}
}

// Log is a capped buffer of log lines whose new lines are also broadcast.
type Log struct {
	*Hub[string]

	mu    sync.Mutex
	lines []string
	max   int
}

// NewLog keeps the last max lines and buffers size lines per subscriber.
func NewLog(max, size int) *Log {
	return &Log{Hub: NewHub[string](size), max: max}
}

// Append adds a line, dropping the oldest past the cap, and sends it.
func (l *Log) Append(line string) {
	l.mu.Lock()
	l.lines = append(l.lines, line)
	if over := len(l.lines) - l.max; over > 0 {
		// Copied down rather than resliced, so the dropped lines' backing
		// array is reused instead of kept alive behind the slice.
		l.lines = l.lines[:copy(l.lines, l.lines[over:])]
	}
	l.mu.Unlock()
	l.Send(line)
}

// Recent returns the last n lines, or all of them when n is out of range.
func (l *Log) Recent(n int) []string {
	l.mu.Lock()
	defer l.mu.Unlock()
	if n <= 0 || n > len(l.lines) {
		n = len(l.lines)
	}
	return append([]string(nil), l.lines[len(l.lines)-n:]...)
}

// Clear drops every buffered line. Subscribers are not affected.
func (l *Log) Clear() {
	l.mu.Lock()
	l.lines = l.lines[:0]
	l.mu.Unlock()
}
