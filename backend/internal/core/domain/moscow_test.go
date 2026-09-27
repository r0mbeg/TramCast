package core_domain

import (
	"testing"
	"time"
)

func TestMoscowLocation(t *testing.T) {
	if Moscow.String() != Timezone {
		t.Fatalf("Moscow = %q, want %q", Moscow, Timezone)
	}
	// The contract example: the MVP horizon starts at 21:00 UTC on 31 October.
	got := time.Date(2025, 11, 1, 0, 0, 0, 0, Moscow).UTC()
	if want := time.Date(2025, 10, 31, 21, 0, 0, 0, time.UTC); !got.Equal(want) {
		t.Fatalf("Moscow midnight = %s, want %s", got, want)
	}
}

func TestIsHourAligned(t *testing.T) {
	india := time.FixedZone("UTC+05:30", 5*60*60+30*60)
	start := time.Date(2025, 11, 1, 0, 0, 0, 0, Moscow)
	for _, tt := range []struct {
		name string
		t    time.Time
		want bool
	}{
		{"Moscow midnight", start, true},
		{"same instant in UTC", start.UTC(), true},
		{"last hour of the MVP horizon", time.Date(2025, 12, 31, 23, 0, 0, 0, Moscow), true},
		{"half past", start.Add(30 * time.Minute), false},
		{"one second", start.Add(time.Second), false},
		{"one nanosecond", start.Add(time.Nanosecond), false},
		// 05:30 in India is 03:00 in Moscow, and 06:00 there is 03:30 here.
		{"whole Moscow hour in a fractional zone", time.Date(2025, 11, 1, 5, 30, 0, 0, india), true},
		{"whole hour only in a fractional zone", time.Date(2025, 11, 1, 6, 0, 0, 0, india), false},
	} {
		if got := IsHourAligned(tt.t); got != tt.want {
			t.Errorf("%s: IsHourAligned(%s) = %v, want %v", tt.name, tt.t, got, tt.want)
		}
	}
}
