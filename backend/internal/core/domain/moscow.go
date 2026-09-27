package core_domain

import (
	"fmt"
	"time"
	_ "time/tzdata" // Windows and static binaries must not depend on the system zone database.
)

// Timezone is the calendar zone of forecasts: dates, hours and their bounds.
const Timezone = "Europe/Moscow"

// Moscow is the location of Timezone, loaded once with time.LoadLocation. The
// embedded time/tzdata is only the fallback when neither the system nor the
// GOROOT zone database exists.
var Moscow = mustLoadLocation(Timezone)

func mustLoadLocation(name string) *time.Location {
	location, err := time.LoadLocation(name)
	if err != nil {
		panic(fmt.Sprintf("load time zone %s: %v", name, err))
	}
	return location
}

// IsHourAligned reports whether t starts a whole hour in Moscow, the rule the
// schema enforces for forecast bounds. It checks Moscow time, not t's own
// location: a zone with a fractional offset may show a whole hour elsewhere.
func IsHourAligned(t time.Time) bool {
	local := t.In(Moscow)
	return local.Minute() == 0 && local.Second() == 0 && local.Nanosecond() == 0
}
