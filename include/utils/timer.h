#ifndef INCLUDE_CORE_TIMER_H_
#define INCLUDE_CORE_TIMER_H_
#include <time.h>

/**
 * @file timer.h
 * Provide access to nano-second clock. Taken from piccolo codebase.
 */
namespace singa {

/**
 * Read CPU timestamp counter.
 */
static uint64_t rdtsc() {
	uint32_t hi, lo;
	__asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
	return (((uint64_t) hi) << 32) | ((uint64_t) lo);
}

/**
 * Get current time in nano second.
 */
inline double Now() {
	timespec tp;
	clock_gettime(CLOCK_MONOTONIC, &tp);
	return tp.tv_sec + 1e-9 * tp.tv_nsec;
}


/**
 * Wrapper class for measuring time in nano second and in CPU cycles.
 */
class Timer {
public:
	Timer() {
		reset();
	}

	void reset() {
		start_time_ = Now();
		start_cycle_ = rdtsc();
	}

	double elapsed() const {
		return Now() - start_time_;
	}

	uint64_t cycles_elapsed() const {
		return rdtsc() - start_cycle_;
	}

	// Rate at which an event occurs.
	double rate(int count) {
		return count / (Now() - start_time_);
	}

	double cycle_rate(int count) {
		return double(cycles_elapsed()) / count;
	}

private:
	double start_time_;
	uint64_t start_cycle_;
};

}  // namespace singa

#endif  // INCLUDE_CORE_TIMER_H_
