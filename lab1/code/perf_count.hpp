#ifndef PERFCOUNTER_H
#define PERFCOUNTER_H

#include <linux/perf_event.h> /* Definition of PERF_* constants */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/syscall.h> /* Definition of SYS_* constants */
#include <unistd.h>
#include <inttypes.h>
#include <functional>


// CPU has only 4 counter registers available for developer in linux
//
#define TOTAL_EVENTS 4

enum COUNTERS{
  NONE   = 0,
  INSTR  = 1,
  BRANCH = 2,
  CACHE  = 3,
  TLB    = 4
};

// Executes perf_event_open syscall and returns fd or -1
long perf_event_open(struct perf_event_attr *hw_event, pid_t pid, int cpu, int group_fd, unsigned long flags);

// Helper function to setup a perf event structure (perf_event_attr; see man perf_open_event)
void configure_event(struct perf_event_attr *pe, uint32_t type, uint64_t config);

void invalid_profile_mode();

COUNTERS configure_counters(perf_event_attr* pe);

void print_results(uint64_t *pe_val, COUNTERS mode);

/*
 * Запускает функцию F и измеряет аппаратные события, настроенные
 * функцией configure_counters().
 *
 * События открываются независимо друг от друга (не в группе). Это
 * позволяет измерять любые комбинации событий, поддерживаемые CPU:
 * аппаратные счётчики Intel имеют ограничения на совместное
 * планирование некоторых событий (например, LLC-события и события
 * L1D), из-за которых группа из четырёх событий может не запуститься.
 */
template <class Function, typename... Args>
auto run_with_counters(Function &&F, Args &&...ArgList)
{
  int      fd[TOTAL_EVENTS];
  bool     active[TOTAL_EVENTS];
  uint64_t pe_val[TOTAL_EVENTS];
  struct   perf_event_attr pe[TOTAL_EVENTS];

  for (int i = 0; i < TOTAL_EVENTS; i++) {
    fd[i] = -1;
    active[i] = false;
    pe_val[i] = 0;
  }

  COUNTERS mode = configure_counters(pe);

  if (mode != COUNTERS::NONE) {
    for (int i = 0; i < TOTAL_EVENTS; i++) {
      fd[i] = perf_event_open(&pe[i], 0, -1, -1, 0);
      if (fd[i] < 0) {
        fprintf(stderr, "warning: event %d is not supported, skipped\n", i);
        continue;
      }
      active[i] = true;
      ioctl(fd[i], PERF_EVENT_IOC_RESET, 0);
      ioctl(fd[i], PERF_EVENT_IOC_ENABLE, 0);
    }
  }

  std::invoke(F, ArgList...);

  if (mode != COUNTERS::NONE) {
    for (int i = 0; i < TOTAL_EVENTS; i++) {
      if (!active[i]) {
        continue;
      }
      ioctl(fd[i], PERF_EVENT_IOC_DISABLE, 0);
      uint64_t value = 0;
      if (read(fd[i], &value, sizeof(value)) == (ssize_t)sizeof(value)) {
        pe_val[i] = value;
      }
    }
  }

  print_results(pe_val, mode);

  for (int i = 0; i < TOTAL_EVENTS; i++) {
    if (fd[i] >= 0) {
      close(fd[i]);
    }
  }
}

#endif
