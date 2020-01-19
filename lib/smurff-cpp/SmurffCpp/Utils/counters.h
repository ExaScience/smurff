/*
 * Copyright (c) 2014-2016, imec
 * All rights reserved.
 */

#pragma once

#ifdef PROFILING

#include <string>
#include <map>

#define COUNTER(name) Counter c(name)

struct Counter {
    Counter *parent;
    std::string name, fullname;
    double start, stop, diff; // wallclock time
    long long count;

    bool total_counter;

    Counter(std::string name);
    Counter(); // needed for std::map in TotalsCounter

    ~Counter();

    void operator+=(const Counter &other);

    std::string as_string(const Counter &total);
};

struct TotalsCounter {
    private:
        std::map<std::string, Counter> data;
        int procid;

    public:
        //c-tor starts PAPI
        TotalsCounter(int = 0);

        //prints results
        void print();

        Counter &operator[](const std::string &name) {
            return data[name];
        }
};

extern TotalsCounter perf_data;

inline void perf_data_print() {
    perf_data.print();
}

#else 

#define COUNTER(name) 

inline void perf_data_print() {}

#endif //BPMF_PROFILING

double tick();
