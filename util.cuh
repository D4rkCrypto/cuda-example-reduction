#pragma once

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <utility>
#include <vector>
#include <numeric>
#include <chrono>
#include <algorithm>
#include <random>
#include <ctime>

#include <boost/format.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

#define N_TESTS 10000

// A function to return a seeded random number generator.
inline std::mt19937& generator() {
    // the generator will only be seeded once (per thread) since it's static
    static thread_local std::mt19937 gen(std::random_device{}());
    return gen;
}

// A function to generate integers in the range [min, max]
inline int my_rand_int(int min, int max) {
    std::uniform_int_distribution<int> dist(min, max);
    return dist(generator());
}

inline void print_timer_banner()
{
    // print header
    auto fmt = boost::format("%20s%10s%20s%20s%20s") % "function" % "trials" % "mean (us)" % "median (us)" % "std. dev.";
    std::cout << fmt << std::endl;
}

class CUDATimer
{
public:
    explicit CUDATimer(std::string func_name)
    {
        func_name_ = std::move(func_name);

        cudaEventCreate(&start_event_);
        cudaEventCreate(&stop_event_);
    }

    ~CUDATimer()
    {
        cudaEventDestroy(start_event_);
        cudaEventDestroy(stop_event_);
        auto n_trials = time_.size();
        auto mean_time = mean(time_);
        auto median_time = median(time_);
        auto stddev = std_dev(time_);
        auto fmt = boost::format("%20s%10i%20.3f%20.3f%20.6f")
                   % func_name_ % n_trials % mean_time % median_time % stddev;
        std::cout << fmt << std::endl;
    }

    inline void start() const {
        cudaEventRecord(start_event_);
    }

    inline void stop() {
        cudaEventRecord(stop_event_);
        cudaEventSynchronize(stop_event_);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start_event_, stop_event_);
        time_.push_back(milliseconds);
    }

private:
    std::string func_name_;
    cudaEvent_t start_event_{}, stop_event_{};
    std::vector<float> time_;

    static float mean(std::vector<float> const& v){
        if(v.empty())
            return 0;

        auto const count = static_cast<float>(v.size());
        return std::reduce(v.begin(), v.end()) / count * 1000;
    }

    static float median(std::vector<float> v)
    {
        size_t size = v.size();

        if (size == 0)
            return 0;

        sort(v.begin(), v.end());
        return v[size / 2] * 1000;
    }

    static double std_dev(std::vector<float> const& v)
    {
        if (v.empty())
            return 0;

        auto const count = static_cast<float>(v.size());
        float mean = std::reduce(v.begin(), v.end()) / count;

        std::vector<double> diff(v.size());

        std::transform(v.begin(), v.end(), diff.begin(), [mean](double x) { return x - mean; });
        double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
        return std::sqrt(sq_sum / count);
    }
};
