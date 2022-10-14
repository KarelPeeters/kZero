#pragma once

#include "util.cu"

// Implementation of https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
struct Welford {
    int count = 0;
    float mean = 0.0;
    float m2 = 0.0;

    __device__ Welford() {}

    __device__ Welford(int count, float mean, float m2) : count(count), mean(mean), m2(m2) {}

    __device__ void append(float value) {
        int count = this->count + 1;
        float delta = value - this->mean;
        float mean = this->mean + delta / count;
        float m2 = this->m2 + delta * (value - mean);

        this->count = count;
        this->mean = mean;
        this->m2 = m2;
    }

    __device__ Welford combine(Welford other) {
        Welford a = *this;
        Welford b = other;

        int count = a.count + b.count;
        float delta = b.mean - a.mean;

        float div_count = (count == 0) ? 1.0 : (float) count;
        float mean = a.mean + delta * (float) b.count / div_count;
        float m2 = a.m2 + b.m2 + delta * delta * (float) (a.count * b.count) / div_count;

        return Welford(count, mean, m2);
    }

    __device__ float final_mean() {
        return (this->count == 0) ? NaN : this->mean;
    }

    __device__ float final_variance() {
        return this->m2 / this->count;
    }
};
