#include <iostream>
#include <random>

#include "Matrix.h"

int main()
{
    const int iterations = 10000;
    const int warmup = iterations / 100;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 2.0f);

    Matrix a(100, 100, Matrix::init::He);
    Matrix b(100, 100, Matrix::init::He);
    Matrix res(100, 100);
    std::vector<float> bias(100);

    std::chrono::duration<double, std::milli> time;

    for (int i = 0; i < bias.size(); i++) {
        bias[i] = dist(gen);
    }

    for (int i = 0; i < warmup; i++) {
        res = a.dot_product_add(b, bias);
    }
    auto s_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        res = a.dot_product_add(b, bias);
    }
    time = std::chrono::high_resolution_clock::now() - s_time;
    std::cout << "dot_product_add time: " << time.count() << " ms\n";

    for (int i = 0; i < warmup; i++) {
        res = a.dot_product(b) + bias;
    }
    s_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        res = a.dot_product(b) + bias;
    }
    time = std::chrono::high_resolution_clock::now() - s_time;
    std::cout << "dot_product + bias time: " << time.count() << " ms\n";
}