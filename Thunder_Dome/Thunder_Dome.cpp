#include <iostream>
#include <random>

#include "Matrix.h"

void dot_product_combined_addition(const int iterations);

int main()
{
    const int iterations = 10000;

    dot_product_combined_addition(iterations);
}

void dot_product_combined_addition(const int iterations) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 2.0f);

    Matrix a(100, 100, Matrix::init::He);
    Matrix b(100, 100, Matrix::init::He);
    Matrix res(100, 100);
    std::vector<float> bias(100);

    std::chrono::duration<double, std::milli> time[2] = { std::chrono::duration<double, std::milli>(0.0), std::chrono::duration<double, std::milli>(0.0) };
    std::chrono::steady_clock::time_point s_time;

    for (int i = 0; i < bias.size(); i++) {
        bias[i] = dist(gen);
    }


    for (int i = 0; i < iterations; i++) {

        s_time = std::chrono::high_resolution_clock::now();
        a.dot_product_add(b, bias);
        time[1] += std::chrono::high_resolution_clock::now() - s_time;

        s_time = std::chrono::high_resolution_clock::now();
        a.dot_product(b) + bias;
        time[0] += std::chrono::high_resolution_clock::now() - s_time;

       
    }

    std::cout << "dot_product + bias time: " << time[0].count() << " ms\n";
    std::cout << "dot_product_add time: " << time[1].count() << " ms\n";
}