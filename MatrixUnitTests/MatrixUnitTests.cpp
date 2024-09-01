#include <iostream>
#include <iomanip>
#include <Windows.h>

#include "Matrix.h"

// Prototypes
bool run_test(bool (*operation)(Matrix a, Matrix b, double precision), std::string name, int iterations, int percent_checks, Matrix a, Matrix b, double precision);
bool matrix_basic_math_check(Matrix a, Matrix b, double precision);
bool matrix_trig_check(Matrix a, Matrix b, double precision);
bool matrix_activation_function_check(Matrix a, Matrix b, double precision);
bool matrix_transpose_check(Matrix a, Matrix b, double precision);
bool matrix_single_float_math_check(Matrix a, Matrix b, double precision);
bool matrix_vector_float_math_check(Matrix a, Matrix b, double precision);
bool matrix_derivative_function_check(Matrix a, Matrix b, double precision);
bool matrix_dot_product_check(Matrix a, Matrix b, double precision);
bool matrix_sums_check(Matrix a, Matrix b, double precision);
bool matrix_set_check(Matrix a, Matrix b, double precision);
bool matrix_segment_check(Matrix a, Matrix b, double precision);
bool matrix_equal_operator_check(Matrix a, Matrix b, double precision);

bool is_near(float a, float b, double precision);

const int RED_TEXT = 4;
const int GREEN_TEXT = 10;
const int WHITE_TEXT = 7;
const int BLUE_TEXT = 1;
const int YELLOW_TEXT = 6;
const int PURPLE_TEXT = 13;

int main()
{
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS);

    int iterations = 10000;
    int percent_checks = 10;
    int interval = iterations / percent_checks;

    const double precision = 0.000001;

    Matrix a = Matrix({
        {5.0f, 3.0f, 6.0f, 1.0f, 7.0f, 1.0f, 9.0f, 2.0f, 1.0f, 10.0f},
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f},
        {3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f},
        {5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f},
        {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f},
        {9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f},
        {11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f},
        {13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f},
        {15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f}
        });

    Matrix b = Matrix({
        {24.0f, 23.0f, 22.0f, 21.0f, 20.0f, 19.0f, 18.0f, 17.0f, 16.0f, 15.0f},
        {22.0f, 21.0f, 20.0f, 19.0f, 18.0f, 17.0f, 16.0f, 15.0f, 14.0f, 13.0f},
        {20.0f, 19.0f, 18.0f, 17.0f, 16.0f, 15.0f, 14.0f, 13.0f, 12.0f, 11.0f},
        {18.0f, 17.0f, 16.0f, 15.0f, 14.0f, 13.0f, 12.0f, 11.0f, 10.0f, 9.0f},
        {16.0f, 15.0f, 14.0f, 13.0f, 12.0f, 11.0f, 10.0f, 9.0f, 8.0f, 7.0f},
        {14.0f, 13.0f, 12.0f, 11.0f, 10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f},
        {12.0f, 11.0f, 10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f},
        {10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f},
        {5.0f, 3.0f, 6.0f, 1.0f, 7.0f, 1.0f, 9.0f, 2.0f, 1.0f, 10.0f}
        });

    Matrix x = Matrix({
        {1.0f, 0.0f},
        {0.0f, 1.0f},
        {0.5f, -0.5f},
        {0.25f, -0.25f},
        });

    std::cout << "x:\n" << "pad: " << (8 - ((x.RowCount * x.ColumnCount) % 8)) << "\nsize: " << x.RowCount << " :: " << x.ColumnCount << std::endl;
    std::cout << "\na:\n" << "pad: " << (8 - ((a.RowCount * a.ColumnCount) % 8)) << "\nsize: " << a.RowCount << " :: " << a.ColumnCount << std::endl;
    std::cout << "\nb:\n" << "pad: " << (8 - ((b.RowCount * b.ColumnCount) % 8)) << "\nsize: " << b.RowCount << " :: " << b.ColumnCount << std::endl;

    bool matrix_basic_math_test = run_test(&matrix_basic_math_check, "Matrix basic math test: ", iterations, percent_checks, a, b, precision);
    bool matrix_trig_test = run_test(&matrix_trig_check, "Matrix trig test: ", iterations, percent_checks, x, a, precision);
    bool matrix_activation_test = run_test(&matrix_activation_function_check, "Matrix activation function test: ", iterations, percent_checks, x, b, precision);
    bool matrix_transpose_test = run_test(&matrix_transpose_check, "Matrix transpose test: ", iterations, percent_checks, x, b, precision);
    bool matrix_single_float_test = run_test(&matrix_single_float_math_check, "Matrix scalar math test: ", iterations, percent_checks, a, b, precision);
    bool matrix_vector_float_test = run_test(&matrix_vector_float_math_check, "Matrix vector math test: ", iterations, percent_checks, a, b, precision);
    bool matrix_derivative_test = run_test(&matrix_derivative_function_check, "Matrix derivative function test: ", iterations, percent_checks, x, b, precision);
    bool matrix_dot_prod_test = run_test(&matrix_dot_product_check, "Matrix dot product test: ", iterations, percent_checks, a, b, precision);
    bool matrix_sum_test = run_test(&matrix_sums_check, "Matrix sums test: ", iterations, percent_checks, a, b, precision);
    bool matrix_set_test = run_test(&matrix_set_check, "Matrix set row/col test: ", iterations, percent_checks, a, b, precision);
    bool matrix_segment_test = run_test(&matrix_segment_check, "Matrix segment test: ", iterations, percent_checks, a, b, precision);
    bool matrix_equal_operator_test = run_test(&matrix_equal_operator_check, "Matrix equal operator test: ", iterations, percent_checks, a, b, precision);

   

    std::cout << "\nFinal Results:\n";
    SetConsoleTextAttribute(hConsole, matrix_basic_math_test ? GREEN_TEXT : RED_TEXT);
    std::cout << "\tMatrix basic math test: " << (matrix_basic_math_test ? "passed" : "failed") << std::endl;

    SetConsoleTextAttribute(hConsole, matrix_trig_test ? GREEN_TEXT : RED_TEXT);
    std::cout << "\tMatrix trig test: " << (matrix_trig_test ? "passed" : "failed") << std::endl;

    SetConsoleTextAttribute(hConsole, matrix_activation_test ? GREEN_TEXT : RED_TEXT);
    std::cout << "\tMatrix activation function test: " << (matrix_activation_test ? "passed" : "failed") << std::endl;

    SetConsoleTextAttribute(hConsole, matrix_transpose_test ? GREEN_TEXT : RED_TEXT);
    std::cout << "\tMatrix transpose test: " << (matrix_transpose_test ? "passed" : "failed") << std::endl;

    SetConsoleTextAttribute(hConsole, matrix_single_float_test ? GREEN_TEXT : RED_TEXT);
    std::cout << "\tMatrix scalar math test: " << (matrix_single_float_test ? "passed" : "failed") << std::endl;

    SetConsoleTextAttribute(hConsole, matrix_vector_float_test ? GREEN_TEXT : RED_TEXT);
    std::cout << "\tMatrix vector math test: " << (matrix_vector_float_test ? "passed" : "failed") << std::endl;

    SetConsoleTextAttribute(hConsole, matrix_derivative_test ? GREEN_TEXT : RED_TEXT);
    std::cout << "\tMatrix derivative function test: " << (matrix_derivative_test ? "passed" : "failed") << std::endl;

    SetConsoleTextAttribute(hConsole, matrix_dot_prod_test ? GREEN_TEXT : RED_TEXT);
    std::cout << "\tMatrix dot product test: " << (matrix_dot_prod_test ? "passed" : "failed") << std::endl;

    SetConsoleTextAttribute(hConsole, matrix_sum_test ? GREEN_TEXT : RED_TEXT);
    std::cout << "\tMatrix sums test: " << (matrix_sum_test ? "passed" : "failed") << std::endl;

    SetConsoleTextAttribute(hConsole, matrix_set_test ? GREEN_TEXT : RED_TEXT);
    std::cout << "\tMatrix set row/col test: " << (matrix_set_test ? "passed" : "failed") << std::endl;

    SetConsoleTextAttribute(hConsole, matrix_segment_test ? GREEN_TEXT : RED_TEXT);
    std::cout << "\tMatrix segment test: " << (matrix_segment_test ? "passed" : "failed") << std::endl;

    SetConsoleTextAttribute(hConsole, matrix_equal_operator_test ? GREEN_TEXT : RED_TEXT);
    std::cout << "\tMatrix equal operator test: " << (matrix_equal_operator_test ? "passed" : "failed") << std::endl;


    SetConsoleTextAttribute(hConsole, WHITE_TEXT);
    return 0;
}

bool run_test(bool (*operation)(Matrix a, Matrix b, double precision), std::string name, int iterations, int percent_checks, Matrix a, Matrix b, double precision) {
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);

    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point t_start;
    std::chrono::duration<double, std::milli> time;
    std::chrono::duration<double, std::milli> t_time;

    SetConsoleTextAttribute(hConsole, BLUE_TEXT);
    std::cout << std::endl << name << "starting\n";

    SetConsoleTextAttribute(hConsole, YELLOW_TEXT);
    int passed = 0;
    int current_percent = percent_checks;
    int interval = iterations / percent_checks;
    int current_interval = interval;
    t_start = std::chrono::high_resolution_clock::now();
    start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; i++) {

        // Every n% of iterations, stop and output percent complete, and time since last output
        if (i == current_interval - 1) {
            time = std::chrono::high_resolution_clock::now() - start;
            std::cout << "\t" << current_percent << "%" << " complete, time = " << time.count() << " ms\n";
            current_interval += interval;
            current_percent += percent_checks;
            start = std::chrono::high_resolution_clock::now();
        }

        passed += (*operation)(a, b, precision);
    }
    t_time = std::chrono::high_resolution_clock::now() - t_start;

    // Check if the number of times the test passed == tests done 
    SetConsoleTextAttribute(hConsole, PURPLE_TEXT);
    std::cout << "\tt_time = " << t_time.count() << " ms :: average = " << (t_time.count() / iterations) << " ms\n";
    SetConsoleTextAttribute(hConsole, (passed == iterations) ? GREEN_TEXT : RED_TEXT);
    std::cout << name << ((passed == iterations) ? "passed" : "failed") << std::endl;
    SetConsoleTextAttribute(hConsole, WHITE_TEXT);

    return (passed == iterations);
}

bool matrix_basic_math_check(Matrix a, Matrix b, double precision) {
    // Element-wise addition check between 2 matrices
    Matrix result = a + b;
    for (int r = 0; r < a.RowCount; r++) {
        for (int c = 0; c < a.ColumnCount; c++) {
            if (!is_near(result(r, c), a(r, c) + b(r, c), precision)) {
                return false;
            }
        }
    }

    std::cout << "pad: " << ((result.RowCount * result.ColumnCount) % 8) << std::endl;
    std::cout << "size: " << result.RowCount << " :: " << result.ColumnCount << std::endl;

    // Element-wise subtraction check between 2 matrices
    result = a - b;
    for (int r = 0; r < a.RowCount; r++) {
        for (int c = 0; c < a.ColumnCount; c++) {
            if (!is_near(result(r, c), a(r, c) - b(r, c), precision)) {
                return false;
            }
        }
    }

    // Element-wise multiplication check between 2 matrices
    result = a * b;
    for (int r = 0; r < a.RowCount; r++) {
        for (int c = 0; c < a.ColumnCount; c++) {
            if (!is_near(result(r, c), a(r, c) * b(r, c), precision)) {
                return false;
            }
        }
    }

    // Element-wise division check between 2 matrices
    result = a / b;
    for (int r = 0; r < a.RowCount; r++) {
        for (int c = 0; c < a.ColumnCount; c++) {
            if (!is_near(result(r, c), a(r, c) / b(r, c), precision)) {
                return false;
            }
        }
    }

    // Element-wise negative check with a
    result = a.Negative();
    for (int r = 0; r < a.RowCount; r++) {
        for (int c = 0; c < a.ColumnCount; c++) {
            if (result(r, c) != -a(r, c)) {
                return false;
            }
        }
    }

    // Element-wise abs check with result (currently hold neg a)
    result = result.Abs();
    for (int r = 0; r < a.RowCount; r++) {
        for (int c = 0; c < a.ColumnCount; c++) {
            if (result(r, c) != std::abs(a(r, c))) {
                return false;
            }
        }
    }

    // Element-wise log check with a
    result = a.Log();
    for (int r = 0; r < a.RowCount; r++) {
        for (int c = 0; c < a.ColumnCount; c++) {
            if (result(r, c) != std::log(a(r, c))) {
                return false;
            }
        }
    }
    return true;
}

bool matrix_trig_check(Matrix a, Matrix b, double precision) {
    // Element-wise sin check with a
    Matrix result = a.Sin();
    for (int r = 0; r < a.RowCount; r++) {
        for (int c = 0; c < a.ColumnCount; c++) {
            if (!is_near(result(r, c), std::sin(a(r, c)), precision)) {
                return false;
            }
        }
    }

    // Element-wise cos check with a
    result = a.Cos();
    for (int r = 0; r < a.RowCount; r++) {
        for (int c = 0; c < a.ColumnCount; c++) {
            if (!is_near(result(r, c), std::cos(a(r, c)), precision)) {
                return false;
            }
        }
    }

    // Element-wise acos check with a
    result = a.Acos();
    for (int r = 0; r < a.RowCount; r++) {
        for (int c = 0; c < a.ColumnCount; c++) {
            if (!is_near(result(r, c), std::acos(a(r, c)), precision)) {
                return false;
            }
        }
    }

    // Element-wise asin check with a
    result = a.Asin();
    for (int r = 0; r < a.RowCount; r++) {
        for (int c = 0; c < a.ColumnCount; c++) {
            if (!is_near(result(r, c), std::asin(a(r, c)), precision)) {
                return false;
            }
        }
    }

    return true;
}

bool matrix_activation_function_check(Matrix a, Matrix b, double precision) {
    // Sigmoid check
    Matrix result = a.Sigmoid();
    for (int r = 0; r < a.RowCount; r++) {
        for (int c = 0; c < a.ColumnCount; c++) {
            if (!is_near(result(r, c), 1.0f / (std::exp(-a(r, c)) + 1.0f), precision)) {
                return false;
            }
        }
    }

    // ReLU check
    result = a.ReLU();
    for (int r = 0; r < a.RowCount; r++) {
        for (int c = 0; c < a.ColumnCount; c++) {
            if (!is_near(result(r, c), (a(r, c) > 0.0f ? a(r, c) : 0.0f), precision)) {
                return false;
            }
        }
    }

    // LeakyReLU check
    result = a.LeakyReLU();
    for (int r = 0; r < a.RowCount; r++) {
        for (int c = 0; c < a.ColumnCount; c++) {
            if (!is_near(result(r, c), (a(r, c) > 0.0f ? a(r, c) : a(r, c) * 0.1f), precision)) {
                return false;
            }
        }
    }

    // ELU check
    result = a.ELU();
    for (int r = 0; r < a.RowCount; r++) {
        for (int c = 0; c < a.ColumnCount; c++) {
            if (!is_near(result(r, c), (a(r, c) > 0.0f ? a(r, c) : std::exp(a(r, c)) - 1), precision)) {
                return false;
            }
        }
    }

    // TanH check
    result = a.Tanh();
    for (int r = 0; r < a.RowCount; r++) {
        for (int c = 0; c < a.ColumnCount; c++) {
            if (!is_near(result(r, c), (std::exp(a(r, c)) - std::exp(-a(r, c))) / (std::exp(a(r, c)) + std::exp(-a(r, c))), precision)) {
                return false;
            }
        }
    }

    // Softplus check
    result = a.Softplus();
    for (int r = 0; r < a.RowCount; r++) {
        for (int c = 0; c < a.ColumnCount; c++) {
            if (!is_near(result(r, c), std::log(std::exp(a(r, c)) + 1.0f), precision)) {
                return false;
            }
        }
    }

    // SiLU check
    result = a.SiLU();
    for (int r = 0; r < a.RowCount; r++) {
        for (int c = 0; c < a.ColumnCount; c++) {
            if (!is_near(result(r, c), a(r, c) / (std::exp(-a(r, c)) + 1.0f), precision)) {
                return false;
            }
        }
    }

    return true;
}

bool matrix_transpose_check(Matrix a, Matrix b, double precision) {

    Matrix result = a.Transpose();
    for (int r = 0; r < a.RowCount; r++) {
        for (int c = 0; c < a.ColumnCount; c++) {
            if (result(c, r) != a(r, c)) {
                return false;
            }
        }
    }
    return true;
}

bool matrix_single_float_math_check(Matrix a, Matrix b, double precision) {
    Matrix result = a + 1;
    for (int r = 0; r < a.RowCount; r++) {
        for (int c = 0; c < a.ColumnCount; c++) {
            if (result(r, c) != a(r, c) + 1) {
                return false;
            }
        }
    }

    result = a - 2;
    for (int r = 0; r < a.RowCount; r++) {
        for (int c = 0; c < a.ColumnCount; c++) {
            if (result(r, c) != a(r, c) - 2) {
                return false;
            }
        }
    }

    result = a * 3;
    for (int r = 0; r < a.RowCount; r++) {
        for (int c = 0; c < a.ColumnCount; c++) {
            if (result(r, c) != a(r, c) * 3) {
                return false;
            }
        }
    }

    result = a / 4;
    for (int r = 0; r < a.RowCount; r++) {
        for (int c = 0; c < a.ColumnCount; c++) {
            if (result(r, c) != a(r, c) / 4) {
                return false;
            }
        }
    }

    return true;
}

bool matrix_vector_float_math_check(Matrix a, Matrix b, double precision) {
    std::vector<float> x = b.Row(0);
    std::vector<float> y = b.Column(0);

    Matrix result = a + x;
    for (int r = 0; r < a.RowCount; r++) {
        for (int c = 0; c < a.ColumnCount; c++) {
            if (result(r, c) != a(r, c) + x[c]) {
                return false;
            }
        }
    }

    result = a + y;
    for (int r = 0; r < a.RowCount; r++) {
        for (int c = 0; c < a.ColumnCount; c++) {
            if (result(r, c) != a(r, c) + y[r]) {
                return false;
            }
        }
    }


    return true;
}

bool matrix_derivative_function_check(Matrix a, Matrix b, double precision) {
    // Sigmoid deivative check
    Matrix result = a.SigmoidDerivative();
    for (int r = 0; r < a.RowCount; r++) {
        for (int c = 0; c < a.ColumnCount; c++) {
            if (!is_near(result(r, c), (1.0f / (std::exp(-a(r, c)) + 1.0f)) * (a(r, c) - (1.0f / (std::exp(-a(r, c)) + 1.0f))), precision)) {
                return false;
            }
        }
    }

    // ReLU deivative check
    result = a.ReLUDerivative();
    for (int r = 0; r < a.RowCount; r++) {
        for (int c = 0; c < a.ColumnCount; c++) {
            if (!is_near(result(r, c), a(r, c) > 0.0f ? 1.0f : 0.0f, precision)) {
                return false;
            }
        }
    }

    // LeakyReLU deivative check
    result = a.LeakyReLUDerivative();
    for (int r = 0; r < a.RowCount; r++) {
        for (int c = 0; c < a.ColumnCount; c++) {
            if (!is_near(result(r, c), (a(r, c) > 0.0f ? 1.0f : 0.1f), precision)) {
                return false;
            }
        }
    }

    // ELU deivative check
    result = a.ELUDerivative();
    for (int r = 0; r < a.RowCount; r++) {
        for (int c = 0; c < a.ColumnCount; c++) {
            if (!is_near(result(r, c), (a(r, c) > 0.0f ? 1.0f : std::exp(a(r, c))), precision)) {
                return false;
            }
        }
    }

    // TanH deivative check
    result = a.TanhDerivative();
    for (int r = 0; r < a.RowCount; r++) {
        for (int c = 0; c < a.ColumnCount; c++) {
            if (!is_near(result(r, c), 1.0f - std::pow((std::exp(a(r, c)) - std::exp(-a(r, c))) / (std::exp(a(r, c)) + std::exp(-a(r, c))), 2), precision)) {
                return false;
            }
        }
    }

    // Softplus deivative check
    result = a.SoftplusDerivative();
    for (int r = 0; r < a.RowCount; r++) {
        for (int c = 0; c < a.ColumnCount; c++) {
            if (!is_near(result(r, c), 1.0f / (std::exp(-a(r, c)) + 1.0f), precision)) {
                return false;
            }
        }
    }

    // SiLU deivative check
    result = a.SiLUDerivative();
    for (int r = 0; r < a.RowCount; r++) {
        for (int c = 0; c < a.ColumnCount; c++) {
            if (!is_near(result(r, c), std::exp(-a(r, c)) + (a(r, c) * std::exp(-a(r, c)) + 1.0f) / (std::pow(std::exp(-a(r, c)) + 1.0f, 2.0f)), precision)) {
                return false;
            }
        }
    }
    return true;
}

bool matrix_dot_product_check(Matrix a, Matrix b, double precision) {

    Matrix dot_a = Matrix({
       {1, 2, 3},
       {4, 5, 6}
        });

    Matrix dot_b = Matrix({
        {7, 8},
        {9, 10},
        {11, 12}
        });

    Matrix check_a = Matrix({
        {58, 64},
        {139, 154}
        });
    Matrix  result = dot_a.dot_product(dot_b);
    for (int r = 0; r < check_a.RowCount; r++) {
        for (int c = 0; c < check_a.ColumnCount; c++) {
            if (result(r, c) != check_a(r, c)) {
                return false;
            }
        }
    }

    dot_a = Matrix({
        {3, 4, 2}
        });

    dot_b = Matrix({
        {13, 9, 7, 15},
        {8, 7, 4, 6},
        {6, 4, 0, 3}
        });

    check_a = Matrix({
        {83, 63, 37, 75}
        });

    result = dot_a.dot_product(dot_b);
    for (int r = 0; r < check_a.RowCount; r++) {
        for (int c = 0; c < check_a.ColumnCount; c++) {
            if (result(r, c) != check_a(r, c)) {
                return false;
            }
        }
    }

    dot_a = Matrix({
        {1, 2, 3}
        });

    std::vector<std::vector<float>> data = {
        {4},
        {5},
        {6}
    };

    dot_b = Matrix(data);

    check_a = Matrix(1, 1, 32);

    result = dot_a.dot_product(dot_b);
    for (int r = 0; r < check_a.RowCount; r++) {
        for (int c = 0; c < check_a.ColumnCount; c++) {
            if (result(r, c) != check_a(r, c)) {
                return false;
            }
        }
    }

    check_a = Matrix({
        {4, 8, 12},
        {5, 10, 15},
        {6, 12, 18}
        });

    result = dot_b.dot_product(dot_a);
    for (int r = 0; r < check_a.RowCount; r++) {
        for (int c = 0; c < check_a.ColumnCount; c++) {
            if (result(r, c) != check_a(r, c)) {
                return false;
            }
        }
    }

    dot_a = Matrix({
        {-2.5f, 0},
        {-2.0f, 0},
        {-1.5f, 0},
        {-1.0f, 0},
        {-0.5f, 0},
        {0.0f, 0},
        {0.5f, 0},
        {1.0f, 0},
        {-2.5f, 0},
        {-2.0f, 0},
        {-1.5f, 0},
        {-1.0f, 0},
        {-0.5f, 0},
        {0.0f, 0},
        {0.5f, 0},
        {1.0f, 0},
        {-0.5f, 0},
        {0.0f, 0},
        {0.5f, 0},
        {1.0f, 0},
        });

    dot_b = Matrix({
        {1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4},
        {2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5},
        });

    check_a = Matrix({
        {-2.5f, -5.0f, -7.5f, -10.0f, -12.5f, -15.0f, -17.5f, -20.0f, -2.5f, -5.0f, -7.5f, -10.0f, -12.5f, -15.0f, -17.5f, -20.0f, -2.5f, -5.0f, -7.5f, -10.0f},
        {-2.0f, -4.0f, -6.0f, -8.0f, -10.0f, -12.0f, -14.0f, -16.0f, -2.0f, -4.0f, -6.0f, -8.0f, -10.0f, -12.0f, -14.0f, -16.0f, -2.0f, -4.0f, -6.0f, -8.0f},
        {-1.5f, -3.0f, -4.5f, -6.0f, -7.5f, -9.0f, -10.5f, -12.0f, -1.5f, -3.0f, -4.5f, -6.0f, -7.5f, -9.0f, -10.5, -12.0f, -1.5f, -3.0f, -4.5f, -6.0f},
        {-1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f, -7.0f, -8.0f, -1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f, -7.0f, -8.0f, -1.0f, -2.0f, -3.0f, -4.0f},
        {-0.5f, -1.0f, -1.5f, -2.0f, -2.5f, -3.0f, -3.5f, -4.0f, -0.5f, -1.0f, -1.5f, -2.0f, -2.5f, -3.0f, -3.5f, -4.0f, -0.5f, -1.0f, -1.5f, -2.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.0f, 0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.0f, 0.5f, 1.0f, 1.5f, 2.0f},
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 1.0f, 2.0f, 3.0f, 4.0f},
        {-2.5f, -5.0f, -7.5f, -10.0f, -12.5f, -15.0f, -17.5f, -20.0f, -2.5f, -5.0f, -7.5f, -10.0f, -12.5f, -15.0f, -17.5f, -20.0f, -2.5f, -5.0f, -7.5f, -10.0f},
        {-2.0f, -4.0f, -6.0f, -8.0f, -10.0f, -12.0f, -14.0f, -16.0f, -2.0f, -4.0f, -6.0f, -8.0f, -10.0f, -12.0f, -14.0f, -16.0f, -2.0f, -4.0f, -6.0f, -8.0f},
        {-1.5f, -3.0f, -4.5f, -6.0f, -7.5f, -9.0f, -10.5f, -12.0f, -1.5f, -3.0f, -4.5f, -6.0f, -7.5f, -9.0f, -10.5, -12.0f, -1.5f, -3.0f, -4.5f, -6.0f},
        {-1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f, -7.0f, -8.0f, -1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f, -7.0f, -8.0f, -1.0f, -2.0f, -3.0f, -4.0f},
        {-0.5f, -1.0f, -1.5f, -2.0f, -2.5f, -3.0f, -3.5f, -4.0f, -0.5f, -1.0f, -1.5f, -2.0f, -2.5f, -3.0f, -3.5f, -4.0f, -0.5f, -1.0f, -1.5f, -2.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.0f, 0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.0f, 0.5f, 1.0f, 1.5f, 2.0f},
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 1.0f, 2.0f, 3.0f, 4.0f},
        {-0.5f, -1.0f, -1.5f, -2.0f, -2.5f, -3.0f, -3.5f, -4.0f, -0.5f, -1.0f, -1.5f, -2.0f, -2.5f, -3.0f, -3.5f, -4.0f, -0.5f, -1.0f, -1.5f, -2.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.0f, 0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.0f, 0.5f, 1.0f, 1.5f, 2.0f},
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 1.0f, 2.0f, 3.0f, 4.0f}
        });

    result = dot_a.dot_product(dot_b);
    for (int r = 0; r < check_a.RowCount; r++) {
        for (int c = 0; c < check_a.ColumnCount; c++) {
            if (result(r, c) != check_a(r, c)) {
                return false;
            }
        }
    }

    dot_a = Matrix({
        {0.5f, 1.5f, 6.0f, 3.5f, 7.0f, 2.0f, 6.0f, 8.0f, 5.5f, 4.25f},
        {2.5f, 7.0f, 2.0f, 6.5f, 2.0f, 8.0f, 9.0f, 1.0f, 0.0f, 7.25f},
        {1.0f, 5.0f, 7.0f, 3.0f, 4.0f, 9.0f, 0.0f, 1.5f, 2.5f, 7.25f},
        {1.25f, 6.0f, 2.5f, 7.5f, 4.5f, 3.5f, 1.5f, 8.5f, 5.5f, 3.5f},
        {6.5f, 5.5f, 2.5f, 2.25f, 5.5f, 3.5f, 7.0f, 9.0f, 1.0f, 4.0f},
        {1.0f, 7.0f, 5.0f, 3.0f, 9.0f, 4.0f, 6.0f, 1.0f, 9.25f, 2.0f},
        {5.0f, 3.0f, 7.0f, 8.0f, 1.0f, 6.0f, 2.0f, 4.0f, 7.5f, 3.25f},
        {4.0f, 2.0f, 7.0f, 9.25f, 2.0f, 1.0f, 6.5f, 8.5f, 2.5f, 1.0f},
        {1.0f, 7.0f, 5.5f, 6.5f, 7.0f, 2.0f, 4.25f, 2.0f, 9.0f, 1.0f},
        {7.0f, 1.0f, 5.0f, 6.25f, 1.0f, 9.0f, 6.0f, 2.0f, 5.5f, 7.5f},
    });

    dot_b = Matrix({
        {6.0f, 2.0f, 2.0f, 9.5f, 2.25f, 8.0f, 1.0f, 5.5f, 2.5f, 4.0f},
        {1.5f, 2.0f, 2.0f, 7.0f, 5.5f, 3.25f, 1.0f, 8.0f, 9.0f, 1.0f},
        {6.25f, 6.0f, 2.0f, 1.0f, 8.0f, 1.0f, 2.0f, 5.0f, 9.0f, 2.5f},
        {1.5f, 6.5f, 7.0f, 4.0f, 5.5f, 8.25f, 1.5f, 8.0f, 7.0f, 2.0f},
        {1.0f, 6.0f, 2.5f, 7.25f, 8.0f, 2.0f, 5.0f, 3.5f, 1.0f, 6.5f},
        {1.0f, 5.5f, 4.0f, 7.25f, 3.5f, 5.5f, 2.5f, 8.5f, 5.0f, 4.0f},
        {1.5f, 2.0f, 7.0f, 2.0f, 5.5f, 8.25f, 2.5f, 2.0f, 1.0f, 6.5f},
        {9.0f, 2.25f, 2.0f, 7.5f, 7.0f, 1.0f, 4.5f, 1.0f, 6.0f, 9.0f},
        {6.5f, 2.0f, 7.0f, 6.25f, 2.5f, 4.5f, 5.0f, 8.0f, 1.0f, 4.0f,},
        {7.0f, 2.0f, 6.5f, 8.0f, 1.5f, 7.0f, 6.25f, 1.0f, 7.0f, 4.5f},
    });

    check_a = Matrix({
        {203.5f, 165.25f, 190.125f, 240.875f, 248.75f, 180.75f, 164.3125f, 182.5f, 199.5f, 231.125f, },
        {131.0f, 164.0f, 217.625f, 256.75f, 207.25f, 272.375f, 125.5625f, 233.0f, 240.5f, 180.125f, },
        {155.25f, 169.875f, 160.625f, 242.625f, 193.375f, 177.0f, 131.5625f, 223.75f, 242.75f, 150.625f },
        {190.375f, 164.625f, 186.0f, 273.5f, 232.0625f, 192.25f, 146.125f, 231.875f, 236.625f, 199.5f },
        {201.25f, 150.125f, 172.5f, 296.75f, 243.5f, 220.4375f, 144.625f, 194.25f, 217.0f, 240.5f },
        {157.375f, 178.25f, 207.25f, 263.0625f, 249.375f, 206.625f, 155.75f, 265.0f, 195.75f, 198.0f },
        {207.75f, 183.5f, 208.125f, 265.125f, 219.375f, 234.75f, 134.8125f, 276.25f, 245.75f, 180.625f },
        {197.125f, 170.75f, 186.25f, 218.125f, 249.375f, 211.6875f, 119.625f, 205.0f, 229.75f, 204.25f },
        {159.5f, 177.25f, 201.25f, 243.0f, 244.875f, 199.4375f, 139.625f, 266.0f, 209.75f, 177.375f },
        {209.375f, 184.625f, 241.5f, 297.375f, 207.125f, 296.0625f, 153.25f, 267.0f, 237.25f, 209.25f },
    });


    result = dot_a.dot_product(dot_b);
    for (int r = 0; r < check_a.RowCount; r++) {
        for (int c = 0; c < check_a.ColumnCount; c++) {
            if (!is_near(result(r, c), check_a(r, c), precision)) {
                std::cout << r << " :: " << c << std::endl;
                std::cout << result(r, c) << " :: " << check_a(r, c) << std::endl << std::endl;
                return false;
            }
        }
    }

    return true;
}

bool matrix_sums_check(Matrix a, Matrix b, double precision) {
    a = Matrix({
        {1, 2},
        {4, 3},
        {7, 9},
        });

    std::vector<float> row_sum = { 3, 7, 16 };
    std::vector<float> col_sum = { 12, 14 };

    std::vector<float> result = a.RowSums();
    for (int i = 0; i < result.size(); i++) {
        if (result[i] != row_sum[i]) {
            return false;
        }
    }
    a.Transpose();
    result = a.ColumnSums();
    for (int i = 0; i < result.size(); i++) {
        if (result[i] != col_sum[i]) {
            return false;
        }
    }


    return true;
}

bool matrix_set_check(Matrix a, Matrix b, double precision) {

    Matrix test = Matrix({
        {1, 2, 3, 4, 5},
        {2, 3, 4, 5, 6},
        {3, 4, 5, 6, 7},
        {4, 5, 6, 7, 8},
        {5, 6, 7, 8, 9},
        });

    std::vector<float> set = { 10, 10, 10, 10, 10 };

    test.SetRow(0, set);

    for (int i = 0; i < test.ColumnCount; i++) {
        if (test(0, i) != set[i]) {
            return false;
        }
    }

    test.SetColumn(0, set);

    for (int i = 0; i < test.RowCount; i++) {
        if (test(i, 0) != set[i]) {
            return false;
        }
    }

    return true;
}

bool matrix_segment_check(Matrix a, Matrix b, double precision) {

    Matrix result = a.SegmentR(0, 3);
    for (int r = 0; r < result.RowCount; r++) {
        for (int c = 0; c < a.ColumnCount; c++) {
            if (result(r, c) != a(r, c)) {
                return false;
            }
        }
    }

    result = a.SegmentC(0, 3);
    for (int r = 0; r < a.RowCount; r++) {
        for (int c = 0; c < result.ColumnCount; c++) {
            if (result(r, c) != a(r, c)) {
                return false;
            }
        }
    }

    return true;
}

bool matrix_equal_operator_check(Matrix a, Matrix b, double precision) {

    Matrix original = a;

    for (int r = 0; r < a.RowCount; r++) {
        for (int c = 0; c < a.ColumnCount; c++) {
            if (original(r, c) != a(r, c)) {
                return false;
            }
        }
    }

    original = a;
    a -= b;
    for (int r = 0; r < a.RowCount; r++) {
        for (int c = 0; c < a.ColumnCount; c++) {
            if (original(r, c) - b(r, c) != a(r, c)) {
                return false;
            }
        }
    }

    original = a;
    a += b;
    for (int r = 0; r < a.RowCount; r++) {
        for (int c = 0; c < a.ColumnCount; c++) {
            if (original(r, c) + b(r, c) != a(r, c)) {
                return false;
            }
        }
    }

    original = a;
    a *= b;
    for (int r = 0; r < a.RowCount; r++) {
        for (int c = 0; c < a.ColumnCount; c++) {
            if (original(r, c) * b(r, c) != a(r, c)) {
                return false;
            }
        }
    }

    original = a;
    a /= b;
    for (int r = 0; r < a.RowCount; r++) {
        for (int c = 0; c < a.ColumnCount; c++) {
            if (original(r, c) / b(r, c) != a(r, c)) {
                return false;
            }
        }
    }

    return true;
}

bool is_near(float a, float b, double precision) {
    return a <= b + precision && a >= b - precision;
}