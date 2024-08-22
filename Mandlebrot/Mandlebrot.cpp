#include <tuple>
#include <complex>

#include "Matrix.h"

static class Mandlebrot {
public:

	std::tuple<Matrix, Matrix> make_dataset(int size, int max_it, int fourier, int taylor, int chebyshev, int legendre, int laguarre, float lower_norm, float upper_norm) {

        float xMin = -2.5f;
        float xMax = 1.0f;

        float yMin = -1.1f;
        float yMax = 1.1f;

        std::random_device rd;
        std::mt19937 gen(rd());

        std::uniform_real_distribution<float> xRand(xMin, xMax);
        std::uniform_real_distribution<float> yRand(yMin, yMax);

        Matrix x(2, size);
        Matrix y(size, 1);

        for (int i = 0; i < size; i++) {
            float m_x = xRand(gen);
            float m_y = yRand(gen);

            float mandle = in_mandlebrot(m_x, m_y, max_it);

            x.SetColumn(i, std::vector<float> {m_x, m_y});
            y(i, 0) = mandle;
        }

        x = x.extract_features(fourier, taylor, chebyshev, legendre,
            laguarre, lower_norm, upper_norm);

		return std::make_tuple(x, y);
	}

    float in_mandlebrot(float x, float y, int max_it) {
        std::complex<double> c(x, y);
        std::complex<double> z = 0;

        for (int i = 0; i < max_it; ++i) {
            z = z * z + c;
            if (std::abs(z) > 2) {
                return 1.0f - (1.0f / (((float)i / 50.0f) + 1.0f)); // Point is outside + smooth
            }
        }
        return 1.0f; // Point is inside the Mandelbrot set
    }



};