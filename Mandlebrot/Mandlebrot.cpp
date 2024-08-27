#include <tuple>
#include <complex>
#include <atlimage.h>


#include "Matrix.h"

class Mandlebrot {
public:

    static enum class gradient_type {
        red_secondary, blue_secondary, green_secondary, random_secondary, no_secondary,
        cyclic, heatmap, sunset, ocean, fire, greyscale, diagonal
    };

	/// <summary>
	/// Returns a dataset such that each row is a different random point in the mandlebrot set
	/// </summary>
	/// <param name="size"></param>
	/// <param name="max_it"></param>
	/// <param name="fourier"></param>
	/// <param name="taylor"></param>
	/// <param name="chebyshev"></param>
	/// <param name="legendre"></param>
	/// <param name="laguarre"></param>
	/// <param name="lower_norm"></param>
	/// <param name="upper_norm"></param>
	/// <returns></returns>
	std::tuple<Matrix, Matrix> make_dataset(int size, int max_it, int fourier, int taylor, int chebyshev, int legendre, int laguarre, float lower_norm, float upper_norm) {

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
            y.matrix[i] = mandle;

        }

        x = x.extract_features(fourier, taylor, chebyshev, legendre,
            laguarre, lower_norm, upper_norm).Transpose();

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

    /// <summary>
    /// Interpolates between the mandlebrot set and provides a width * height dataset such that each row is a different pixel
    /// </summary>
    /// <param name="width"></param>
    /// <param name="height"></param>
    /// <param name="fourier"></param>
    /// <param name="taylor"></param>
    /// <param name="chebyshev"></param>
    /// <param name="legendre"></param>
    /// <param name="laguarre"></param>
    /// <param name="lower_norm"></param>
    /// <param name="upper_norm"></param>
    /// <returns></returns>
    Matrix create_image_features(int width, int height, int fourier, int taylor, int chebyshev, int legendre, int laguarre, float lower_norm, float upper_norm) {

        float scaleX = (std::abs(xMin - xMax)) / (width - 1);
        float scaleY = (std::abs(yMin - yMax)) / (height - 1);

        Matrix image(2, width * height);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                std::vector<float> val = { xMin + (float)x * scaleX, yMin + (float)y * scaleY };
                image.SetColumn(x + (y * width), val);
            }
        }

        image = image.extract_features(fourier, taylor, chebyshev, legendre, laguarre, lower_norm, upper_norm).Transpose();

        return image;
    }

     static std::vector<float> gradient(int x, int y, float value, float conf_thresh, gradient_type type) {

         switch (type) {
         case gradient_type::red_secondary:
             return { value * 255.0f, value > conf_thresh ? 255.0f : 0.0f, value > conf_thresh ? 255.0f : value };
             break;
         case gradient_type::blue_secondary:
             return {value > conf_thresh ? 255.0f : 0.0f, value > conf_thresh ? 255.0f : value, value * 255.0f };
             break;
         case gradient_type::green_secondary:
             return { value > conf_thresh ? 255.0f : 0.0f, value * 255.0f, value > conf_thresh ? 255.0f : value };
             break;
         case gradient_type::random_secondary: {
                  std::vector<float> c = {
                value > conf_thresh ? 255.0f : 0,
                value > conf_thresh ? 255.0f : 0,
                value > conf_thresh ? 255.0f : 0
             };

             c[rand() % 3] = value * 255.0f;
             return c;
         }
             break;
         case gradient_type::no_secondary:
             return std::vector<float>(3, (value * 255.0f));
             break;
         case gradient_type::cyclic:
             return {
                 127.5f * (1 + std::sin(value * 6.28318f)),
                 127.5f * (1 + std::sin(value * 6.28318f + 2.09439f)),
                 127.5f * (1 + std::sin(value * 6.28318f + 4.18879f))
             };
             break;
         case gradient_type::heatmap:
             return {
                 value > conf_thresh ? 255.0f : value * 255.0f,
                 value > conf_thresh ? 0.0f : (1.0f - value) * 255.0f,
                 value > conf_thresh ? 0.0f : (1.0f - value) * 255.0f
             };
             break;
         case gradient_type::sunset:
             return {
                 value * 255.0f,
                 (1.0f - value) * 64.0f,
                 (1.0f - value) * 128.0f
             };
             break;
         case gradient_type::ocean:
             return {
                 0.0f,
                 value * 255.0f,
                 (1.0f - value) * 255.0f
             };
             break;
         case gradient_type::fire:
             return {
                 value * 255.0f,
                 value * 128.0f,
                 0.0f
             };
             break;
         case gradient_type::greyscale:
             return {
                 (value * 127.5f) + 127.5f,
                 (value * 127.5f) + 127.5f,
                 (value * 127.5f) + 127.5f
             };
             break;
         case gradient_type::diagonal:
             return {
                 fmod((x + y * 255.0f), 256.0f) * value,
                 fmod((x - y * 255.0f + 256.0f), 256.0f) * 0.0f,
                 fmod((x * y * 255.0f), 256.0f) * value
             };
             break;
         }
         
     }

private:
    float xMin = -2.5f;
    float xMax = 1.0f;

    float yMin = -1.1f;
    float yMax = 1.1f;
};