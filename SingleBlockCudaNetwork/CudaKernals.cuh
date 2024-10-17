#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// dot prods
__global__ void dot_prod(float* a, float* b, float* c, size_t a_r, size_t a_c, size_t b_r, size_t b_c);
__global__ void dot_prod_t_a(float* a, float* b, float* c, size_t a_r, size_t a_c, size_t b_r, size_t b_c);
__global__ void dot_prod_t_b(float* a, float* b, float* c, size_t a_r, size_t a_c, size_t b_r, size_t b_c);

// math util
__global__ void horizontal_add(float* a, float* b, size_t a_r, size_t a_c);
__global__ void horizontal_sum(float* a, float* b, size_t a_r, size_t a_c);

// weight and bias updates
__global__ void update_weights(float* weight, float* d_weight, float lr, size_t n);
__global__ void update_bias(float* bias, float* d_bias, float lr, size_t n);


// activations
__global__ void leaky_relu(float* x, float* y, size_t rows, size_t columns);

// derivatives
__global__ void leaky_relu_derivative(float* x, float* y, size_t r, size_t c);


// loss
__global__ void one_hot_loss(float* pred, float* loss, float* y, size_t rows, size_t columns);
__global__ void mae_loss(float* pred, float* loss, float* y, size_t rows, size_t columns);

// score
__global__ void accuracy_score(float* pred, float* y, int* correct, size_t rows, size_t columns);