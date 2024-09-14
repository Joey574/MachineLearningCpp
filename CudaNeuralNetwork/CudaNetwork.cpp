#include "CudaNetwork.h"


void CudaNetwork::define(std::vector<int> dimesions) {
	this->m_dimensions = dimesions;
}

void CudaNetwork::compile() {
	// malloc size for network
}

void CudaNetwork::fit(Matrix x, Matrix y) {

	//// Allocate memory for the flattened matrix on the device
	//float* c_network;
	//float* c_results;

	//cudaMalloc(&_network, _network_size * sizeof(float));
	//cudaMalloc(&_results, _results_size * sizeof(float));

	//// Transfer data from host to device
	//cudaMemcpy(c_network, _network, _network_size * sizeof(float), cudaMemcpyHostToDevice);
	//cudaMemcpy(c_results, _results, _results_size * sizeof(float), cudaMemcpyHostToDevice);

	//// Set execution configuration parameters
	//int thr_per_blk = 256;
	//int blk_in_grid = (_network_size + thr_per_blk - 1) / thr_per_blk;

	//// Launch kernel
	//forward_prop <<<blk_in_grid, thr_per_blk>>> (c_network, c_results, _dimensions);

	//// Copy the updated data back to the host
	//cudaMemcpy(_results, c_results, _results_size * sizeof(float), cudaMemcpyDeviceToHost);

	//// Clean up
	//cudaFree(c_network);
	//cudaFree(c_results);

	int rows = 10;
	int columns = 10;

	float* h_a = (float*)malloc(rows * columns * sizeof(float));
	float* h_b = (float*)malloc(rows * columns * sizeof(float));
	float* h_c = (float*)malloc(rows * columns * sizeof(float));

	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < columns; c++) {
			h_a[r * columns + c] = r;
			std::cout << h_a[r * columns + c] << " ";
		} std::cout << "\n";
	} std::cout << "\n";

	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < columns; c++) {
			h_b[r * columns + c] = r;
			std::cout << h_b[r * columns + c] << " ";
		} std::cout << "\n";
	} std::cout << "\n";


	float* c_a;
	float* c_b;
	float* c_c;

	// Allocate memory on the gpu
	cudaMalloc(&c_a, rows * columns * sizeof(float));
	cudaMalloc(&c_b, rows * columns * sizeof(float));
	cudaMalloc(&c_c, rows * columns * sizeof(float));

	// Transfer data from host to device
	cudaMemcpy(c_a, h_a, rows * columns * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(c_b, h_b, rows * columns * sizeof(float), cudaMemcpyHostToDevice);

	// Set execution configuration parameters
	int thr_per_blk = 256;
	int blk_in_grid = ((rows * columns) + thr_per_blk - 1) / thr_per_blk;


	// Set execution configuration parameters
	// 
	// 16x16 threads per block (adjust as needed)
	dim3 threadsPerBlock(16, 16); 

	dim3 numBlocks((columns + threadsPerBlock.x - 1) / threadsPerBlock.x,
		(rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

	// Launch kernel
	element_add<<<blk_in_grid, thr_per_blk>>>(c_a, c_b, c_c, 10, 10);
	//element_add(c_a, c_b, c_c, 10, 10);

	// Copy the updated data back to the host
	cudaMemcpy(h_c, c_c, rows * columns * sizeof(float), cudaMemcpyDeviceToHost);

	// Clean up
	cudaFree(c_a);
	cudaFree(c_b);
	cudaFree(c_c);

	free(h_a);
	free(h_b);

	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < columns; c++) {
			std::cout << h_c[r * columns + c] << " ";
		} std::cout << "\n";
	}

	free(h_c);
}
