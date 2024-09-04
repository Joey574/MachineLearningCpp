#include "CudaNetwork.h"


void CudaNetwork::define(std::vector<int> dimesions) {
	this->_dimensions = dimesions;
}

void CudaNetwork::compile() {
	// malloc size for network
}

void CudaNetwork::fit(Matrix x, Matrix y) {

	// Allocate memory for the flattened matrix on the device
	float* c_network;
	float* c_results;

	cudaMalloc(&_network, _network_size * sizeof(float));
	cudaMalloc(&_results, _results_size * sizeof(float));

	// Transfer data from host to device
	cudaMemcpy(c_network, _network, _network_size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(c_results, _results, _results_size * sizeof(float), cudaMemcpyHostToDevice);

	// Set execution configuration parameters
	int thr_per_blk = 256;
	int blk_in_grid = (_network_size + thr_per_blk - 1) / thr_per_blk;

	// Launch kernel
	forward_prop <<<blk_in_grid, thr_per_blk>>> (c_network, c_results, _dimensions);

	// Copy the updated data back to the host
	cudaMemcpy(_results, c_results, _results_size * sizeof(float), cudaMemcpyDeviceToHost);

	// Clean up
	cudaFree(c_network);
	cudaFree(c_results);
}
