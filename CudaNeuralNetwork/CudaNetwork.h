#pragma once
#include <vector>

#include "Matrix.h"
#include "cuda_network.cuh"


class CudaNetwork
{
public:

	void define(std::vector<int> dimensions);
	void compile();
	void fit(Matrix x, Matrix y);

	~CudaNetwork() {

	}

private:

	std::vector<int> _dimensions;

	float* _network;
	float* _results;
	float* _derivs;

	int _network_size;
	int _results_size;
	int _derivs_size;

};

