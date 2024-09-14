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
		free(m_network);
		free(m_results);
		free(m_derivs);
	}

private:

	std::vector<int> m_dimensions;

	float* m_network;
	float* m_results;
	float* m_derivs;

	int m_network_size;
	int m_results_size;
	int m_derivs_size;

};

