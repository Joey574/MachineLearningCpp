#pragma once

#include <iostream>
#include <vector>

#include "Matrix.h"

class NeuralNetwork {

public:


private:

	struct network {
		std::vector<Matrix> weights;
		// switch to modern at somepoint
		std::vector<std::vector<float>> biases;
	};

	network m_network;

	struct result_matrices {

	};

	struct derivative_matrices {

	};

	template<float l1, float l2>
	struct back_prop {
		static network weight_update(network net, derivative_matrices derivs, float learning_rate) {

			for (int i = 0; i < net.weights.size(); i++) {
				if constexpr (l1) {

				}

				if constexpr (l2) {

				}
			}

			
		}
	};

};