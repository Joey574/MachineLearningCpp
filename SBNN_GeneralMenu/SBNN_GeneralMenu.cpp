#include <iostream>
#include <Windows.h>

#include "SingleBlockNeuralNetwork.h"
#include "Matrix.h"

// datasets
#include "MNIST.cpp"
#include "FMNIST.cpp"
#include "Mandlebrot.cpp"

struct network_init {
	std::vector<size_t> dims;
	std::vector<NeuralNetwork::activation_functions> activation;
	NeuralNetwork::loss_metric loss;
	NeuralNetwork::loss_metric metric;
	NeuralNetwork::weight_init weights;
};


void menu(NeuralNetwork& model);
void load_network(NeuralNetwork& model);
void initialize_network(network_init& init);

bool is_num(const std::string& str) {
	for (size_t i = 0; i < str.size(); i++) {
		if (str[i] < '0' || str[i] > '9') {
			return false;
		}
	}
	return true;
}
size_t to_num(const std::string& str) {
	size_t dims = 0;
	for (size_t i = 0; i < str.size(); i++) {
		dims += (str[i] - '0') * powf(10, str.size() - i - 1);
	}
	return dims;
}

int main()
{
	system("CLS");
	SetPriorityClass(GetStdHandle, REALTIME_PRIORITY_CLASS);
	srand(time(0));

	NeuralNetwork model;

	menu(model);
}

void menu(NeuralNetwork& model) {
	std::string input;

	network_init init;

	L1:
	system("CLS");
	std::cout << "Select a dataset:\n1: MNIST\n2: FMNIST\n3: Mandlebrot\nInput: ";
	std::cin >> input;

	if (!is_num(input)) {
		goto L1;
	}

	Matrix x_train, y_train, x_test, y_test;
	size_t end;

	switch (to_num(input)) {
	case 1:
		std::tie(x_train, y_train, x_test, y_test) = MNIST::load_data(0, 0, 0, 0, 0, 0.0f, 1.0f);
		end = 10;
		break;
	case 2:
		std::tie(x_train, y_train, x_test, y_test) = FMNIST::load_data(0, 0, 0, 0, 0, 0.0f, 1.0f);
		end = 10;
		break;
	case 3:
		std::tie(x_train, y_train) = Mandlebrot::make_dataset(100000, 100, 0, 0, 0, 0, 0, 0.0f, 1.0f);
		end = 1;
		break;
	default:
		goto L1;
	}


	L2:
	system("CLS");
	std::cout << "Select an option:\n1: Load existing network\n2: Create new network\nInput: ";
	std::cin >> input;

	switch (input[0]) {
	case '1':
		load_network(model);
		break;
	case '2':
		initialize_network(init);
		break;
	default:
		goto L2;
	}

	init.dims[0] = x_train.ColumnCount;
	init.dims.back() = end;

	model.define(init.dims, init.activation);
	model.compile(init.loss, init.metric, init.weights);

	model.fit(
		x_train,
		y_train,
		x_test,
		y_test,
		320,
		100,
		0.01f,
		true,
		1,
		0.1f
	);
}

void load_network(NeuralNetwork& model) {
	std::string input;


}
void initialize_network(network_init& init) {
	std::string input;


	L1:
	system("CLS");
	std::cout << "Number of hidden layers?\nInput: ";
	std::cin >> input;

	if (!is_num(input)) {
		goto L1;
	}
	
	size_t dims = to_num(input);

	init.dims = std::vector<size_t>(dims + 2, 0);
	init.activation = std::vector<NeuralNetwork::activation_functions>(dims + 1);


	// nodes in hidden layer
	for (size_t i = 0; i < dims; i++) {

		L2:
		system("CLS");
		std::cout << "Number of nodes in layer " + std::to_string(i + 1).append("?\nInput: ");
		std::cin >> input;

		if (!is_num(input)) {
			goto L2;
		}

		init.dims[i + 1] = to_num(input);
	}

	// hidden layer activation
	for (size_t i = 0; i < dims; i++) {

		L3:
		system("CLS");
		std::cout << "Activation function for layer " + std::to_string(i + 1).append("?\n1: relu\n2: leaky_relu\n3: elu\n4: sigmoid\nInput: ");
		std::cin >> input;

		if (!is_num(input)) {
			goto L3;
		}

		switch (to_num(input)) {
		case 1:
			init.activation[i] = NeuralNetwork::activation_functions::relu;
			break;
		case 2:
			init.activation[i] = NeuralNetwork::activation_functions::leaky_relu;
			break;
		case 3:
			init.activation[i] = NeuralNetwork::activation_functions::elu;
			break;
		case 4:
			init.activation[i] = NeuralNetwork::activation_functions::sigmoid;
			break;
		default:
			goto L3;
		}
	}

	L4:
	system("CLS");
	std::cout << "Activation function for output layer?\n1: relu\n2: leaky_relu\n3: elu\n4: sigmoid\nInput: ";
	std::cin >> input;

	if (!is_num(input) || to_num(input) > 4) {
		goto L4;
	}

	switch (to_num(input)) {
	case 1:
		init.activation[dims] = NeuralNetwork::activation_functions::relu;
		break;
	case 2:
		init.activation[dims] = NeuralNetwork::activation_functions::leaky_relu;
		break;
	case 3:
		init.activation[dims] = NeuralNetwork::activation_functions::elu;
		break;
	case 4:
		init.activation[dims] = NeuralNetwork::activation_functions::sigmoid;
		break;
	default:
		goto L4;
	}


	L5:
	system("CLS");
	std::cout << "Loss function?\n1: mae\n2: one_hot\nInput: ";
	std::cin >> input;

	if (!is_num(input)) {
		goto L5;
	}

	switch (to_num(input)) {
	case 1:
		init.loss = NeuralNetwork::loss_metric::mae;
		break;
	case 2:
		init.loss = NeuralNetwork::loss_metric::one_hot;
		break;
	default:
		goto L5;
	}


	L6:
	system("CLS");
	std::cout << "Score function?\n1: mae\n2: accuracy\nInput: ";
	std::cin >> input;

	if (!is_num(input)) {
		goto L6;
	}

	switch (to_num(input)) {
	case 1:
		init.metric = NeuralNetwork::loss_metric::mae;
		break;
	case 2:
		init.metric = NeuralNetwork::loss_metric::accuracy;
		break;
	default:
		goto L6;
	}


	L7:
	system("CLS");
	std::cout << "Weight initialization\n1: He\n2: Normalize\n3: Xavier\nInput: ";
	std::cin >> input;

	if (!is_num(input)) {
		goto L7;
	}

	switch (to_num(input)) {
	case 1:
		init.weights = NeuralNetwork::weight_init::he;
		break;
	case 2:
		init.weights = NeuralNetwork::weight_init::normalize;
		break;
	case 3:
		init.weights = NeuralNetwork::weight_init::xavier;
		break;
	default:
		goto L7;
	}
	system("CLS");
}