#include "SingleBlockNeuralNetwork.h"

std::string NeuralNetwork::clean_time(double time) {
	const double hour = 3600000.00;
	const double minute = 60000.00;
	const double second = 1000.00;
	std::string out;

	if (time / hour > 1.00) {
		out = std::to_string(time / hour).append(" hours");
	}
	else if (time / minute > 1.00) {
		out = std::to_string(time / minute).append(" minutes");
	}
	else if (time / second > 1.00) {
		out = std::to_string(time / second).append(" seconds");
	}
	else {
		out = std::to_string(time).append("(ms)");
	}
	return out;
}
std::string NeuralNetwork::summary() {

	std::string out = "summary:\n\tdims := | ";

	for (size_t i = 0; i < m_dimensions.size(); i++) {
		out.append(std::to_string(m_dimensions[i])).append(" | ");
	}

	out.append("\n\tactivations := | ");
	for (size_t i = 0; i < m_activation_data.size(); i++) {
		switch (m_activation_data[i].type) {
		case activation_functions::relu:
			out.append("relu | ");
			break;
		case activation_functions::leaky_relu:
			out.append("leaky_relu | ");
			break;
		case activation_functions::elu:
			out.append("elu | ");
			break;
		case activation_functions::sigmoid:
			out.append("sigmoid | ");
			break;
		case activation_functions::softmax:
			out.append("softmax | ");
			break;
		default:
			out.append("NaN | ");
		}
	}

	out.append("\n\tparameters := ").append(std::to_string(m_network_size));
	out.append("\n\tnetwork_size := ");

	double net_size = m_network_size * sizeof(float);

	const double gb = 1000000000.00;
	const double mb = 1000000.00;
	const double kb = 1000.00;

	if (net_size > gb) {
		out.append(std::to_string(net_size / gb)).append(" gb\n");
	} else if (net_size > mb) {
		out.append(std::to_string(net_size / mb)).append(" mb\n");
	} else if (net_size > kb) {
		out.append(std::to_string(net_size / kb)).append(" kb\n");
	} else {
		out.append(std::to_string(net_size)).append(" bytes\n");
	}

	out.append("\n");
	return out;
}

void NeuralNetwork::serialize(std::string filepath) {
	std::ofstream fw(filepath, std::ios::binary);

	// write dims of network
	size_t dims = m_dimensions.size();
	fw.write(reinterpret_cast<const char*>(&dims), sizeof(size_t));
	fw.write(reinterpret_cast<const char*>(m_dimensions.data()), dims * sizeof(size_t));

	// write network
	fw.write(reinterpret_cast<const char*>(m_network), m_network_size * sizeof(float));

	fw.close();
}
void NeuralNetwork::deserialize(std::string filepath) {
	loaded = true;
	std::ifstream fr(filepath, std::ios::binary);

	if (!fr.is_open()) {
		std::cout << "Network could not be found\n";
	}

	// read dims size
	size_t dims;
	fr.read(reinterpret_cast<char*>(&dims), sizeof(size_t));
	m_dimensions = std::vector<size_t>(dims);

	// read dims
	fr.read(reinterpret_cast<char*>(m_dimensions.data()), dims * sizeof(size_t));

	// compute m_network_size
	m_network_size = 0;
	for (int i = 0; i < m_dimensions.size() - 1; i++) {
		m_network_size += (m_dimensions[i] * m_dimensions[i + 1]);
		m_network_size += m_dimensions[i + 1];
	}

	// read network
	fr.read(reinterpret_cast<char*>(m_network), m_network_size * sizeof(float));

	fr.close();
}