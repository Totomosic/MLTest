#include "NeuralNetwork.h"
#include <iostream>

namespace ML
{

	ActivationFunction RELU = {
		"relu",
		[](const Eigen::MatrixXd& x) { return x.unaryExpr([](double x) { return std::max(x, 0.0); }); },
		[](const Eigen::MatrixXd& x) { return x.unaryExpr([](double x) { return (x > 0) ? 1.0 : 0.0; }); }
	};
	ActivationFunction LINEAR = {
		"linear",
		[](const Eigen::MatrixXd& x) { return x.unaryExpr([](double x) { return x; }); },
		[](const Eigen::MatrixXd& x) { return x.unaryExpr([](double x) { return 1.0; }); }
	};

	NeuralNetwork::NeuralNetwork(const std::vector<Layer>& topology, int input_cols)
		: m_Weights(), m_Biases(), m_Functions(), m_Cache(), m_LearningRate(0.05)
	{
		int previous_cols = input_cols;
		for (size_t i = 0; i < topology.size(); i++)
		{
			m_Weights.push_back(Eigen::MatrixXd::Random(previous_cols, topology[i].Size));
			m_Biases.push_back(Eigen::RowVectorXd::Random(topology[i].Size));
			m_Functions.push_back(topology[i].Function);
			m_Cache.push_back({});
			previous_cols = topology[i].Size;
		}
	}

	void NeuralNetwork::set_learning_rate(double rate)
	{
		m_LearningRate = rate;
	}

	void NeuralNetwork::feed_forward(const Eigen::MatrixXd& input)
	{
		Eigen::MatrixXd current_layer = input;
		for (size_t i = 0; i < m_Weights.size(); i++)
		{
			Eigen::MatrixXd z = (current_layer * m_Weights[i]).rowwise() + m_Biases[i];
			Eigen::MatrixXd a = m_Functions[i].Function(z);
			m_Cache[i] = { z, a };
			current_layer = a;
		}
	}

	void NeuralNetwork::back_propagate(const Eigen::MatrixXd& input, const Eigen::MatrixXd& expected)
	{
		Eigen::MatrixXd previous = expected;
		Eigen::MatrixXd next = input;
		if (m_Cache.size() >= 2)
		{
			next = m_Cache[m_Cache.size() - 2].AfterActivation;
		}
		Eigen::MatrixXd error = (m_Cache.back().AfterActivation - expected);

		for (int i = m_Weights.size() - 1; i >= 0; i--)
		{
			Eigen::MatrixXd dLoss = error.cwiseProduct(m_Functions[i].Derivative(m_Cache[i].BeforeActivation));
			error = dLoss * m_Weights[i].transpose();
			Eigen::MatrixXd dLoss_w = 2.0 / next.rows() * (next.transpose() * dLoss);
			Eigen::MatrixXd dLoss_b = 1.0 / next.rows() * dLoss;

			m_Weights[i] = m_Weights[i] - m_LearningRate * dLoss_w;
			m_Biases[i] = m_Biases[i] - m_LearningRate * dLoss_b.row(0);

			if (i >= 2)
				next = m_Cache[(size_t)i - 2].AfterActivation;
			else
				next = input;
		}
	}

	Eigen::MatrixXd NeuralNetwork::evaluate(const Eigen::MatrixXd& value) const
	{
		Eigen::MatrixXd current = value;
		for (size_t i = 0; i < m_Weights.size(); i++)
		{
			/*std::cout << "Current" << std::endl;
			std::cout << current << std::endl;
			std::cout << "Weight" << std::endl;
			std::cout << m_Weights[i] << std::endl;
			std::cout << "Bias" << std::endl;
			std::cout << broadcast_bias(m_Biases[i], value.rows()) << std::endl;*/
			current = m_Functions[i].Function((current * m_Weights[i]).rowwise() + m_Biases[i]);
		}
		return current;
	}

	bool NeuralNetwork::save(const std::string& filename) const
	{
		return false;
	}

	NeuralNetwork NeuralNetwork::load(const std::string& filename)
	{
		std::vector<Layer> topology;
		int input_cols = 0;
		return NeuralNetwork(topology, input_cols);
	}

}
