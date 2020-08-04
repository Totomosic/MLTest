#include "NeuralNetwork.h"
#include <iostream>

namespace ML
{

	NeuralNetwork::NeuralNetwork(const std::vector<Layer>& topology, int input_cols)
		: m_Weights(), m_Biases(), m_Layers(), m_Functions(), m_Cache(), m_LearningRate(0.05)
	{
		int previous_cols = input_cols;
		for (size_t i = 0; i < topology.size(); i++)
		{
			m_Weights.push_back(Eigen::MatrixXd::Random(previous_cols, topology[i].Size));
			m_Biases.push_back(Eigen::MatrixXd::Random(1, topology[i].Size));
			m_Layers.push_back(Eigen::MatrixXd::Constant(1, topology[i].Size, 0));
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
		const Eigen::MatrixXd* current_layer = &input;
		for (size_t i = 0; i < m_Layers.size(); i++)
		{
			Eigen::MatrixXd z = (*current_layer) * m_Weights[i] + broadcast_bias(m_Biases[i], input.rows());
			Eigen::MatrixXd a = z.unaryExpr(m_Functions[i].Function);
			m_Cache[i] = { z, a };
			m_Layers[i] = a;
			current_layer = &m_Layers[i];
		}
	}

	void NeuralNetwork::back_propagate(const Eigen::MatrixXd& input, const Eigen::MatrixXd& expected)
	{
		Eigen::MatrixXd previous = expected;
		Eigen::MatrixXd next = input;
		if (m_Cache.size() >= 2)
		{
			next = m_Cache[m_Cache.size() - 2].A;
		}
		Eigen::MatrixXd error = m_Layers.back() - expected;

		for (int i = m_Layers.size() - 1; i >= 0; i--)
		{
			Eigen::MatrixXd dLoss = error.cwiseProduct(m_Cache[i].Z.unaryExpr(m_Functions[i].Derivative));
			error = dLoss * m_Weights[i].transpose();
			Eigen::MatrixXd dLoss_w = 1.0 / next.cols() * (next.transpose() * dLoss);
			Eigen::MatrixXd dLoss_b = 1.0 / next.cols() * Eigen::MatrixXd(Eigen::MatrixXd::Constant(dLoss.cols(), dLoss.rows(), 1.0) * dLoss);

			m_Weights[i] = m_Weights[i] - m_LearningRate * dLoss_w;
			m_Biases[i] = m_Biases[i] - m_LearningRate * dLoss_b.row(0);

			if (i > 1)
				next = m_Cache[i - 2].A;
			else
				next = input;
		}
	}

	Eigen::MatrixXd NeuralNetwork::evaluate(const Eigen::MatrixXd& value) const
	{
		Eigen::MatrixXd current = value;
		for (size_t i = 0; i < m_Layers.size(); i++)
		{
			/*std::cout << "Current" << std::endl;
			std::cout << current << std::endl;
			std::cout << "Weight" << std::endl;
			std::cout << m_Weights[i] << std::endl;
			std::cout << "Bias" << std::endl;
			std::cout << broadcast_bias(m_Biases[i], value.rows()) << std::endl;*/
			current = (current * m_Weights[i] + broadcast_bias(m_Biases[i], value.rows())).unaryExpr(m_Functions[i].Function);
		}
		return current;
	}

	Eigen::MatrixXd NeuralNetwork::broadcast_bias(const Eigen::MatrixXd& bias, int rows) const
	{
		Eigen::MatrixXd result = Eigen::MatrixXd::Constant(rows, bias.cols(), 0);
		for (int i = 0; i < rows; i++)
		{
			result.row(i) = bias.row(0);
		}
		return result;
	}

}
