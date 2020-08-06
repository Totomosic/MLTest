#include "NeuralNetwork.h"
#include <iostream>

namespace ML
{

	NeuralNetwork::NeuralNetwork(int inputCols)
		: m_InputDimension(inputCols), m_Layers(), m_LearningRate(0.05), m_LastRoundResults()
	{
	}

	void NeuralNetwork::SetLearningRate(double rate)
	{
		m_LearningRate = rate;
	}

	void NeuralNetwork::Compile()
	{
		const Layer* currentLayer = nullptr;
		for (const std::unique_ptr<Layer>& layer : m_Layers)
		{
			if (currentLayer)
				layer->Initialize(*currentLayer);
			else
				layer->InitializeAsInput(m_InputDimension);
			currentLayer = layer.get();
		}
	}

	void NeuralNetwork::FeedForward(const Eigen::MatrixXd& input)
	{
		Eigen::MatrixXd current = input;
		for (const auto& layer : m_Layers)
		{
			layer->FeedForward(current);
			m_LastRoundResults.push_back(current);
		}
	}

	void NeuralNetwork::BackPropagate(const Eigen::MatrixXd& input, const Eigen::MatrixXd& expected)
	{
		Eigen::MatrixXd previous = input;
		if (m_LastRoundResults.size() >= 2)
		{
			previous = m_LastRoundResults[m_LastRoundResults.size() - 2];
		}
		Eigen::MatrixXd error = m_Layers.back()->GetError(expected);

		for (int i = m_Layers.size() - 1; i >= 0; i--)
		{
			error = m_Layers[i]->BackPropagate(m_LearningRate, error, previous);

			if (i >= 2)
				previous = m_LastRoundResults[(size_t)i - 2];
			else
				previous = input;
		}
		m_LastRoundResults.clear();
	}

	Eigen::MatrixXd NeuralNetwork::Evaluate(const Eigen::MatrixXd& value) const
	{
		Eigen::MatrixXd current = value;
		for (const auto& layer : m_Layers)
		{
			/*std::cout << "Current" << std::endl;
			std::cout << current << std::endl;
			std::cout << "Weight" << std::endl;
			std::cout << m_Weights[i] << std::endl;
			std::cout << "Bias" << std::endl;
			std::cout << broadcast_bias(m_Biases[i], value.rows()) << std::endl;*/
			layer->Evaluate(current);
		}
		return current;
	}

	bool NeuralNetwork::Save(const std::string& filename) const
	{
		return false;
	}

	NeuralNetwork NeuralNetwork::Load(const std::string& filename)
	{
		std::vector<Layer> topology;
		int input_cols = 0;
		return NeuralNetwork(input_cols);
	}

}
