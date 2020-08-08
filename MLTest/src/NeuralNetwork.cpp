#include "NeuralNetwork.h"
#include <iostream>
#include <fstream>

#include "Layers/DenseLayer.h"

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

	Eigen::MatrixXd NeuralNetwork::BackPropagate(const Eigen::MatrixXd& input, const Eigen::MatrixXd& expected)
	{
		const Eigen::MatrixXd* previous = &input;
		if (m_LastRoundResults.size() >= 2)
		{
			previous = &m_LastRoundResults[m_LastRoundResults.size() - 2];
		}
		Eigen::MatrixXd error = m_Layers.back()->GetError(expected);
		Eigen::MatrixXd originalError = error;

		for (int i = m_Layers.size() - 1; i >= 0; i--)
		{
			error = m_Layers[i]->BackPropagate(m_LearningRate, error, *previous);

			if (i >= 2)
				previous = &m_LastRoundResults[(size_t)i - 2];
			else
				previous = &input;
		}
		m_LastRoundResults.clear();
		return originalError;
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
		OutputMemoryStream stream;
		stream.Write(m_InputDimension);
		stream.Write(m_LearningRate);
		stream.Write((int)m_Layers.size());
		for (const auto& layer : m_Layers)
		{
			layer->Save(stream);
		}
		std::ofstream file(filename, std::ios::out | std::ios::binary);
		if (file.good())
		{
			file.write((const char*)stream.GetBufferPtr(), stream.GetDataSize());
			return true;
		}
		return false;
	}

	NeuralNetwork NeuralNetwork::Load(const std::string& filename)
	{
		std::ifstream file(filename, std::ifstream::ate | std::ifstream::binary);
		if (file.good())
		{
			int inputDimension;
			double learningRate;
			int layerCount;

			size_t filesize = file.tellg();
			file.seekg(0);

			std::byte* buffer = new std::byte[filesize];
			file.read((char*)buffer, filesize);

			InputMemoryStream stream(buffer, filesize);
			delete[] buffer;

			stream.Read(inputDimension);
			stream.Read(learningRate);
			stream.Read(layerCount);

			NeuralNetwork nn(inputDimension);
			nn.SetLearningRate(learningRate);

			for (int i = 0; i < layerCount; i++)
			{
				LayerHeader header;
				Deserialize(stream, header);
				if (header.Name == "Dense")
				{
					nn.AddLayer(DenseLayer::Load(stream));
				}
			}

			return nn;
		}
		return NeuralNetwork(0);
	}

}
