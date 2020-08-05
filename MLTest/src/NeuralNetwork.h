#pragma once
#include "Activation.h"
#include "Layer.h"
#include <vector>
#include <memory>

namespace ML
{

	class NeuralNetwork
	{
	private:
		struct Cache
		{
		public:
			Eigen::MatrixXd BeforeActivation;
			Eigen::MatrixXd AfterActivation;
		};

	private:
		int m_InputDimension;
		std::vector<std::unique_ptr<Layer>> m_Layers;
		double m_LearningRate;

		std::vector<Eigen::MatrixXd> m_LastRoundResults;

	public:
		NeuralNetwork(int input_cols);

		void SetLearningRate(double rate);

		template<typename T, typename ... Args>
		T& AddLayer(Args&& ... args)
		{
			auto layer = std::make_unique<T>(std::forward<Args>(args)...);
			T* ptr = layer.get();
			m_Layers.push_back(std::move(layer));
			return *ptr;
		}

		void Compile();
		void FeedForward(const Eigen::MatrixXd& input);
		void BackPropagate(const Eigen::MatrixXd& input, const Eigen::MatrixXd& expected);

		Eigen::MatrixXd Evaluate(const Eigen::MatrixXd& value) const;

		bool Save(const std::string& filename) const;

	public:
		static NeuralNetwork Load(const std::string& filename);

	};

}