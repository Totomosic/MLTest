#pragma once
#include "Eigen/Dense"
#include <vector>
#include <functional>

namespace ML
{

	struct ActivationFunction
	{
	public:
		std::function<double(double)> Function;
		std::function<double(double)> Derivative;
	};

	struct Layer
	{
	public:
		int Size;
		ActivationFunction Function;
	};

	class NeuralNetwork
	{
	private:
		struct Cache
		{
		public:
			Eigen::MatrixXd Z;
			Eigen::MatrixXd A;
		};

	private:
		std::vector<Eigen::MatrixXd> m_Weights;
		std::vector<Eigen::MatrixXd> m_Biases;
		std::vector<Eigen::MatrixXd> m_Layers;
		std::vector<ActivationFunction> m_Functions;
		std::vector<Cache> m_Cache;

		double m_LearningRate;

	public:
		NeuralNetwork(const std::vector<Layer>& topology, int input_cols);

		void set_learning_rate(double rate);

		void feed_forward(const Eigen::MatrixXd& input);
		void back_propagate(const Eigen::MatrixXd& input, const Eigen::MatrixXd& expected);

		Eigen::MatrixXd evaluate(const Eigen::MatrixXd& value) const;

	private:
		Eigen::MatrixXd broadcast_bias(const Eigen::MatrixXd& bias, int rows) const;

	};

}