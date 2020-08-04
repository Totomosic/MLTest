#pragma once
#include "Eigen/Dense"
#include <vector>
#include <functional>

namespace ML
{

	using MatrixFn = std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)>;

	struct ActivationFunction
	{
	public:
		std::string Name;
		MatrixFn Function;
		MatrixFn Derivative;
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
			Eigen::MatrixXd BeforeActivation;
			Eigen::MatrixXd AfterActivation;
		};

	private:
		std::vector<Eigen::MatrixXd> m_Weights;
		std::vector<Eigen::RowVectorXd> m_Biases;
		std::vector<ActivationFunction> m_Functions;
		std::vector<Cache> m_Cache;

		double m_LearningRate;

	public:
		NeuralNetwork(const std::vector<Layer>& topology, int input_cols);

		void set_learning_rate(double rate);

		void feed_forward(const Eigen::MatrixXd& input);
		void back_propagate(const Eigen::MatrixXd& input, const Eigen::MatrixXd& expected);

		Eigen::MatrixXd evaluate(const Eigen::MatrixXd& value) const;

		bool save(const std::string& filename) const;

	public:
		static NeuralNetwork load(const std::string& filename);

	};

	extern ActivationFunction RELU;
	extern ActivationFunction LINEAR;

}