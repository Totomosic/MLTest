#pragma once
#include "Eigen/Dense"
#include <functional>
#include <string>

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

	extern ActivationFunction RELU;
	extern ActivationFunction TANH;
	extern ActivationFunction SIGMOID;
	extern ActivationFunction LINEAR;

}
