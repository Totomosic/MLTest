#pragma once
#include "Eigen/Dense"
#include <functional>
#include <string>
#include <unordered_map>

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
	extern ActivationFunction SWISH;

	extern std::unordered_map<std::string, ActivationFunction> ACTIVATION_MAP;

}
