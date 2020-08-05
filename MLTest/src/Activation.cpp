#include "Activation.h"

namespace ML
{

	ActivationFunction RELU = {
		"relu",
		[](const Eigen::MatrixXd& x) { return x.unaryExpr([](double x) { return std::max(x, 0.0); }); },
		[](const Eigen::MatrixXd& x) { return x.unaryExpr([](double x) { return (x > 0) ? 1.0 : 0.0; }); }
	};

	ActivationFunction TANH = {
		"tanh",
		[](const Eigen::MatrixXd& x) { return x.unaryExpr([](double x) { return std::tanh(x); }); },
		[](const Eigen::MatrixXd& x) { return x.unaryExpr([](double x) { return 1 - std::pow(std::tanh(x), 2.0); }); }
	};

	static auto sigmoid = [](double x) { return 1.0 / (1.0 + std::exp(-x)); };

	ActivationFunction SIGMOID = {
		"sigmoid",
		[](const Eigen::MatrixXd& x) { return x.unaryExpr(sigmoid); },
		[](const Eigen::MatrixXd& x) { return x.unaryExpr([](double x) { return sigmoid(x) * (1.0 - sigmoid(x)); }); }
	};

	ActivationFunction LINEAR = {
		"linear",
		[](const Eigen::MatrixXd& x) { return x.unaryExpr([](double x) { return x; }); },
		[](const Eigen::MatrixXd& x) { return x.unaryExpr([](double x) { return 1.0; }); }
	};

}
