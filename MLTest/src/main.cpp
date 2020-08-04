#include "Eigen/Dense"
#include <iostream>
#include <time.h>
#include <random>
#include "NeuralNetwork.h"

using namespace Eigen;

int main()
{
	srand(42);

	ML::ActivationFunction relu = { [](double x) { return std::max(x, 0.0); }, [](double x) { return (x > 0) ? 1.0 : 0.0; } };
	ML::ActivationFunction linear = { [](double x) { return x; }, [](double x) { return 1.0; } };

	int sample_size = 256;

	ML::NeuralNetwork nn({ { 64, relu }, { 1, linear } }, 1);
	nn.set_learning_rate(0.0002);
	for (int i = 0; i < 25000; i++)
	{
		MatrixXd data = Eigen::MatrixXd::Random(sample_size, 1);
		MatrixXd result = data.unaryExpr([](double x) { return x * x; });
		nn.feed_forward(data);
		nn.back_propagate(data, result);
	}
	std::cout << (nn.evaluate(MatrixXd::Constant(1, 1, 0.5))(0, 0)) << std::endl;

	return 0;
}
