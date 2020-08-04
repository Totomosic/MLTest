#include "Eigen/Dense"
#include <iostream>
#include <time.h>
#include <random>
#include "NeuralNetwork.h"

int main()
{
	int sample_size = 512;

	ML::NeuralNetwork nn({ { 16, ML::RELU }, { 1, ML::LINEAR } }, 2);
	nn.set_learning_rate(0.07);
	for (int i = 0; i < 50000; i++)
	{
		Eigen::MatrixXd data = Eigen::MatrixXd::Random(sample_size, 2);
		Eigen::MatrixXd result = Eigen::MatrixXd::Constant(sample_size, 1, 0.0);
		for (int i = 0; i < sample_size; i++)
		{
			result(i, 0) = data.row(i).sum();
		}
		nn.feed_forward(data);
		nn.back_propagate(data, result);
	}
	Eigen::Vector2d point(0.5, -0.3);
	std::cout << (nn.evaluate(point.transpose())) << std::endl;

	return 0;
}
