#include "Eigen/Dense"
#include <iostream>
#include <time.h>
#include <random>
#include "NeuralNetwork.h"

#include "Layers/DenseLayer.h"

int main()
{
	int sample_size = 512;

	ML::NeuralNetwork nn(2);

	nn.AddLayer<ML::DenseLayer>(4).SetActivation(ML::RELU);
	nn.AddLayer<ML::DenseLayer>(1).SetActivation(ML::LINEAR);

	nn.Compile();

	nn.SetLearningRate(0.005);
	for (int i = 0; i < 50000; i++)
	{
		Eigen::MatrixXd data = Eigen::MatrixXd::Random(sample_size, 2);
		Eigen::MatrixXd result = Eigen::MatrixXd::Constant(sample_size, 1, 0.0);
		for (int i = 0; i < sample_size; i++)
		{
			result(i, 0) = data.row(i).sum();
		}
		nn.FeedForward(data);
		nn.BackPropagate(data, result);
	}
	Eigen::Vector2d point(0.52, 0.15);
	std::cout << (nn.Evaluate(point.transpose())) << std::endl;

	return 0;
}
