#include "Eigen/Dense"
#include <iostream>
#include <time.h>
#include <random>
#include "NeuralNetwork.h"

#include "Layers/DenseLayer.h"

#define READ_MODEL 1
#define WRITE_MODEL 1

int main()
{
	// Approximates e^x where x belongs to [-2, 2]

#if WRITE_MODEL
	int batchSize = 512;
	int epochs = 200000;

	ML::NeuralNetwork nn(1);

	nn.AddLayer<ML::DenseLayer>(32).SetActivation(ML::RELU);
	nn.AddLayer<ML::DenseLayer>(1).SetActivation(ML::LINEAR);

	nn.Compile();

	nn.SetLearningRate(0.01);
	for (int i = 0; i < epochs; i++)
	{
		Eigen::MatrixXd data = Eigen::MatrixXd::Random(batchSize, 1) * 2.0;
		Eigen::MatrixXd result = data.unaryExpr([](double x) { return std::exp(x); });
		nn.FeedForward(data);
		Eigen::MatrixXd error = nn.BackPropagate(data, result);
		if (i % 2000 == 0)
			std::cout << "Error: " << error.sum() << " (" << i << "/" << epochs << " " << (int)((double)i / epochs * 100) << "%)" << std::endl;
	}
	Eigen::MatrixXd point = Eigen::MatrixXd::Constant(1, 1, 1.0);
	std::cout << (nn.Evaluate(point.transpose())) << std::endl;

	nn.Save("Network.dat");
#endif
#if READ_MODEL
	{
		ML::NeuralNetwork nn = ML::NeuralNetwork::Load("Network.dat");
		Eigen::MatrixXd point = Eigen::MatrixXd::Constant(1, 1, 1.0);
		std::cout << (nn.Evaluate(point.transpose())) << std::endl;
	}
#endif

	return 0;
}
