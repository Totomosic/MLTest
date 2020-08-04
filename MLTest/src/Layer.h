#pragma once
#include "Eigen/Dense"

namespace ML
{

	class Layer;

	class Layer
	{
	private:
		Eigen::MatrixXd m_Weights;
		Eigen::MatrixXd m_Bias;

	public:
		Layer();
	};

}