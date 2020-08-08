#pragma once
#include "../Layer.h"

namespace ML
{

	class DenseLayer : public Layer
	{
	private:
		int m_Dimension;
		Eigen::MatrixXd m_Weights;
		Eigen::RowVectorXd m_Biases;

		Eigen::MatrixXd m_Before;
		Eigen::MatrixXd m_After;

	public:
		DenseLayer(int dimension);

		int GetOutputDimension() const override;

		void Initialize(const Layer& previous) override;
		void InitializeAsInput(int inputDimension) override;

		void FeedForward(Eigen::MatrixXd& values) override;
		Eigen::MatrixXd BackPropagate(double learningRate, const Eigen::MatrixXd& error, const Eigen::MatrixXd& previous) override;
		void Evaluate(Eigen::MatrixXd& values) override;
		Eigen::MatrixXd GetError(const Eigen::MatrixXd& values) override;

		void Save(OutputMemoryStream& stream) const override;

	public:
		static std::unique_ptr<DenseLayer> Load(InputMemoryStream& stream);
	};

}