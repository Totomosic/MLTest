#pragma once
#include "Activation.h"
#include "Stream.h"

namespace ML
{

	struct LayerHeader
	{
	public:
		std::string Name;
	};

	inline void Serialize(OutputMemoryStream& stream, const LayerHeader& header)
	{
		stream.Write(header.Name);
	}

	inline void Deserialize(InputMemoryStream& stream, LayerHeader& header)
	{
		stream.Read(header.Name);
	}

	class Layer
	{
	private:
		ActivationFunction m_Activation;

	public:
		Layer() = default;
		virtual ~Layer() = default;

		inline const ActivationFunction& GetActivation() const { return m_Activation; }
		inline void SetActivation(const ActivationFunction& fn) { m_Activation = fn; }
		virtual int GetOutputDimension() const = 0;

		virtual void Initialize(const Layer& previous) = 0;
		virtual void InitializeAsInput(int inputDimension) = 0;
		
		virtual void FeedForward(Eigen::MatrixXd& values) = 0;
		virtual Eigen::MatrixXd BackPropagate(double learningRate, const Eigen::MatrixXd& error, const Eigen::MatrixXd& previous) = 0;
		virtual void Evaluate(Eigen::MatrixXd& values) = 0;
		virtual Eigen::MatrixXd GetError(const Eigen::MatrixXd& values) = 0;

		virtual void Save(OutputMemoryStream& stream) const = 0;

	protected:
		inline void Activate(Eigen::MatrixXd& values) { values = m_Activation.Function(values); }
		inline void ActivateDerivative(Eigen::MatrixXd& values) { values = m_Activation.Derivative(values); }

	};

}
