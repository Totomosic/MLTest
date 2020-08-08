#include "DenseLayer.h"
#include <iostream>

namespace ML
{

    DenseLayer::DenseLayer(int dimension) : Layer(),
        m_Dimension(dimension), m_Weights(), m_Biases(), m_Before(), m_After()
    {
    }

    int DenseLayer::GetOutputDimension() const
    {
        return m_Dimension;
    }

    void DenseLayer::Initialize(const Layer& previous)
    {
        InitializeAsInput(previous.GetOutputDimension());
    }

    void DenseLayer::InitializeAsInput(int inputDimension)
    {
        m_Weights = Eigen::MatrixXd::Random(inputDimension, m_Dimension);
        m_Biases = Eigen::RowVectorXd::Random(m_Dimension);
    }

    void DenseLayer::FeedForward(Eigen::MatrixXd& values)
    {
        m_Before = (values * m_Weights).rowwise() + m_Biases;
        m_After = m_Before;
        Activate(m_After);
        values = m_After;
    }

    Eigen::MatrixXd DenseLayer::BackPropagate(double learningRate, const Eigen::MatrixXd& error, const Eigen::MatrixXd& previous)
    {
        Eigen::MatrixXd derivative = m_Before;
        ActivateDerivative(derivative);
        Eigen::MatrixXd dLoss = error.cwiseProduct(derivative);
        Eigen::MatrixXd newError = dLoss * m_Weights.transpose();
        Eigen::MatrixXd dLoss_w = 2.0 / previous.rows() * (previous.transpose() * dLoss);
        Eigen::MatrixXd dLoss_b = 1.0 / previous.rows() * dLoss;

        m_Weights = m_Weights - learningRate * dLoss_w;
        m_Biases = m_Biases - learningRate * dLoss_b.row(0);
        return newError;
    }

    void DenseLayer::Evaluate(Eigen::MatrixXd& values)
    {
        values = (values * m_Weights).rowwise() + m_Biases;
        Activate(values);
    }

    Eigen::MatrixXd DenseLayer::GetError(const Eigen::MatrixXd& values)
    {
        return m_After - values;
    }

    void DenseLayer::Save(OutputMemoryStream& stream) const
    {
        LayerHeader header = { "Dense" };
        Serialize(stream, header);
        stream.Write(GetOutputDimension());
        stream.Write(GetActivation().Name);
        stream.Write(m_Weights);
        stream.Write(m_Biases);
    }

    std::unique_ptr<DenseLayer> DenseLayer::Load(InputMemoryStream& stream)
    {
        int dimension;
        stream.Read(dimension);

        std::unique_ptr<DenseLayer> layer = std::make_unique<DenseLayer>(dimension);

        std::string activationName;
        stream.Read(activationName);
        stream.Read(layer->m_Weights);
        stream.Read(layer->m_Biases);

        layer->SetActivation(ACTIVATION_MAP[activationName]);
        return layer;
    }

}
