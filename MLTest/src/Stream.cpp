#include "Stream.h"
#include <algorithm>
#include <iostream>

namespace ML
{

    OutputMemoryStream::OutputMemoryStream() : OutputMemoryStream(1)
    {
    }

    OutputMemoryStream::OutputMemoryStream(size_t capacity)
        : m_Capacity(capacity), m_Head(0), m_Buffer(std::make_unique<std::byte[]>(capacity))
    {
    }

    size_t OutputMemoryStream::GetCapacity() const
    {
        return m_Capacity;
    }

    size_t OutputMemoryStream::GetDataSize() const
    {
        return m_Head;
    }

    void* OutputMemoryStream::GetBufferPtr() const
    {
        return (void*)m_Buffer.get();
    }

    void OutputMemoryStream::Reserve(size_t capacity)
    {
        if (m_Capacity < capacity)
        {
            std::byte* ptr = m_Buffer.release();
            m_Capacity = std::max(m_Capacity * 2, capacity);
            m_Buffer = std::make_unique<std::byte[]>(m_Capacity);
            memcpy(m_Buffer.get(), ptr, m_Head);
            delete[] ptr;
        }
    }

    void OutputMemoryStream::Write(const void* data, size_t size)
    {
        Reserve(m_Head + size);
        memcpy(m_Buffer.get() + m_Head, data, size);
        m_Head += size;
    }

    void OutputMemoryStream::Write(const int& data)
    {
        Write(&data, sizeof(int));
    }

    void OutputMemoryStream::Write(const float& data)
    {
        Write(&data, sizeof(float));
    }

    void OutputMemoryStream::Write(const double& data)
    {
        Write(&data, sizeof(double));
    }

    void OutputMemoryStream::Write(const std::string& data)
    {
        int length = data.size();
        Write(length);
        Write((const void*)data.data(), length * sizeof(char));
    }

    void OutputMemoryStream::Write(const Eigen::RowVectorXd& data)
    {
        Write((int)data.cols());
        for (int i = 0; i < data.cols(); i++)
            Write(data(0, i));
    }

    void OutputMemoryStream::Write(const Eigen::MatrixXd& data)
    {
        Write((int)data.rows());
        Write((int)data.cols());
        for (int i = 0; i < data.rows(); i++)
            for (int j = 0; j < data.cols(); j++)
                Write(data(i, j));
    }

    InputMemoryStream::InputMemoryStream(const void* data, size_t size)
        : m_Capacity(size), m_Head(0), m_Buffer(std::make_unique<std::byte[]>(size))
    {
        memcpy(m_Buffer.get(), data, size);
    }

    size_t InputMemoryStream::GetCapacity() const
    {
        return m_Capacity;
    }

    size_t InputMemoryStream::GetDataSize() const
    {
        return m_Capacity - m_Head;
    }

    const void* InputMemoryStream::GetBufferPtr() const
    {
        return (const void*)(m_Buffer.get());
    }

    void InputMemoryStream::Read(void* buffer, size_t size)
    {
        memcpy(buffer, m_Buffer.get() + m_Head, size);
        m_Head += size;
    }

    void InputMemoryStream::Read(int& data)
    {
        Read(&data, sizeof(int));
    }

    void InputMemoryStream::Read(float& data)
    {
        Read(&data, sizeof(float));
    }

    void InputMemoryStream::Read(double& data)
    {
        Read(&data, sizeof(double));
    }

    void InputMemoryStream::Read(std::string& data)
    {
        int length;
        Read(length);
        data.resize(length);
        Read(data.data(), length * sizeof(char));
    }

    void InputMemoryStream::Read(Eigen::RowVectorXd& data)
    {
        int cols;
        Read(cols);
        data = Eigen::RowVectorXd::Constant((Eigen::Index)cols, 0);
        for (int i = 0; i < data.cols(); i++)
            Read(data(0, i));
    }

    void InputMemoryStream::Read(Eigen::MatrixXd& data)
    {
        int rows;
        int cols;
        Read(rows);
        Read(cols);
        data = Eigen::MatrixXd::Constant(rows, cols, 0.0);
        for (int i = 0; i < data.rows(); i++)
            for (int j = 0; j < data.cols(); j++)
                Read(data(i, j));
    }

}
