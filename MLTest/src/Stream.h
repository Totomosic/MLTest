#pragma once
#include <memory>
#include <string>
#include "Eigen/Dense"

namespace ML
{

	class OutputMemoryStream
	{
	private:
		size_t m_Capacity;
		size_t m_Head;
		std::unique_ptr<std::byte[]> m_Buffer;

	public:
		OutputMemoryStream();
		OutputMemoryStream(size_t capacity);

		size_t GetCapacity() const;
		size_t GetDataSize() const;
		void* GetBufferPtr() const;

		void Reserve(size_t capacity);

		void Write(const void* data, size_t size);
		void Write(const int& data);
		void Write(const float& data);
		void Write(const double& data);
		void Write(const std::string& data);
		void Write(const Eigen::RowVectorXd& data);
		void Write(const Eigen::MatrixXd& data);
	};

	class InputMemoryStream
	{
	private:
		size_t m_Capacity;
		size_t m_Head;
		std::unique_ptr<std::byte[]> m_Buffer;

	public:
		InputMemoryStream(const void* data, size_t size);

		size_t GetCapacity() const;
		size_t GetDataSize() const;
		const void* GetBufferPtr() const;

		void Read(void* buffer, size_t size);
		void Read(int& data);
		void Read(float& data);
		void Read(double& data);
		void Read(std::string& data);
		void Read(Eigen::RowVectorXd& data);
		void Read(Eigen::MatrixXd& data);
	};

}
