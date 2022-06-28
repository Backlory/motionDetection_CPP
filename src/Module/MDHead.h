#pragma once
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/cuda.h>
#include <map>
#include <string>
#include <vector>
#include <tuple>

using std::string;
using std::vector;
using std::tuple;
using cv::Mat;
using torch::Tensor;
using torch::indexing::Slice;
using torch::indexing::None;


class UpConvImpl :public torch::nn::Module {
public:
	UpConvImpl(const int in_channels, const int out_channels);
	Tensor forward(const Tensor&, const Tensor&);
private:
	torch::nn::ConvTranspose2d upconv{ nullptr };
	torch::nn::Conv2d conv1{ nullptr };
	torch::nn::Conv2d conv2{ nullptr };
};
TORCH_MODULE(UpConv);


class DownConvImpl :public torch::nn::Module {
public:
	DownConvImpl(const int in_channels, const int out_channels, const bool pooling = true);
	tuple<Tensor, Tensor> forward(const Tensor&);
private:
	torch::nn::Conv2d conv1{ nullptr };
	torch::nn::Conv2d conv2{ nullptr };
	torch::nn::MaxPool2d maxpool{ nullptr };
	torch::nn::Identity identity{ nullptr };
	torch::nn::Sequential pool;
};
TORCH_MODULE(DownConv);


class MDHeadImpl :public torch::nn::Module {
public:
	MDHeadImpl();
	Tensor forward(const Tensor&, const Tensor&);
private:
	torch::nn::ModuleList down_convs;
	torch::nn::ModuleList up_convs;
	DownConv downConv1{ nullptr };
	DownConv downConv2{ nullptr };
	DownConv downConv3{ nullptr };
	DownConv downConv4{ nullptr };
	UpConv upConv4{ nullptr };
	UpConv upConv3{ nullptr };
	UpConv upConv2{ nullptr };
	torch::nn::Conv2d conv_final{ nullptr };
};
TORCH_MODULE(MDHead);