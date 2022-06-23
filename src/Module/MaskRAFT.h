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


/*=======================*/
class ResidualBlockImpl :public torch::nn::Module {
public:
	ResidualBlockImpl(int in_planes, int planes, string norm_fn = "batch", int stride = 1);
	~ResidualBlockImpl() {};
	Tensor forward(Tensor ten);
private:
	// func
	// veriable
	// layers
	torch::nn::Conv2d conv1{ nullptr }, conv2{ nullptr };
	torch::nn::ReLU relu{ nullptr };

	torch::nn::BatchNorm2d    norm1_bn{ nullptr };
	torch::nn::BatchNorm2d    norm2_bn{ nullptr };
	torch::nn::InstanceNorm2d norm1_in{ nullptr };
	torch::nn::InstanceNorm2d norm2_in{ nullptr };

	torch::nn::Conv2d		  conv_ds{ nullptr };
	torch::nn::BatchNorm2d    norm_ds_bn{ nullptr };
	torch::nn::InstanceNorm2d norm_ds_in{ nullptr };

	torch::nn::Sequential norm1;
	torch::nn::Sequential norm2;
	torch::nn::Sequential downsample;
	;
};
TORCH_MODULE(ResidualBlock);
/*================================*/

class BasicEncoderImpl :public torch::nn::Module {
public:
	BasicEncoderImpl(int output_dim = 128, string norm_fn = "batch", float dropout = 0.0);
	~BasicEncoderImpl() {};
	Tensor forward(Tensor ten);
private:
	// func
	torch::nn::Sequential _make_layer(int dim, int stride = 1);
	// veriable
	string norm_fn;
	int in_planes;
	// layers
	torch::nn::BatchNorm2d    norm1_bn{ nullptr };
	torch::nn::InstanceNorm2d norm1_in{ nullptr };

	torch::nn::Conv2d conv1{ nullptr };
	torch::nn::ReLU relu{ nullptr };

	torch::nn::Sequential norm1;
	torch::nn::Sequential layer1{ nullptr };
	torch::nn::Sequential layer2{ nullptr };
	torch::nn::Sequential layer3{ nullptr };

	torch::nn::Conv2d conv2{ nullptr };

};
TORCH_MODULE(BasicEncoder);
/*====================================================*/

class MASK_RAFTImpl :public torch::nn::Module {
public:
	MASK_RAFTImpl(int gridLength, std::map<string, string>* pargs);
	tuple<Tensor, Tensor> forward(
		Tensor& img_t0_ten, Tensor&,
		vector<Tensor>& Masks,
		int iters, Tensor& last_flow);

private:
	// func

	// veriable
	int hidden_dim, context_dim;
	int corr_levels, corr_radius;
	float dropout;
	bool alternate_corr;
	int gridLength;

	// layers
	BasicEncoder fnet{ nullptr };
	BasicEncoder cnet{ nullptr };
	;
};
TORCH_MODULE(MASK_RAFT);
/*====================================================*/



//class BasicEncoderImpl :public torch::nn::Module {
//public:
//	BasicEncoderImpl(int output_dim = 128, string norm_fn = "batch", float dropout = 0.0);
//	tuple<Tensor> forward(Tensor& ten);
//private:
//	// func
//	// veriable
//	// layers
//	;
//};
//TORCH_MODULE(BasicEncoder);
/*====================================================*/
