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

/*=======================*/


//class downsampleImpl :public torch::nn::Module {
//public:
//	downsampleImpl(int in_planes, int planes, int kernel_size = 1, int stride = 2, string norm_fn = "batch");
//	tuple<Tensor> forward(Tensor& ten);
//private:
//	torch::nn::Conv2d		  conv_ds{ nullptr };
//	torch::nn::BatchNorm2d    norm_ds_bn{ nullptr };
//	torch::nn::InstanceNorm2d norm_ds_in{ nullptr };
//};
//TORCH_MODULE(downsample);

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
	torch::nn::Sequential norm3;
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
	torch::nn::Sequential layer1;
	torch::nn::Sequential layer2;
	torch::nn::Sequential layer3;

	torch::nn::Conv2d conv2{ nullptr };

};
TORCH_MODULE(BasicEncoder);
/*====================================================*/

class BasicMotionEncoderImpl :public torch::nn::Module {
public:
	BasicMotionEncoderImpl(int corr_levels, int corr_radius);
	Tensor forward(Tensor flow, Tensor corr);
private:
	// layers
	torch::nn::Conv2d convc1{ nullptr }, convc2{ nullptr };
	torch::nn::Conv2d convf1{ nullptr }, convf2{ nullptr };
	torch::nn::Conv2d conv{ nullptr };
	torch::nn::ReLU relu{ nullptr };
	;
};
TORCH_MODULE(BasicMotionEncoder);

class SepConvGRUImpl :public torch::nn::Module {
public:
	SepConvGRUImpl(int hidden_dim = 128, int input_dim = 192 + 128);
	Tensor forward(Tensor h, Tensor x);
private:
	// layers
	torch::nn::Conv2d convz1{ nullptr };
	torch::nn::Conv2d convr1{ nullptr };
	torch::nn::Conv2d convq1{ nullptr };
	torch::nn::Conv2d convz2{ nullptr };
	torch::nn::Conv2d convr2{ nullptr };
	torch::nn::Conv2d convq2{ nullptr };
	;
};
TORCH_MODULE(SepConvGRU);

class FlowHeadImpl :public torch::nn::Module {
public:
	FlowHeadImpl(int input_dim = 128, int hidden_dim = 256);
	Tensor forward(Tensor x);
private:
	torch::nn::Conv2d conv1{ nullptr };
	torch::nn::Conv2d conv2{ nullptr };
	torch::nn::ReLU relu{ nullptr };
};
TORCH_MODULE(FlowHead);

class BasicUpdateBlockImpl :public torch::nn::Module {
public:
	BasicUpdateBlockImpl(int corr_levels, int corr_radius, int hidden_dim = 128);
	tuple<Tensor, Tensor, Tensor> forward(Tensor net, Tensor inp, Tensor corr, Tensor flow);
private:
	// func
	// veriable3
	int corr_levels;
	int corr_radius;

	BasicMotionEncoder encoder{ nullptr };
	SepConvGRU gru{ nullptr };
	FlowHead flow_head{ nullptr };
	torch::nn::Sequential mask;

	// layers
	;
};
TORCH_MODULE(BasicUpdateBlock);
/*=====================================================*/

class MaskCorrBlock {
public:
	MaskCorrBlock(Tensor fmap1, Tensor fmap2,
		int num_levels = 4, int radius = 4,
		int gridLength = 4);
	Tensor __call__(Tensor);
	static Tensor corrCoculate(Tensor fmap1, Tensor fmap2);
	vector<Tensor> corr_pyramid;

private:
	int num_levels;
	int radius;
	Tensor corr;
	int gridLength;
	int h_num, w_num;
	Tensor Mask_big, Mask_big_2;

};
/*====================================================*/

class MASK_RAFTImpl :public torch::nn::Module {
public:
	MASK_RAFTImpl(int gridLength, std::map<string, string>* pargs);
	tuple<Tensor, Tensor> forward(
		Tensor& img_t0_ten, Tensor&,
		vector<Tensor>& Masks,
		int iters, Tensor& last_flow);

private:
	// veriable
	int hidden_dim, context_dim;
	int corr_levels, corr_radius;
	float dropout;
	bool alternate_corr;
	int gridLength;

	// layers
	BasicEncoder fnet{ nullptr };
	BasicEncoder cnet{ nullptr };
	BasicUpdateBlock update_block{ nullptr };
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
