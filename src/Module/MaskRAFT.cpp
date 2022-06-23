#include "MaskRAFT.h"

inline torch::nn::Conv2dOptions conv_options(int64_t in_planes, int64_t out_planes, int64_t kerner_size,
	int64_t stride = 1, int64_t padding = 0, bool with_bias = false) {
	torch::nn::Conv2dOptions conv_options = torch::nn::Conv2dOptions(in_planes, out_planes, kerner_size);
	conv_options.stride(stride);
	conv_options.padding(padding);
	conv_options.bias(with_bias);
	return conv_options;
}
inline torch::nn::MaxPool2dOptions maxpool_options(int kernel_size, int stride) {
	torch::nn::MaxPool2dOptions maxpool_options(kernel_size);
	maxpool_options.stride(stride);
	return maxpool_options;
}


/*===============================================================================================*/
/*===============================================================================================*/
/*===============================================================================================*/

ResidualBlockImpl::ResidualBlockImpl(int in_planes, int planes, string norm_fn, int stride) {
	this->conv1 = torch::nn::Conv2d(conv_options(in_planes, planes, 3, stride, 1));
	this->conv2 = torch::nn::Conv2d(conv_options(planes, planes, 3, 1, 1));
	this->relu = torch::nn::ReLU(torch::nn::ReLUOptions(true));

	// 正则化模块装填
	this->norm1_bn = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(planes));
	this->norm2_bn = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(planes));
	this->norm1_in = torch::nn::InstanceNorm2d(torch::nn::InstanceNorm2dOptions(planes));
	this->norm2_in = torch::nn::InstanceNorm2d(torch::nn::InstanceNorm2dOptions(planes));
	if (norm_fn == "batch") {
		this->norm1->push_back(this->norm1_bn);
		this->norm2->push_back(this->norm2_bn);
	}
	else if (norm_fn == "instance") {
		this->norm1->push_back(this->norm1_in);
		this->norm2->push_back(this->norm2_in);
	}

	// 下采样模块装填
	this->conv_ds = torch::nn::Conv2d(conv_options(in_planes, planes, 1, stride, 0));
	this->norm_ds_bn = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(planes));
	this->norm_ds_in = torch::nn::InstanceNorm2d(torch::nn::InstanceNorm2dOptions(planes));
	if (stride != 1) {
		this->downsample->push_back(this->conv_ds);
		if (norm_fn == "batch") {
			this->downsample->push_back(this->norm_ds_bn);
		}
		else if (norm_fn == "instance") {
			this->downsample->push_back(this->norm_ds_in);
		}
	}
	else {
		this->downsample->push_back(torch::nn::Identity());
	}
		
	// 模块注册
	this->conv1 = register_module("conv1", this->conv1);
	this->norm1 = register_module("norm1", this->norm1);
	this->relu = register_module("relu", this->relu);
	this->conv2 = register_module("conv2", this->conv2);
	this->norm2 = register_module("norm2", this->norm2);
	this->downsample = register_module("downsample", this->downsample);

}


Tensor ResidualBlockImpl::forward(Tensor  x) {
	auto y = x;
	y = this->conv1->forward(y);
	y = this->norm1->forward(y);
	y = this->relu->forward(y);
	y = this->conv2->forward(y);
	y = this->norm2->forward(y);
	y = this->relu->forward(y);

	x = this->downsample->forward(x);
	y = this->relu(x + y);
	return y;
}

/*===============================================================================================*/
/*===============================================================================================*/
/*===============================================================================================*/

torch::nn::Sequential BasicEncoderImpl::_make_layer(int dim, int stride) {
	torch::nn::Sequential layers;
	layers->push_back(ResidualBlock(this->in_planes, dim, this->norm_fn, stride));
	layers->push_back(ResidualBlock(dim, dim, this->norm_fn, 1));
	this->in_planes = dim;
	return layers;
}

BasicEncoderImpl::BasicEncoderImpl(int output_dim, string norm_fn, float dropout) {
	this->norm_fn = norm_fn;
	// 正则化模块装填
	this->norm1_bn = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(64));
	this->norm1_in = torch::nn::InstanceNorm2d(torch::nn::InstanceNorm2dOptions(64));
	if (norm_fn == "batch") {
		this->norm1->push_back(this->norm1_bn);
	}
	else if (norm_fn == "instance") {
		this->norm1->push_back(this->norm1_in);
	}

	this->conv1 = torch::nn::Conv2d(conv_options(3, 64, 7, 2, 3));
	this->relu = torch::nn::ReLU(torch::nn::ReLUOptions(true));

	this->in_planes = 64;

	this->layer1 = torch::nn::Sequential(this->_make_layer(64, 1));
	this->layer2 = torch::nn::Sequential(this->_make_layer(96, 2));
	this->layer3 = torch::nn::Sequential(this->_make_layer(128, 2));

	this->conv2 = torch::nn::Conv2d(conv_options(128, output_dim, 1));

	register_module("conv1", this->conv1);
	register_module("norm1", this->norm1);
	register_module("relu", this->relu);
	register_module("layer1", this->layer1);
	register_module("layer2", this->layer2);
	register_module("layer3", this->layer3);
	register_module("conv2", this->conv2);
}

Tensor BasicEncoderImpl::forward(Tensor x) {
	x = this->conv1->forward(x);
	x = this->norm1->forward(x);
	x = this->relu->forward(x);

	x = this->layer1->forward(x);
	x = this->layer2->forward(x);
	x = this->layer3->forward(x);
	
	x = this->conv2->forward(x);
	return x;
}
/*===============================================================================================*/
/*===============================================================================================*/
/*===============================================================================================*/

MASK_RAFTImpl::MASK_RAFTImpl(int gridLength, std::map<string, string>* pargs) {
	//init
	this->hidden_dim = 128;
	this->context_dim = 128;
	this->corr_levels = 4;
	this->corr_radius = 4;
	this->dropout = 0;
	this->alternate_corr = false;


	this->fnet = BasicEncoder(256, "instance", this->dropout);
	this->cnet = BasicEncoder(this->hidden_dim + this->context_dim, "batch", this->dropout);

	//self.update_block = BasicUpdateBlock(self.args, hidden_dim = hdim);

	std::cout << "weights have been loaded for MASK-RAFT..." << std::endl;
	this->gridLength = gridLength / 8;

	register_module("fnet", this->fnet);
	register_module("cnet", this->cnet);
}

tuple<torch::Tensor, torch::Tensor> MASK_RAFTImpl::forward(
	torch::Tensor& img_t0_ten, torch::Tensor& img_t1_ten,
	vector<torch::Tensor>& Masks,
	int iters, torch::Tensor& last_flow) {
	
	//初始化
	auto opt = torch::TensorOptions().dtype(torch::kFloat).device(img_t0_ten.device());
	Tensor flow_up = torch::rand({ 1,1,640,640 }, opt);

	//this->fnet->to(img_t0_ten.device());
	//this->cnet->to(img_t0_ten.device());
	
	// input
	img_t0_ten = 2 * img_t0_ten - 1.0;
	img_t1_ten = 2 * img_t1_ten - 1.0;
	img_t0_ten = img_t0_ten.contiguous();
	img_t1_ten = img_t1_ten.contiguous();

	//ResidualBlockImpl temp = ResidualBlockImpl(3,16,"batch", 2);	//	模块检查
	//std::cout << temp << std::endl;
	//auto m = temp.forward(img_t0_ten);
	//std::cout << m.sizes() << std::endl;
	//std::cout << m.max() << std::endl;
	//std::cout << m.min() << std::endl;

	//BasicEncoder temp = BasicEncoder(256, "instance", 0);	//	模块检查
	//std::cout << temp << std::endl;
	//auto m = temp->forward(img_t0_ten);
	//std::cout << m.sizes() << std::en
	//std::cout << m.max() << std::endl;
	//std::cout << m.min() << std::endl;

	auto fmap1 = this->fnet->forward(img_t0_ten);
	auto fmap2 = this->fnet->forward(img_t1_ten);

	auto cmap1 = this->cnet->forward(img_t0_ten);
	
	return { flow_up, fmap1 };
}