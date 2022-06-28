#include "MaskRAFT.h"

inline torch::nn::Conv2dOptions conv_options(const int64_t in_planes, const int64_t out_planes, const int64_t kerner_size,
	const int64_t stride = 1, const int64_t padding = 0, const bool with_bias = true) {
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

	this->conv_ds = torch::nn::Conv2d(conv_options(in_planes, planes, 1, stride, 0));
	this->norm_ds_bn = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(planes));
	this->norm_ds_in = torch::nn::InstanceNorm2d(torch::nn::InstanceNorm2dOptions(planes));
	// 下采样模块装填
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
	std::cout << this->norm1_bn << std::endl;
	auto m = this->norm1_bn->named_parameters();
	for (auto w : m) {
		std::cout << w.key() << std::endl;
		std::cout << w.value().sizes() << std::endl;
	}

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


MaskCorrBlock::MaskCorrBlock(Tensor fmap1, Tensor fmap2,
	int num_levels, int radius, int gridLength) {
	this->num_levels = num_levels;
	this->radius = radius;
	this->gridLength = gridLength;

	this->corr = this->corrCoculate(fmap1, fmap2);
	int batch = corr.size(0);
	int h1 = corr.size(1);
	int w1 = corr.size(2);
	int dim = corr.size(3);
	int h2 = corr.size(4);
	int w2 = corr.size(5);
	int h1w1 = h1 * w1;
	this->h_num = h1 / this->gridLength;
	this->w_num = w1 / this->gridLength;

	corr = corr.reshape({ h1w1, dim, h2, w2 });
	this->corr_pyramid.push_back(corr);
	for (int i = 0; i < this->num_levels - 1; i++) {
		corr = torch::nn::functional::avg_pool2d(corr, torch::nn::AvgPool2dOptions(2).stride(2));
		this->corr_pyramid.push_back(corr);
	}
	//std::cout << this->corr_pyramid.size();
	//std::cout << this->corr_pyramid[0].sizes() << std::endl;
	//std::cout << this->corr_pyramid[1].sizes() << std::endl;
	//std::cout << this->corr_pyramid[2].sizes() << std::endl;
	//std::cout << this->corr_pyramid[3].sizes() << std::endl;
	//std::cout << this->corr_pyramid[3].index({ 500,0 }) << std::endl;
}

Tensor bilinear_sampler(Tensor img, Tensor coords) {
	int C = img.size(1);
	int H = img.size(2);
	int W = img.size(3);
	vector<Tensor> xygrid = coords.split(1, 3); //第四维


	Tensor xgrid = 2 * (xygrid[0] / (W - 1)) - 1;
	Tensor ygrid = 2 * (xygrid[1] / (H - 1)) - 1;
	auto grid = torch::cat({ xgrid, ygrid }, -1);
	img = torch::nn::functional::grid_sample(
		img, grid,
		torch::nn::functional::GridSampleFuncOptions().align_corners(true)
	);

	return img;
}

Tensor MaskCorrBlock::__call__(Tensor coords) {
	coords = coords.permute({ 0,2,3,1 });
	int r = this->radius;
	int batch = coords.size(0);
	int h1 = coords.size(1);
	int w1 = coords.size(2);
	auto dx = torch::linspace(-r, r, 2 * r + 1).to(coords.device());
	auto dy = torch::linspace(-r, r, 2 * r + 1).to(coords.device());
	auto delta = torch::stack(torch::meshgrid({ dy, dx }, "ij"), /*dim*/2);
	auto delta_lvl = delta.view({ 1, 2 * r + 1, 2 * r + 1, 2 });
	vector<Tensor> out_pyramid;
	for (int i = 0; i < this->num_levels; i++) {
		auto centroid_lvl = coords.reshape({ batch * h1 * w1, 1, 1, 2 }) / (1 << i);
		auto coords_lvl = centroid_lvl + delta_lvl;
		auto corr_layer = this->corr_pyramid[i];
		corr_layer = bilinear_sampler(corr_layer, coords_lvl);
		corr_layer = corr_layer.view({ batch, h1, w1, -1 });
		out_pyramid.push_back(corr_layer);
	}

	auto out = torch::cat(out_pyramid, -1);
	out = out.permute({ 0,3,1,2 }).contiguous().to(torch::kF32);
	return out;
}

Tensor MaskCorrBlock::corrCoculate(Tensor fmap1, Tensor fmap2) {
	int batch = fmap1.size(0);
	int dim = fmap1.size(1);
	int ht = fmap1.size(2);
	int wd = fmap1.size(3);
	int htwd = ht * wd;
	fmap1 = fmap1.view({ batch, dim, htwd });
	fmap2 = fmap2.view({ batch, dim, htwd });

	auto corr = torch::matmul(fmap1.transpose(1, 2), fmap2);
	corr = corr.view({ batch, ht, wd, 1, ht, wd });
	auto k = torch::tensor({ dim }, torch::TensorOptions().dtype(torch::kF32).device(corr.device()));
	corr.div_(torch::sqrt(k));
	return corr;
}

/*===============================================================================================*/
/*===============================================================================================*/
/*===============================================================================================*/
BasicMotionEncoderImpl::BasicMotionEncoderImpl(int corr_levels, int corr_radius) {
	int cor_planes = corr_levels * ((2 * corr_radius + 1) * (2 * corr_radius + 1));

	this->convc1 = torch::nn::Conv2d(conv_options(cor_planes, 256, 1, 1, 0));
	this->convc2 = torch::nn::Conv2d(conv_options(256, 192, 3, 1, 1));
	this->convf1 = torch::nn::Conv2d(conv_options(2, 128, 7, 1, 3));
	this->convf2 = torch::nn::Conv2d(conv_options(128, 64, 3, 1, 1));
	this->conv = torch::nn::Conv2d(conv_options(64 + 192, 128 - 2, 3, 1, 1));
	this->relu = torch::nn::ReLU(torch::nn::ReLUOptions(true));

	register_module("convc1", this->convc1);
	register_module("convc2", this->convc2);
	register_module("convf1", this->convf1);
	register_module("convf2", this->convf2);
	register_module("conv", this->conv);
	register_module("relu", this->relu);
}

Tensor BasicMotionEncoderImpl::forward(Tensor flow, Tensor corr) {
	auto cor = this->convc1->forward(corr);
	cor = this->relu->forward(cor);
	cor = this->convc2->forward(cor);
	cor = this->relu->forward(cor);

	//std::cout << flow.dtype() << std::endl;
	//std::cout << this->convf1 << std::endl;
	//std::cout << flow << std::endl;
	auto flo = this->convf1->forward(flow);
	flo = this->relu->forward(flo);
	flo = this->convf2->forward(flo);
	flo = this->relu->forward(flo);

	auto cor_flo = torch::cat({ cor, flo }, 1);
	auto out = this->relu(this->conv->forward(cor_flo));
	return torch::cat({ out, flow }, 1);
}

SepConvGRUImpl::SepConvGRUImpl(int hidden_dim, int input_dim) {
	auto conv1option = torch::nn::Conv2dOptions(hidden_dim + input_dim, hidden_dim, { 1,5 }).padding({ 0,2 });
	this->convz1 = torch::nn::Conv2d(conv1option);
	this->convr1 = torch::nn::Conv2d(conv1option);
	this->convq1 = torch::nn::Conv2d(conv1option);

	auto conv2option = torch::nn::Conv2dOptions(hidden_dim + input_dim, hidden_dim, { 5,1 }).padding({ 2,0 });
	this->convz2 = torch::nn::Conv2d(conv2option);
	this->convr2 = torch::nn::Conv2d(conv2option);
	this->convq2 = torch::nn::Conv2d(conv2option);

	register_module("convz1", this->convz1);
	register_module("convr1", this->convr1);
	register_module("convq1", this->convq1);
	register_module("convz2", this->convz2);
	register_module("convr2", this->convr2);
	register_module("convq2", this->convq2);
}

Tensor SepConvGRUImpl::forward(Tensor h, Tensor x) {
	auto hx = torch::cat({ h,x }, 1);
	auto z = torch::sigmoid(this->convz1->forward(hx));
	auto r = torch::sigmoid(this->convr1->forward(hx));
	auto q = torch::tanh(
		this->convq1->forward(torch::cat({ r * h, x }, 1))
	);
	h = (1 - z) * h + z * q;

	hx = torch::cat({ h,x }, 1);
	z = torch::sigmoid(this->convz2->forward(hx));
	r = torch::sigmoid(this->convr2->forward(hx));
	q = torch::tanh(
		this->convq2->forward(torch::cat({ r * h, x }, 1))
	);
	h = (1 - z) * h + z * q;
	return h;
}

FlowHeadImpl::FlowHeadImpl(int input_dim, int hidden_dim) {
	this->conv1 = torch::nn::Conv2d(conv_options(input_dim, hidden_dim, 3, 1, 1));
	this->conv2 = torch::nn::Conv2d(conv_options(hidden_dim, 2, 3, 1, 1));
	this->relu = torch::nn::ReLU(torch::nn::ReLUOptions(true));
	register_module("conv1", this->conv1);
	register_module("relu", this->relu);
	register_module("conv2", this->conv2);
}

Tensor FlowHeadImpl::forward(Tensor x) {
	x = this->conv1->forward(x);
	x = this->relu->forward(x);
	x = this->conv2->forward(x);
	return x;
}

BasicUpdateBlockImpl::BasicUpdateBlockImpl(
	int corr_levels, int corr_radius,
	int hidden_dim) {

	this->corr_levels = corr_levels;
	this->corr_radius = corr_radius;

	this->encoder = BasicMotionEncoder(this->corr_levels, this->corr_radius);
	this->gru = SepConvGRU(hidden_dim, 128 + hidden_dim);
	this->flow_head = FlowHead(hidden_dim, 256);
	this->mask->push_back(torch::nn::Conv2d(conv_options(128, 256, 3, 1, 1)));
	this->mask->push_back(torch::nn::ReLU(torch::nn::ReLUOptions(true)));
	this->mask->push_back(torch::nn::Conv2d(conv_options(256, 64 * 9, 1, 1, 0)));

	register_module("encoder", this->encoder);
	register_module("gru", this->gru);
	register_module("flow_head", this->flow_head);
	register_module("mask", this->mask);
}
tuple<Tensor, Tensor, Tensor> BasicUpdateBlockImpl::forward(
	Tensor net, Tensor inp, Tensor corr, Tensor flow) {

	auto motion_features = this->encoder->forward(flow, corr);

	inp = torch::cat({ inp, motion_features }, 1);
	net = this->gru->forward(net, inp);
	auto delta_flow = this->flow_head->forward(net);
	auto up_mask = this->mask->forward(net) * 0.25;
	return { net, up_mask, delta_flow };
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
	this->update_block = BasicUpdateBlock(this->corr_levels, this->corr_radius, this->hidden_dim);

	std::cout << "weights have been loaded for MASK-RAFT..." << std::endl;
	this->gridLength = gridLength / 8;

	register_module("fnet", this->fnet);
	register_module("cnet", this->cnet);
	register_module("update_block", this->update_block);
}

tuple<torch::Tensor, torch::Tensor> MASK_RAFTImpl::forward(
	torch::Tensor& img_t0_ten, torch::Tensor& img_t1_ten,
	vector<torch::Tensor>& Masks,
	int iters, torch::Tensor& last_flow) {

	//初始化
	auto opt = torch::TensorOptions().dtype(torch::kHalf).device(img_t0_ten.device());

	//auto image1 = cv::imread(R"(E:\codes\220318-myalgorithm\myalgorithm\res\000.png)"); //【】【】【】
	//auto image2 = cv::imread(R"(E:\codes\220318-myalgorithm\myalgorithm\res\001.png)");//【】【】【】
	//image1.convertTo(image1, CV_32FC3, 1.0F / 255.0F);//【】【】【】
	//image2.convertTo(image2, CV_32FC3, 1.0F / 255.0F);//【】【】【】
	//std::cout << image1.size() << std::endl;//【】【】【】
	//img_t0_ten = torch::from_blob(image1.data,
	//	{ 1, image1.rows, image1.cols, image1.channels() }, torch::kFloat32); //【】【】【】
	//img_t0_ten = img_t0_ten.permute({ 0,3,1,2 }).cuda();
	//img_t1_ten = torch::from_blob(image2.data,
	//	{ 1, image2.rows, image2.cols, image2.channels() }, torch::kFloat32); //【】【】【】
	//img_t1_ten = img_t1_ten.permute({ 0,3,1,2 }).cuda(); //【】【】【】

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

	int64_t h_img_fea = fmap1.size(2);
	int64_t w_img_fea = fmap1.size(3);
	int h_num = h_img_fea / this->gridLength;
	int w_num = w_img_fea / this->gridLength;

	// Mask 处理
	auto Mask_small = Masks[0];
	auto Mask_small_2 = Masks[1];
	//Mask_small = torch::ones_like(Mask_small);//【】【】【】
	//Mask_small_2 = torch::ones_like(Mask_small_2);//【】【】【】
	auto inter_opt = torch::nn::functional::InterpolateFuncOptions()
		.mode(torch::kNearest)
		.size(vector<int64_t>({ h_img_fea , w_img_fea }));
	auto Mask_big_1 = torch::nn::functional::interpolate(Mask_small, inter_opt);
	auto Mask_big_2 = torch::nn::functional::interpolate(Mask_small_2, inter_opt);

	fmap1 = fmap1 * Mask_big_1;
	fmap2 = fmap2 * Mask_big_2;


	// 计算corr
	MaskCorrBlock corr_fn = MaskCorrBlock(fmap1, fmap2,
		this->corr_levels, this->corr_radius, this->gridLength);

	//上下文
	auto cmap1 = this->cnet->forward(img_t0_ten);
	cmap1 = cmap1 * Mask_big_1;

	vector<Tensor> net_inp = torch::split(cmap1, 128, 1);
	Tensor net = torch::tanh(net_inp[0]);
	Tensor inp = torch::relu(net_inp[1]);


	//光流初始化
	auto coords_list = torch::meshgrid({
		torch::arange(h_img_fea , torch::TensorOptions().device(fmap1.device())),
		torch::arange(w_img_fea , torch::TensorOptions().device(fmap1.device())) });
	auto coords0 = torch::stack({ coords_list[1], coords_list[0] }, 0).unsqueeze(0).to(torch::kF32);
	auto coords1 = coords0.clone();
	if (last_flow.size(0) > 0) {
		coords1 = coords1 + last_flow;
	}

	//GRU循环
	auto up_mask = torch::zeros({ h_num, w_num, 576, this->gridLength, this->gridLength }).to(fmap1.device());
	for (int idx = 0; idx < iters; idx++) {
		coords1 = coords1.detach();
		auto corr = corr_fn.__call__(coords1);
		auto flow = coords1 - coords0;
		Tensor delta_flow;
		auto result = this->update_block->forward(net, inp, corr, flow);
		std::tie(net, up_mask, delta_flow) = result;

		coords1 += delta_flow;
		coords1 = Mask_big_1 * coords1 + (1 - Mask_big_1) * coords0;
	}
	last_flow = coords1 - coords0;

	//upsample
	up_mask = up_mask.view({ 1, 1, 9, 8, 8, h_img_fea, w_img_fea });
	up_mask = torch::softmax(up_mask, 2);
	auto flow_up = torch::nn::functional::unfold(
		8 * last_flow, torch::nn::functional::UnfoldFuncOptions(3).padding(1)
	);
	flow_up = flow_up.view({ 1,2,9,1,1,h_img_fea, w_img_fea });
	flow_up = torch::sum(up_mask * flow_up, 2);
	flow_up = flow_up.permute({ 0, 1, 4, 2, 5, 3 });
	flow_up = flow_up.reshape({ 1,2,8 * h_img_fea , 8 * w_img_fea });

	return { flow_up, fmap1 };
}