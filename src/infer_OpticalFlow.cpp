#include "infer_OpticalFlow.h"

infer_OpticalFlow::infer_OpticalFlow(std::map<string, string>* pargs):model(32, pargs) {
	
	std::cout << "infer_OpticalFlow initializing..." << std::endl;

	this->args = pargs;
	this->gridLength = 32;
	this->iters = 10;
	this->last_flow = torch::zeros({1});

	//// 设备与模型
	if((*this->args)["ifUseGPU"]=="true") {
		if (torch::cuda::is_available())
			this->device = torch::kCUDA;
	}
	else {
		this->device = torch::kCPU;
	}
	this->model->to(this->device);
	this->last_flow = this->last_flow.to(this->device);
	this->model->eval();
	//
	this->Pool = torch::nn::MaxPool2d(
		torch::nn::MaxPool2dOptions({ this->gridLength, this->gridLength })
	);
}

infer_OpticalFlow::~infer_OpticalFlow() {
	;
}

torch::Tensor img2ten(Mat img_float) {// 注意这里的ten是浅拷贝，img_float不能销毁，它俩指向同一个内存。只支持float32
	assert(img_float.type == float);
	auto ten = torch::from_blob(img_float.data,
		{ 1, img_float.rows, img_float.cols, img_float.channels()}, torch::kFloat32);
	ten = ten.permute({ 0,3,1,2 });
	return ten;
}

Mat ten2img(torch::Tensor ten) { // 1,3,h,w转img。只支持float32
	assert(ten.dtype == torch::kFloat32);
	assert(ten.size(0) == 1);
	ten = ten.squeeze(0);
	int frame_h = ten.size(1);
	int frame_w = ten.size(2);
	ten = ten.toType(torch::kF32);

	if (ten.size(0) == 3) {
		cv::Mat img (frame_h, frame_w, CV_32FC3);
		std::memcpy(img.data, ten.data_ptr(), sizeof(float) * ten.numel());
		return img;
	}
	else if (ten.size(0) == 1) {
		cv::Mat img(frame_h, frame_w, CV_32FC1);
		std::memcpy(img.data, ten.data_ptr(), sizeof(float) * ten.numel());
		return img;
	}
	else {
		throw 0;
		return cv::Mat();
	}

}

void infer_OpticalFlow::inference(Mat img_t0, Mat img_t1_warp, Mat moving_mask, 
	torch::Tensor& flo_ten, torch::Tensor& fmap1_ten) {
	
	//输入数据读取
	img_t0.convertTo(img_t0, CV_32FC3, 1.0F / 255.0F);
	img_t1_warp.convertTo(img_t1_warp, CV_32FC3, 1.0F / 255.0F);

	torch::Tensor img_t0_ten = img2ten(img_t0);
	torch::Tensor img_t1_ten = img2ten(img_t1_warp);

	// 检查
	/*std::cout << img_t0_ten.sizes() << std::endl;
	std::cout << img_t1_ten.sizes() << std::endl;
	std::cout << "123:" << img_t0.at<cv::Vec3f>(0, 150) << std::endl;
	std::cout << img_t0_ten.index({ 0,"...",0,150 }) << std::endl;
	Mat img_t0_resunme = ten2img(img_t0_ten);
	std::cout << "456:" << img_t0.at<cv::Vec3f>(44, 150) << std::endl;
	std::cout << img_t0_resunme.at<cv::Vec3f>(44, 150) << std::endl;*/

	// mask处理
	moving_mask.convertTo(moving_mask, CV_32FC1, 1.0F / 255.0F);
	torch::Tensor moving_mask_ten = img2ten(moving_mask);
	moving_mask_ten = this->Pool->forward({ moving_mask_ten });

	
	Mat moving_mask_dilated = ten2img(moving_mask_ten);
	moving_mask_dilated.convertTo(moving_mask_dilated, CV_8UC1, 255);
	auto structs = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
	cv::dilate(moving_mask_dilated, moving_mask_dilated, structs);

	moving_mask_dilated.convertTo(moving_mask_dilated, CV_32FC1, 1.0F / 255.0F);
	torch::Tensor moving_mask_dilated_ten = img2ten(moving_mask_dilated);

	vector<torch::Tensor> Masks{ moving_mask_ten.to(this->device), moving_mask_dilated_ten.to(this->device) };
	// 检查
	/*std::cout << moving_mask_ten.sizes() << std::endl;
	std::cout << moving_mask_ten.dtype() << std::endl;
	std::cout << moving_mask_ten << std::endl;
	std::cout << moving_mask_dilated_ten.sizes() << std::endl;
	std::cout << moving_mask_dilated_ten.dtype() << std::endl;
	std::cout << moving_mask_dilated_ten << std::endl*/
	//光流推理
	img_t0_ten = img_t0_ten.to(this->device);
	img_t1_ten = img_t1_ten.to(this->device);
	auto result = this->model->forward(img_t0_ten, img_t1_ten, Masks, this->iters, this->last_flow);
	std::tie(flo_ten, fmap1_ten) = result;
	return; //flo_ten, fmap1_ten
}

