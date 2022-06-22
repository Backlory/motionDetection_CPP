#include "infer_OpticalFlow.h"

MASK_RAFTImpl::MASK_RAFTImpl(int gridLength, std::map<string, string>* pargs) {
	;
}

torch::Tensor MASK_RAFTImpl::forward(torch::Tensor x) {
	return x;
}



/*==============================================================================*/
/*==============================================================================*/
/*==============================================================================*/
/*==============================================================================*/

infer_OpticalFlow::infer_OpticalFlow(std::map<string, string>* pargs){//:model(16, pargs)
	
	std::cout << "infer_OpticalFlow initializing..." << std::endl;

	this->args = pargs;
	this->gridLength = 16;
	this->iters = 10;

	//// 设备与模型
	if((*this->args)["ifUseGPU"]=="true" && torch::cuda::is_available()) {
		this->device = torch::kCUDA;
	}
	else {
		this->device = torch::kCPU;
	}
	/*this->model.to(this->device);
	this->model.eval();*/
	//
	/*this->Pool = torch::nn::MaxPool2d(
		torch::nn::MaxPool2dOptions({ this->gridLength, this->gridLength })
	);*/
}

infer_OpticalFlow::~infer_OpticalFlow() {
	;
}

torch::Tensor img2ten(Mat& img) {
	Mat img_float;
	img.convertTo(img_float, CV_32FC3, 1.0F / 255.0F);
	torch::Tensor img_t0_ten = torch::from_blob(img_float.data, 
		{ 1, img_float.rows, img_float.cols, img_float.channels() }, 
		torch::kByte);
	img_t0_ten = img_t0_ten.permute({ 0,3,1,2 });
	return img_t0_ten;
}

void infer_OpticalFlow::inference(Mat& img_t0, Mat& img_t1_warp, Mat& moving_mask, 
	torch::Tensor& flo_ten, torch::Tensor& fmap1_ten) {
	//输入数据读取
	torch::Tensor img_t0_ten = img2ten(img_t0);
	torch::Tensor img_t1_ten = img2ten(img_t1_warp);
	torch::Tensor moving_mask_ten = img2ten(moving_mask);
	
	std::cout << img_t0_ten.sizes() << std::endl;
	std::cout << img_t1_ten.sizes() << std::endl;
	std::cout << moving_mask_ten.sizes() << std::endl;


	return;
}