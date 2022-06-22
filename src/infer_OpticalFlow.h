#pragma once
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/cuda.h>
#include <map>
#include <string>

using std::string;
using cv::Mat;


class MASK_RAFTImpl :public torch::nn::Module {
public:
	MASK_RAFTImpl(int gridLength, std::map<string, string>* pargs);
	torch::Tensor forward(torch::Tensor);
private:
	//layers
	;
};
TORCH_MODULE(MASK_RAFT);

/*======================================================*/

class infer_OpticalFlow
{
public:
	infer_OpticalFlow(std::map<string, string>*);
	~infer_OpticalFlow();
	void inference(Mat & img_t0, Mat& img_t1_warp, Mat& moving_mask, torch::Tensor& flo_ten, torch::Tensor& fmap1_ten);
private:
	std::map<string, string>* args;

	int gridLength;
	int iters;
	//MASK_RAFTImpl model;
	torch::DeviceType device;
	//torch::nn::MaxPool2d Pool;
};

