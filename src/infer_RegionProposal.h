#pragma once
#include <opencv2/opencv.hpp>
#include <map>
#include <string>
#include <torch/torch.h>
#include <torch/cuda.h>


using std::string;
using cv::Mat;

class infer_RegionProposal
{
public:
	infer_RegionProposal(std::map<string, string>*);
	~infer_RegionProposal();
	void inference(Mat &img_t0, Mat& img_t1_warp, Mat& diffWarp, Mat& moving_mask);
private:
	std::map<string, string>* args;
	void droplittlearea(Mat& moving_mask, float thres_area = 9, float thres_hmw = 3, float thres_k = 0.2);
};

