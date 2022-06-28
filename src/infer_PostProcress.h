#pragma once
#include <opencv2/opencv.hpp>
#include <map>
#include <string>
#include <tuple>
#include <torch/torch.h>
#include "infer_OpticalFlow.h"

using std::string;
using cv::Mat;
using std::tuple;
using torch::Tensor;

class infer_PostProcress
{
public:
	infer_PostProcress() {};
	~infer_PostProcress() {};
	tuple<Mat, Mat, Mat, Mat> inference(
		const  Mat& img_t0,
		const Mat& H_warp,
		const Mat& out_last,
		const Tensor& out_ten,
		const Tensor& flo_ten
	);
};

