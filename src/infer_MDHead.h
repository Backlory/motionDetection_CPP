#pragma once
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/cuda.h>
#include <map>
#include <string>
#include <vector>
#include <tuple>
#include <Aten/autocast_mode.h>

#include "MDHead.h"
#include "infer_OpticalFlow.h"

using std::vector;
using std::tuple;
using std::string;
using cv::Mat;
using torch::Tensor;
using torch::indexing::Slice;
using torch::indexing::None;

class infer_MDHead
{
public:
	infer_MDHead(std::map<string, string>*);
	~infer_MDHead();
	Tensor inference(Tensor&, Tensor&);
private:
	std::map<string, string>* args;
	MDHead model{ nullptr };
	torch::DeviceType device;

};

