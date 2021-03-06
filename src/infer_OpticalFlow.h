#pragma once
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/cuda.h>
#include <map>
#include <string>
#include <vector>
#include <tuple>
#include <Aten/autocast_mode.h>

#include <Module/MaskRAFT.h>

using std::string;
using std::vector;
using std::tuple;
using cv::Mat;
using torch::Tensor;



/*======================================================*/
Mat ten2img(torch::Tensor ten);
torch::Tensor img2ten(Mat img_float);
torch::Tensor readTensorFromPt(std::string filename);
string replaceParamName(string name, string x1, string x2);

class infer_OpticalFlow
{
public:
	infer_OpticalFlow(std::map<string, string>*);
	~infer_OpticalFlow();
	void inference(Mat img_t0, Mat img_t1_warp, Mat moving_mask, Tensor& flo_ten, Tensor& fmap1_ten);
private:
	std::map<string, string>* args;

	int gridLength;
	int iters;
	MASK_RAFT model{ nullptr };
	torch::DeviceType device;
	torch::nn::MaxPool2d Pool{ nullptr };
	Tensor last_flow;
};