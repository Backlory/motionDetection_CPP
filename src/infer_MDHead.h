#pragma once
#include <opencv2/opencv.hpp>
#include <map>
#include <string>

using std::string;
using cv::Mat;

class infer_MDHead
{
public:
	infer_MDHead(std::map<string, string>*);
	~infer_MDHead();
private:
	std::map<string, string>* args;
};

