#pragma once
#include <opencv2/opencv.hpp>
#include <map>
#include <string>

using std::string;
using cv::Mat;

class infer_PostProcress
{
public:
	infer_PostProcress(std::map<string, string>*);
	~infer_PostProcress();
private:
	std::map<string, string>* args;
};

