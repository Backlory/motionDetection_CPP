#pragma once
#include <opencv2/opencv.hpp>
#include <map>
#include <string>
#include <vector>

using std::string;
using std::vector;
using cv::Mat;


class infer_HomoSwitcher
{
public:
	infer_HomoSwitcher(std::map<string, string> *);
	~infer_HomoSwitcher();
	void inference(Mat& , Mat& ,string& , Mat& , double& ,Mat& , Mat& , Mat&);
private:
	std::map<string, string> *args;
	void infer_RANSAC(Mat& img_base, Mat& img_t1, Mat& H_warp);
	void frameDifferenceDetect(Mat& img_base_gray, Mat& img_t1_gray, Mat& img_t1_warp_gray,
		double& diffOrigin_score, double &diffWarp_score, double& effect,
		Mat& diffOrigin, Mat& diffWarp);
};

