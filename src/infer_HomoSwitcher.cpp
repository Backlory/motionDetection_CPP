#include "infer_HomoSwitcher.h"



infer_HomoSwitcher::infer_HomoSwitcher(std::map<string, string>* pargs) {
	std::cout << "infer_HomoSwitcher initializing..." << std::endl;
	this->args = pargs;
}

infer_HomoSwitcher::~infer_HomoSwitcher() {
	//delete (this->args);
}

void infer_HomoSwitcher::infer_RANSAC(Mat & img_base_gray, Mat & img_t1_gray, Mat & H_warp) {
	try {
		auto orb = cv::ORB::create(200,1.200000048F,1);					//创建ORB特征点提取对象，设置提取点数
		vector<cv::KeyPoint> kpts_orb_base, kpts_orb_t1;	//存放提取的特征点
		Mat dec_orb_base, dec_orb_t1; 						//存放特征点描述子
		orb->detectAndCompute(img_t1_gray, Mat(), kpts_orb_base, dec_orb_base);
		orb->detectAndCompute(img_base_gray, Mat(), kpts_orb_t1, dec_orb_t1);
		if (kpts_orb_base.size() < 10 || kpts_orb_t1.size() < 10) {
			throw 0;
		}

		auto bf_matcher = cv::BFMatcher::create(cv::NORM_HAMMING, false);
		vector<cv::DMatch> matches_bf;
		bf_matcher->match(dec_orb_base, dec_orb_t1, matches_bf);

		//特征点筛选
		//float good_rate = 0.9f;//设置筛选率为0.5，只保留50%的匹配
		//int num_good_matchs = matches_bf.size() * good_rate;
		//std::sort(matches_bf.begin(), matches_bf.end());
		//matches_bf.erase(matches_bf.begin() + num_good_matchs, matches_bf.end());
		//Mat result_bf;
		//drawMatches(img_t1_gray, kpts_orb_base, img_base_gray, kpts_orb_t1, matches_bf, result_bf);
		//cv::imshow("123", result_bf);
		//cv::waitKey(0);
		//
		vector<cv::Point2f>points_base;
		vector<cv::Point2f>points_t1;
		for (size_t t = 0; t < matches_bf.size(); t++) {
			points_base.push_back(kpts_orb_base[matches_bf[t].queryIdx].pt);
			points_t1.push_back(kpts_orb_t1[matches_bf[t].trainIdx].pt);
		}
		//根据对应的特征点获取从demo->scene的变换矩阵
		H_warp = findHomography(points_base, points_t1, cv::RANSAC);
	}
	catch (int) {
		std::cout << "error!" << std::endl;
		H_warp = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
	}

	if (H_warp.empty()) {
		H_warp = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
	}
}

void infer_HomoSwitcher::inference(
	Mat & img_base, Mat& img_t1,
	string& alg_type, 
	Mat& img_t1_warp, double& effect,
	Mat& diffOrigin, Mat& diffWarp, Mat& H_warp) {
	
	//结果初始化
	int h = img_base.rows;
	int w = img_base.cols;
	alg_type = "None";
	effect = 0.5;
	double diffOrigin_score = 0, diffWarp_score = 0;
	H_warp = (cv::Mat_<double>(3, 3) << 1,0,0,0,1,0,0,0,1);

	Mat img_base_gray, img_t1_gray, img_t1_warp_gray;
	cv::cvtColor(img_base, img_base_gray, cv::COLOR_BGR2GRAY);
	cv::cvtColor(img_t1, img_t1_gray, cv::COLOR_BGR2GRAY);
	
	//RANSAC
	alg_type = "RANSAC";
	this->infer_RANSAC(img_base_gray, img_t1_gray, H_warp);
	cv::warpPerspective(img_t1, img_t1_warp, H_warp, cv::Size(w, h));
	cv::cvtColor(img_t1_warp, img_t1_warp_gray, cv::COLOR_BGR2GRAY);

	this->frameDifferenceDetect(img_base_gray, img_t1_gray, img_t1_warp_gray,
		diffOrigin_score, diffWarp_score, effect,
		diffOrigin, diffWarp);
	if (effect <= 0) {
		alg_type = "None";
		effect = 0;
		H_warp = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
		img_t1.copyTo(img_t1_warp);
		diffOrigin.copyTo(diffWarp);
	}

	return;
}


void infer_HomoSwitcher::frameDifferenceDetect(Mat& img_base_gray, Mat& img_t1_gray, Mat& img_t1_warp_gray,
	double& diffOrigin_score, double& diffWarp_score, double& effect,
	Mat& diffOrigin, Mat& diffWarp) {
	Mat mask;
	cv::threshold(img_t1_warp_gray, mask, 1, 1, cv::THRESH_BINARY);

	cv::absdiff(img_base_gray, img_t1_gray, diffOrigin);
	cv::absdiff(img_base_gray, img_t1_warp_gray, diffWarp);

	cv::multiply(diffOrigin, mask, diffOrigin);
	cv::multiply(diffWarp, mask, diffWarp);

	diffOrigin_score = cv::sum(diffOrigin)[0];
	diffWarp_score = cv::sum(diffWarp)[0];
	effect = 1 - (diffWarp_score + 1) / (diffOrigin_score + 1);
	return;
}