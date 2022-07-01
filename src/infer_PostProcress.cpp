#include "infer_PostProcress.h"

tuple<Mat, Mat, Mat, Mat> infer_PostProcress::inference(
	const  Mat& img_t0,
	const Mat& H_warp,
	const Mat& out_last,
	const Tensor & out_ten,
	const Tensor& flo_ten) {

	//输入转换为矩阵
	Mat out = ten2img(out_ten);
	out.convertTo(out, CV_8UC1, 255);
	Mat img_t0_enhancement, img_t0_arrow;
	Mat out_this = out.clone();
	
	//历史信息融合
	if (out_last.data)
		cv::bitwise_or(out, out_last, out);

	//运动区增强
	img_t0_enhancement = img_t0.clone();
	vector<Mat> channels;
	cv::split(img_t0_enhancement, channels);
	channels[0].copyTo(channels[0], (cv::Scalar(255)- out));
	channels[1].copyTo(channels[1], (cv::Scalar(255) - out));
	channels[2].copyTo(channels[2], (cv::Scalar(255) - out));
	channels[2] += out;
	cv::merge(channels, img_t0_enhancement);
	std::cout << img_t0_enhancement.type() << std::endl;
	
	//运动矢量提取
	Mat H_warp_inv, labels, stats, centroids;
	double d = cv::determinant(H_warp);
	if (d != 0) {
		cv::invert(H_warp, H_warp_inv);
	}
	else {
		H_warp_inv = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
	}
		
	img_t0_arrow = img_t0.clone();
	cv::connectedComponentsWithStats(out, labels, stats, centroids);

	for (int i = 1; i < stats.rows; i++) {
		if (stats.at<int>(i, cv::CC_STAT_AREA) > 20) {
			//查找中心点
			int x = stats.at<int>(i, cv::CC_STAT_LEFT);
			int y = stats.at<int>(i, cv::CC_STAT_TOP);
			int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
			int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
			int x_center = x + w / 2;
			int y_center = y + h / 2;
			
			//还原光流
			auto flo_point = flo_ten.index({ 0,"...", y_center , x_center });
			double v_w = -flo_point[0].to(torch::kCPU).item().toDouble(); //获取t0-t1warp光流
			double v_h = -flo_point[1].to(torch::kCPU).item().toDouble();
			Mat v_mat = (cv::Mat_<double>(3, 1) << v_w + x_center, v_h + y_center, 1); //转换t0-t1warp光流的终点坐标
			Mat v_warp_mat;
			v_warp_mat = H_warp_inv * v_mat; 

			v_warp_mat.convertTo(v_warp_mat, CV_64FC1);
			double v_w_warp = v_warp_mat.at<double>(0, 0); //提取出转换后的光流终点坐标
			double v_h_warp = v_warp_mat.at<double>(0, 1);

			//缩放光流
			
			double dx = v_w_warp - x_center;
			double dy = v_h_warp - y_center;
			
			double k = pow((pow(dx, 2) + pow(dy, 2)), 0.5);
			if (k < 10.0F) {
				dx = 0;
				dy = 0;
			}
			
			//箭头
			cv::Point p1 = cv::Point(x,y);
			cv::Point p2 = cv::Point(x+w,y+h);
			cv::Point p3 = cv::Point(x_center, y_center);
			cv::Point p4 = cv::Point(x_center + dx, y_center + dy);
			std::cout << p1 << std::endl;
			std::cout << p2 << std::endl;
			std::cout << p3 << std::endl;
			std::cout << p4 << std::endl;
			cv::rectangle(img_t0_arrow,         p1, p2, cv::Scalar(0, 0, 255), 2);
			cv::arrowedLine(img_t0_arrow,       p3, p4, cv::Scalar(0, 0, 0), 2, 8, 0, 0.2);
			cv::arrowedLine(img_t0_enhancement, p3, p4, cv::Scalar(0, 0, 0), 2, 8, 0, 0.2);
		}
	}
	

	//三通道转换
	cv::cvtColor(out, out, cv::COLOR_GRAY2BGR);

	return{ out, img_t0_enhancement, img_t0_arrow, out_this };
}