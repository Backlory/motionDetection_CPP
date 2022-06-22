#include "infer_RegionProposal.h"


infer_RegionProposal::infer_RegionProposal(std::map<string, string>* pargs) {
	std::cout << "infer_RegionProposal initializing..." << std::endl;

	this->args = pargs;
}

infer_RegionProposal::~infer_RegionProposal() {
	//delete (this->args);
}

void infer_RegionProposal::inference(Mat& img_t0, Mat& img_t1_warp, Mat& diffWarp, Mat& moving_mask) {
    int ps = 32;
    int h_ps = ps>>1;

    cv::threshold(diffWarp, moving_mask, 3, 255, cv::THRESH_BINARY);
    cv::medianBlur(moving_mask, moving_mask, 3);

    auto structs = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::erode(moving_mask, moving_mask, structs);
    cv::erode(moving_mask, moving_mask, structs);

    this->droplittlearea(moving_mask, 50, 3, 0.2);
    //moving_mask = cv::Scalar(255) - moving_mask;
    //this->droplittlearea(moving_mask, 10000, 999, 0.2);
    //moving_mask = cv::Scalar(255) - moving_mask;
    
    /*
    // 池化获取区域最大值
    torch::Tensor moving_mask_ten = torch::from_blob(moving_mask.data, { 1,moving_mask.rows, moving_mask.cols, 1}, torch::kByte);
    moving_mask_ten = moving_mask_ten.to(torch::kCUDA);
    moving_mask_ten = moving_mask_ten.permute({ 0, 3, 1, 2 }); //转为N*C*H*W
    moving_mask_ten = moving_mask_ten.to(torch::kFloat);
    auto Pool = torch::nn::MaxPool2d(
        torch::nn::MaxPool2dOptions({ps, ps})
    );
    moving_mask_ten = Pool->forward({ moving_mask_ten }).to(torch::kCPU);// (1,1c,20h,20w)
    
    // 转回opencv的Mat
    moving_mask_ten = moving_mask_ten.squeeze(0).detach().permute({ 1, 2, 0 }); // (1c,20h,20w) -> (20h,20w,1c)
    auto moving_mask_grid = cv::Mat(moving_mask_ten.size(0), moving_mask_ten.size(1), CV_32FC(1), moving_mask_ten.data_ptr<float>());
    moving_mask_grid.convertTo(moving_mask_grid, CV_8UC1);
    cv::resize(moving_mask_grid, moving_mask_grid, moving_mask.size(), 0,0, cv::INTER_NEAREST);
    
    //展示
    Mat img_t0_gray;
    cv::cvtColor(img_t0, img_t0_gray, cv::COLOR_BGR2GRAY);
    img_t0_gray.copyTo(moving_mask_grid, moving_mask_grid);
    Mat outshow;
    assert(moving_mask.channels() == 1);
    assert(moving_mask_grid.channels() == 1);
    cv::Mat matArray[] = { moving_mask, moving_mask_grid };
    cv::hconcat(matArray, 2, outshow);
    cv::imshow("test2", outshow);
    cv::waitKey(1);*/

	return;
}

void infer_RegionProposal::droplittlearea(Mat& moving_mask, float thres_area, float thres_hmw, float thres_k) {
    Mat labels, stats, centroids;
    cv::connectedComponentsWithStats(moving_mask, labels, stats, centroids);

    bool checkpass = true;
    for (int i = 0; i < stats.rows;i++) {
        checkpass = true;

        if (stats.at<int>(i, cv::CC_STAT_AREA) < thres_area) // 如果面积比阈值小
            checkpass = false;

        //double hmw = stats.at<int>(i, cv::CC_STAT_WIDTH) / stats.at<int>(i, cv::CC_STAT_HEIGHT); 
        //if (hmw > thres_hmw || hmw < (1 / thres_hmw))  // 如果宽高比过小或过大
        //    checkpass = false;

        float area = stats.at<int>(i, cv::CC_STAT_WIDTH) * stats.at<int>(i, cv::CC_STAT_HEIGHT);
        if (stats.at<int>(i, cv::CC_STAT_AREA) < thres_k * area) //如果面积占有率太小
            checkpass = false;

        if (!checkpass) {
            moving_mask(cv::Rect(stats.at<int>(i, 0), stats.at<int>(i, 1), stats.at<int>(i, 2), stats.at<int>(i, 3))) = 0;
        }
    }
    return;
}