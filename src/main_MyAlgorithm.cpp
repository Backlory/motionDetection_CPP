// MyAlgorithm.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>

#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/cuda.h>
#include "infer_all.h"

string get_path() {
    string path;
    std::cout << "please input filedir, of press Enter to use default setting..." << std::endl;
    std::cin >> path;
    std::cout << "Path = " << path << std::endl;
    return path;
}

int main()
{
    std::cout << "Hello World!\n";
    std::cout << "cuDNN : " << torch::cuda::cudnn_is_available() << std::endl;
    std::cout << "CUDA : " << torch::cuda::is_available() << std::endl;
    std::cout << "Device count : " << torch::cuda::device_count() << std::endl;


    infer_all infer;
    string path{ R"(E:\dataset\dataset-fg-det\Janus_UAV_Dataset\train_video\video_all.mp4)"};
    cv::VideoCapture cap;

    Mat img_t0, img_t1;
    Mat diffOrigin, diffWarp, moving_mask, out;
    Mat img_t0_enhancement, img_t0_arrow;
    double effect = 0;
    string alg_type = "";
    float temp_rate_1 = 0;
    Mat flo_out;
    cv::namedWindow("test", cv::WINDOW_FREERATIO);

    // 初始化
    //path = get_path();
    cap.open(path);
    if (!cap.isOpened())
        return 0;

    int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int framesNum = cap.get(cv::CAP_PROP_FRAME_COUNT);
    int frameRate = cap.get(cv::CAP_PROP_FPS);
    std::cout << "视频宽度： " << width << std::endl;
    std::cout << "视频高度： " << height << std::endl;
    std::cout << "视频总帧数： " << framesNum << std::endl;
    std::cout << "帧率： " << frameRate << std::endl;
    
    // 循环
    cap >> img_t1;
    cap >> img_t0;
    int idx = 0;
    while (!img_t0.empty()) {
        //inference

        double t_cost = (double)cv::getTickCount();

        infer.step(img_t0, img_t1, 
            diffOrigin, diffWarp, moving_mask, out,
            img_t0_enhancement, img_t0_arrow,
            effect, alg_type, temp_rate_1, flo_out);

        t_cost = ((double)cv::getTickCount() - t_cost) / cv::getTickFrequency() * 1000;

        // 
        std::cout << "\r                                                                                        ";
        std::cout << "\r"
            << "[" << idx << "/" << framesNum
            << "]-" << (int)(t_cost) << " ms >>> "
            << "alg_type=" << alg_type << ", "
            << "effect=" << effect << ", "
            << "temp_rate_1=" << temp_rate_1 << ", "
            << "size=" << out.size;
        Mat outshow;
        /*cv::Mat matArray[] = {img_t0, moving_mask, flo_out, out, img_t0_enhancement};*/
        cv::Mat matArray[] = {img_t0, diffOrigin, diffWarp, moving_mask };
        cv::hconcat(matArray, 4, outshow);
        cv::imshow("test", outshow);
        cv::waitKey(1);
        
        // 保存
        img_t0.copyTo(img_t1);
        cap >> img_t0;
        idx++;
    }
    return 0;
}