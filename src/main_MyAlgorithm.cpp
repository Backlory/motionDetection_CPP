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

    torch::autograd::GradMode::set_enabled(false); // 关闭梯度

    cv::VideoCapture cap;
    
    Mat img_t0, img_t1;
    Mat diffOrigin, diffWarp, moving_mask, out;
    Mat img_t0_enhancement, img_t0_arrow;
    double effect = 0;
    string alg_type = "";
    float temp_rate_1 = 0;
    Mat flo_out;
    cv::namedWindow("test", cv::WINDOW_FREERATIO);

    infer_all infer;
    string path{ R"(E:\dataset\dataset-fg-det\Janus_UAV_Dataset\train_video\video_all.mp4)" };
    path = R"(E:\dataset\dataset-fg-det\Janus_UAV_Dataset\train_video\video_all.mp4)";
    path = R"(E:\dataset\dataset-fg-det\video3.mp4)";
    while (true) {
        std::cout << "请输入视频路径：" << std::endl;
        std::cin >> path;
        if (path.empty()) {
            path = R"(E:\dataset\dataset-fg-det\Janus_UAV_Dataset\train_video\video_all.mp4)";
        }
        cap.open(path);
        if (cap.isOpened()) {
            break;
        }
        else {
            cap.release();
        }
    }

    // 初始化
    //path = get_path();
    
    
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
    for (int idx = 1; idx < framesNum-1; idx++) {
        //inference
        double t_cost = (double)cv::getTickCount();
        Mat img_t0_resized, img_t1_resized;
        cv::resize(img_t0, img_t0_resized, cv::Size(512, 512));
        cv::resize(img_t1, img_t1_resized, cv::Size(512, 512));

        infer.step(img_t0_resized, img_t1_resized,
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
        Mat outshow, outshow1, outshow2;
        cv::Mat matArray1[] = { img_t0_resized, diffWarp, moving_mask };
        cv::Mat matArray2[] = { flo_out,out, img_t0_enhancement };
        cv::hconcat(matArray1, 3, outshow1);
        cv::hconcat(matArray2, 3, outshow2);

        cv::Mat matArray[] = {outshow1, outshow2};
        cv::vconcat(matArray, 2, outshow);

        cv::imshow("test", outshow);
        cv::waitKey(1);
        
        // 保存
        img_t0.copyTo(img_t1);
        cap >> img_t0;
    }
    cv::destroyAllWindows();
    cap.release();
    return 0;
}