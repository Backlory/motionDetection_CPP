#include"infer_all.h"

void toc(string name, double t_cost) {
    std::cout << "["<<name<<"] cost=" << ((double)cv::getTickCount() - t_cost) / cv::getTickFrequency() * 1000 << "ms" << std::endl;
}

infer_all::infer_all() {
    this->args["ifUseGPU"] = "true";
    this->args["ifDataAugment"] = "true";

    this->args["RSHomoNet_weights"] = "model_Train_Homo_and_save_bs32_96.pkl";

    this->args["RAFT_model"] = "model/thirdparty_RAFT/model/raft-sintel.pth";
    this->args["RAFT_mixed_precision"] = "true";
    
    this->args["MDHead_weights"] = "weights/model_Train_MDHead_and_save_bs8_60.pkl";

    this->infer_H = new infer_HomoSwitcher(&args);
    this->infer_R = new infer_RegionProposal(&args);
    this->infer_O = new infer_OpticalFlow(&args);
    this->infer_M = new infer_MDHead(&args);
    this->infer_P = new infer_PostProcress();
}

infer_all::~infer_all() {
    delete this->infer_H;
    delete this->infer_R;
    delete this->infer_O;
    delete this->infer_M;
    delete this->infer_P;
}

void infer_all::step(Mat& img_t0, Mat& img_t1, 
    Mat& diffOrigin, Mat& diffWarp, Mat &moving_mask, Mat &out,
    Mat &img_t0_enhancement, Mat &img_t0_arrow,
    double& effect, string& alg_type, float& temp_rate_1, Mat& flo_out) {

    double t_cost = (double)cv::getTickCount();
    
    //
    int h = img_t0.rows;
    int w = img_t0.cols;
    h = h / 8 * 8;
    w = w / 8 * 8;
    cv::resize(img_t0, img_t0, cv::Size(w, h));
    cv::resize(img_t1, img_t1, cv::Size(w, h));
    toc("resize", t_cost);t_cost = (double)cv::getTickCount();
    //
    Mat img_t1_warp, H_warp;
    this->infer_H->inference(img_t0, img_t1, alg_type, img_t1_warp, effect, diffOrigin, diffWarp, H_warp);
    toc("infer_H", t_cost);t_cost = (double)cv::getTickCount();
    
    //
    this->infer_R->inference(img_t0, img_t1_warp, diffWarp, moving_mask);
    temp_rate_1 = cv::mean(moving_mask)[0] / 255;
    toc("infer_R", t_cost);t_cost = (double)cv::getTickCount();

    torch::Tensor flo_ten, fmap1_ten;
    this->infer_O->inference(img_t0, img_t1_warp, moving_mask, flo_ten, fmap1_ten);
    toc("infer_O", t_cost); t_cost = (double)cv::getTickCount();
    // 光流可视化，4ms，没必要
    auto flo_ten_xy = (flo_ten - flo_ten.min()) / (flo_ten.max() - flo_ten.min());
    flo_ten_xy = flo_ten_xy.detach().cpu();
    auto flo_ten_x = flo_ten_xy.index({ 0,0 });
    auto flo_ten_y = flo_ten_xy.index({ 0,1 });
    Mat flo_x_mat(h,w, CV_32FC1);
    Mat flo_y_mat(h,w, CV_32FC1);
    std::memcpy(flo_x_mat.data, flo_ten_x.data_ptr(), sizeof(float) * flo_ten_x.numel());
    std::memcpy(flo_y_mat.data, flo_ten_y.data_ptr(), sizeof(float) * flo_ten_x.numel());
    flo_x_mat.convertTo(flo_x_mat, CV_8UC1, 255);
    flo_y_mat.convertTo(flo_y_mat, CV_8UC1, 255);
    vector<Mat> flo_merge{ flo_x_mat , flo_x_mat , flo_y_mat };
    cv::merge(flo_merge, flo_out);
    toc("infer_O_viz", t_cost);t_cost = (double)cv::getTickCount();

    //
    auto out_ten = this->infer_M->inference(flo_ten, fmap1_ten);
    toc("infer_M", t_cost); t_cost = (double)cv::getTickCount();

    //
    std::tie(out, img_t0_enhancement, img_t0_arrow, out_last) = this->infer_P->inference(
        img_t0, H_warp, out_last, out_ten, flo_ten
    );

    //三通道处理
    cv::cvtColor(diffOrigin, diffOrigin, cv::COLOR_GRAY2BGR);
    cv::cvtColor(diffWarp, diffWarp, cv::COLOR_GRAY2BGR);
    cv::cvtColor(moving_mask, moving_mask, cv::COLOR_GRAY2BGR);
    assert(diffOrigin.channels() == 3);
    assert(diffWarp.channels() == 3);
    assert(moving_mask.channels() == 3);
    return;
}