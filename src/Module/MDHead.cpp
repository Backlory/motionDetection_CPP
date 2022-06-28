#include "MDHead.h"


DownConvImpl::DownConvImpl(const int in_channels, const int out_channels, const bool pooling) {
    this->conv1 = torch::nn::Conv2d(
        torch::nn::Conv2dOptions(in_channels, out_channels, 3).padding(1).groups(1).bias(true)
    );
    this->conv2 = torch::nn::Conv2d(
        torch::nn::Conv2dOptions(out_channels, out_channels, 3).padding(1).groups(1).bias(true)
    );
    this->maxpool = torch::nn::MaxPool2d(
        torch::nn::MaxPool2dOptions(2).stride(2)
    );

    this->identity = torch::nn::Identity();
    if (pooling)
        this->pool->push_back(this->maxpool);
    else
        this->pool->push_back(this->identity);
    register_module("conv1", this->conv1);
    register_module("conv2", this->conv2);
    register_module("pool", this->pool);
}

tuple<Tensor, Tensor> DownConvImpl::forward(const Tensor& input) {
    auto x = this->conv1(input);
    x = torch::nn::functional::relu(x);
    x = this->conv2(x);
    x = torch::nn::functional::relu(x);
    auto x_pooled = this->pool->forward(x);
    return { x_pooled, x };
}
/*================================================*/

UpConvImpl::UpConvImpl(const int in_channels, const int out_channels) {
    this->upconv = torch::nn::ConvTranspose2d(
        torch::nn::ConvTranspose2dOptions(in_channels, out_channels, 2).stride(2)
    );
    this->conv1 = torch::nn::Conv2d(
        torch::nn::Conv2dOptions(2 * out_channels, out_channels, 3).padding(1).groups(1).bias(true)
    );
    this->conv2 = torch::nn::Conv2d(
        torch::nn::Conv2dOptions(out_channels, out_channels, 3).padding(1).groups(1).bias(true)
    );
    register_module("upconv", this->upconv);
    register_module("conv1", this->conv1);
    register_module("conv2", this->conv2);
}

Tensor UpConvImpl::forward(const Tensor& from_down, const Tensor& from_up) {
    auto from_up_dn = this->upconv->forward(from_up);
    auto x = torch::cat({ from_up_dn, from_down }, 1);
    x = this->conv1->forward(x);
    x = torch::nn::functional::relu(x);
    x = this->conv2->forward(x);
    x = torch::nn::functional::relu(x);
    return x;
}

/*================================================*/

MDHeadImpl::MDHeadImpl() {
    this->downConv1 = DownConv(2, 16, true);
    this->downConv2 = DownConv(16, 32, true);
    this->downConv3 = DownConv(32, 64, true);
    this->downConv4 = DownConv(64, 128, false);
    this->upConv4 = UpConv(128 + 256, 64);
    this->upConv3 = UpConv(64, 32);
    this->upConv2 = UpConv(32, 16);
    this->conv_final = torch::nn::Conv2d(
        torch::nn::Conv2dOptions(16, 2, 1).padding(0).groups(1).bias(true)
    );

    this->down_convs->push_back(downConv1);
    this->down_convs->push_back(downConv2);
    this->down_convs->push_back(downConv3);
    this->down_convs->push_back(downConv4);
    this->up_convs->push_back(upConv4);
    this->up_convs->push_back(upConv3);
    this->up_convs->push_back(upConv2);

    register_module("down_convs", this->down_convs);
    register_module("up_convs", this->up_convs);
    register_module("conv_final", this->conv_final);
}

Tensor MDHeadImpl::forward(const Tensor& flo, const Tensor& fea) {

    //光流转换
    auto flo_u = flo.index({ Slice(), Slice(0,1) });
    auto flo_v = flo.index({ Slice(), Slice(1,2) }); //y方向
    auto flo_r = torch::square(flo_u) + torch::square(flo_v);
    auto temp_max = flo_r.max();
    flo_r = flo_r.div(temp_max);
    auto flo_a = torch::arctan(flo_v.div(flo_u + 1e-5)) / 3.14159 + 0.5;
    auto flo_ra = torch::cat({ flo_r, flo_a }, /*dim*/1).detach();

    Tensor x, x_beforepool1, x_beforepool2, x_beforepool3, x_beforepool4;
    std::tie(x, x_beforepool1) = this->down_convs[0]->as<DownConv>()->forward(flo_ra);
    std::tie(x, x_beforepool2) = this->down_convs[1]->as<DownConv>()->forward(x);
    std::tie(x, x_beforepool3) = this->down_convs[2]->as<DownConv>()->forward(x);
    std::tie(x, x_beforepool4) = this->down_convs[3]->as<DownConv>()->forward(x);

    x = torch::cat({ x, fea }, 1);

    x = this->up_convs[0]->as<UpConv>()->forward(x_beforepool3, x);
    x = this->up_convs[1]->as<UpConv>()->forward(x_beforepool2, x);
    x = this->up_convs[2]->as<UpConv>()->forward(x_beforepool1, x);

    x = this->conv_final->forward(x);

    return x;
}