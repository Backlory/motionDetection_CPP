#include "infer_OpticalFlow.h"
#include <regex>


//replaceParamName(name, "123", "456")
string replaceParamName(string name, string x1, string x2) {
	auto m = name.find(x1);
	auto mm = x1.length();
	if (m != name.npos)
		name.replace(m, mm, x2);
	return name;
}

torch::Tensor img2ten(Mat img_float) {// ע�������ten��ǳ������img_float�������٣�����ָ��ͬһ���ڴ档ֻ֧��float32
	assert(img_float.type() == CV_32F);
	auto ten = torch::from_blob(img_float.data,
		{ 1, img_float.rows, img_float.cols, img_float.channels() }, torch::kFloat32);
	ten = ten.permute({ 0,3,1,2 });
	return ten;
}

Mat ten2img(torch::Tensor ten) { // 1,3,h,wתimg��ֻ֧��float32
	assert(ten.dtype() == torch::kFloat32);
	assert(ten.size(0) == 1);
	ten = ten.squeeze(0).to(torch::kCPU);
	int frame_h = ten.size(1);
	int frame_w = ten.size(2);

	if (ten.size(0) == 3) {
		cv::Mat img(frame_h, frame_w, CV_32FC3);
		std::memcpy(img.data, ten.data_ptr(), sizeof(float) * ten.numel());
		return img;
	}
	else if (ten.size(0) == 2) {
		cv::Mat img(frame_h, frame_w, CV_32FC2);
		std::memcpy(img.data, ten.data_ptr(), sizeof(float) * ten.numel());
		return img;
	}
	else if (ten.size(0) == 1) {
		cv::Mat img(frame_h, frame_w, CV_32FC1);
		std::memcpy(img.data, ten.data_ptr(), sizeof(float) * ten.numel());
		return img;
	}
	else {
		throw 0;
		return cv::Mat();
	}

}

torch::Tensor readTensorFromPt(std::string filename) {
	std::ifstream input(filename, std::ios::binary);
	if (!input) {
		std::cout << "fail to open the file " << filename << std::endl;
		throw - 1;
	}
	std::vector<char> f(
		(std::istreambuf_iterator<char>(input)),
		(std::istreambuf_iterator<char>())
	);
	input.close();
	torch::IValue x = torch::pickle_load(f);
	torch::Tensor my_tensor = x.toTensor();
	return my_tensor;
}

/*================================================================*/

infer_OpticalFlow::infer_OpticalFlow(std::map<string, string>* pargs):model(32, pargs) {
	
	std::cout << "infer_OpticalFlow initializing..." << std::endl;

	this->args = pargs;
	this->gridLength = 32;
	this->iters = 10;
	this->last_flow = torch::zeros(0);

	// Ȩ��
	// 1.��python���ֶ�������ÿ������һ���ļ�
	// 2.��C++�а��ļ�����������������ֶ�ת��OutputArchive�У������OutputArchive�ļ�
	torch::serialize::OutputArchive archiveO;
	auto params = this->model->named_parameters();
	for (auto name : params.keys()) {
		std::cout << name << std::endl;
		auto param = params[name];
		std::cout << "sizes = " << param.sizes() << std::endl;
		std::cout << std::endl;

		auto name_replaced = replaceParamName(name, "cnet.norm1.0.", "cnet.norm1.");
		name_replaced = replaceParamName(name_replaced, ".0.norm1.0.", ".0.norm1.");
		name_replaced = replaceParamName(name_replaced, ".0.norm2.0.", ".0.norm2.");
		name_replaced = replaceParamName(name_replaced, ".1.norm1.0.", ".1.norm1.");
		name_replaced = replaceParamName(name_replaced, ".1.norm2.0.", ".1.norm2.");
		torch::Tensor ten1 = readTensorFromPt("res/sintel-weights/" + name_replaced + ".pkl");
		assert(param.sizes() == ten1.sizes());
		assert(param.dtype() == ten1.dtype());
		assert(param.device() == ten1.device());
		archiveO.write(name, ten1, true);
	}
	auto buffrs = this->model->named_buffers();
	for (auto name : buffrs.keys()) {
		std::cout << name << std::endl;
		auto buffr = buffrs[name];
		std::cout << "sizes = " << buffr.sizes() << std::endl;
		std::cout << std::endl;

		auto name_replaced = replaceParamName(name, "cnet.norm1.0.", "cnet.norm1.");
		name_replaced = replaceParamName(name_replaced, ".0.norm1.0.", ".0.norm1.");
		name_replaced = replaceParamName(name_replaced, ".0.norm2.0.", ".0.norm2.");
		name_replaced = replaceParamName(name_replaced, ".1.norm1.0.", ".1.norm1.");
		name_replaced = replaceParamName(name_replaced, ".1.norm2.0.", ".1.norm2.");
		torch::Tensor ten1 = readTensorFromPt("res/sintel-weights/" + name_replaced + ".pkl");
		assert(buffr.sizes() == ten1.sizes());
		assert(buffr.dtype() == ten1.dtype());
		assert(buffr.device() == ten1.device());
		archiveO.write(name, ten1, true);
	}
	archiveO.save_to("res/temp");
	//3.��ȡArchive�ļ����������еĲ����ֶ�ת��ģ����
	torch::serialize::InputArchive archiveI;
	archiveI.load_from("res/temp");
	torch::NoGradGuard no_grad;
	auto params_in = this->model->named_parameters(true /*recurse*/);
	auto buffers_in = this->model->named_buffers(true /*recurse3--*/);
	for (auto& val : params_in)
		archiveI.read(val.key(), val.value(), /*is_buffer*/ false);
	for (auto& val : buffers_in)
		archiveI.read(val.key(), val.value(), /*is_buffer*/ true);
	//4.����ǰģ�͵Ľṹ��������������python���������
	torch::serialize::OutputArchive archiveO_model_check2;
	this->model->save(archiveO_model_check2);
	archiveO_model_check2.save_to("res/sintel-weight-cpp.pt");


	//// �豸
	if((*this->args)["ifUseGPU"]=="true") {
		if (torch::cuda::is_available())
			this->device = torch::kCUDA;
	}
	else {
		this->device = torch::kCPU;
	}
	this->model->to(this->device);
	this->model->eval();
	this->last_flow = this->last_flow.to(this->device);
	//
	this->Pool = torch::nn::MaxPool2d(
		torch::nn::MaxPool2dOptions({ this->gridLength, this->gridLength })
	);
}

infer_OpticalFlow::~infer_OpticalFlow() {
	;
}

void infer_OpticalFlow::inference(Mat img_t0, Mat img_t1_warp, Mat moving_mask, 
	torch::Tensor& flo_ten, torch::Tensor& fmap1_ten) {
	
	//�������ݶ�ȡ
	img_t0.convertTo(img_t0, CV_32FC3, 1.0F / 255.0F);
	img_t1_warp.convertTo(img_t1_warp, CV_32FC3, 1.0F / 255.0F);

	torch::Tensor img_t0_ten = img2ten(img_t0);
	torch::Tensor img_t1_ten = img2ten(img_t1_warp);

	// ���
	/*std::cout << img_t0_ten.sizes() << std::endl;
	std::cout << img_t1_ten.sizes() << std::endl;
	std::cout << "123:" << img_t0.at<cv::Vec3f>(0, 150) << std::endl;
	std::cout << img_t0_ten.index({ 0,"...",0,150 }) << std::endl;
	Mat img_t0_resunme = ten2img(img_t0_ten);
	std::cout << "456:" << img_t0.at<cv::Vec3f>(44, 150) << std::endl;
	std::cout << img_t0_resunme.at<cv::Vec3f>(44, 150) << std::endl;*/

	// mask����
	moving_mask.convertTo(moving_mask, CV_32FC1, 1.0F / 255.0F);
	torch::Tensor moving_mask_ten = img2ten(moving_mask);
	moving_mask_ten = this->Pool->forward({ moving_mask_ten });

	
	Mat moving_mask_dilated = ten2img(moving_mask_ten);
	moving_mask_dilated.convertTo(moving_mask_dilated, CV_8UC1, 255);
	auto structs = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
	cv::dilate(moving_mask_dilated, moving_mask_dilated, structs);

	moving_mask_dilated.convertTo(moving_mask_dilated, CV_32FC1, 1.0F / 255.0F);
	torch::Tensor moving_mask_dilated_ten = img2ten(moving_mask_dilated);

	vector<torch::Tensor> Masks{ moving_mask_ten.to(this->device), moving_mask_dilated_ten.to(this->device) };
	// ���
	/*std::cout << moving_mask_ten.sizes() << std::endl;
	std::cout << moving_mask_ten.dtype() << std::endl;
	std::cout << moving_mask_ten << std::endl;
	std::cout << moving_mask_dilated_ten.sizes() << std::endl;
	std::cout << moving_mask_dilated_ten.dtype() << std::endl;
	std::cout << moving_mask_dilated_ten << std::endl*/
	//��������
	img_t0_ten = img_t0_ten.to(this->device);
	img_t1_ten = img_t1_ten.to(this->device);
	
	at::autocast::set_enabled(true);
	auto result = this->model->forward(img_t0_ten, img_t1_ten, Masks, this->iters, this->last_flow);
	at::autocast::clear_cache();
	at::autocast::set_enabled(false);

	std::tie(flo_ten, fmap1_ten) = result;

	return; //flo_ten, fmap1_ten
}

