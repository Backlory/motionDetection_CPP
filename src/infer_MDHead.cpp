#include "infer_MDHead.h"

/*================================================*/
/*================================================*/
/*================================================*/

infer_MDHead::infer_MDHead(std::map<string, string>* pargs) {
	std::cout << "infer_MDHead initializing..." << std::endl;
	
	this->args = pargs;
	this->model = MDHead();

	// load weights for MDHead
	torch::serialize::OutputArchive archiveO;
	auto params = this->model->named_parameters();
	for (auto name : params.keys()) {
		//std::cout << name << std::endl;
		auto param = params[name];
		//std::cout << "mdhead-param sizes = " << param.sizes() << std::endl;

		if (name.find("num_batches") == name.npos) {
			torch::Tensor ten1 = readTensorFromPt("res/mdhead-weights/" + name + ".pkl");
			assert(param.sizes() == ten1.sizes());
			assert(param.dtype() == ten1.dtype());
			assert(param.device() == ten1.device());
			archiveO.write(name, ten1, false);
			//std::cout << param.sizes() << ":" << ten1.sizes() << std::endl;
			//std::cout << param.dtype() << ":" << ten1.dtype() << std::endl;
		}
		else {
			//std::cout << "使用原始张量！" << std::endl;
			archiveO.write(name, param, false);
			//std::cout << param.dtype() << "=" << param.sizes() << std::endl;
		}
		//std::cout << std::endl;
	}
	std::cout << "-===mdhead===" << std::endl;

	auto buffrs = this->model->named_buffers();
	if (buffrs.size() > 0) {
		for (auto name : buffrs.keys()) {
			//std::cout << name << std::endl;
			auto buffr = buffrs[name];
			//std::cout << "mdhead-buffer sizes = " << buffr.sizes() << std::endl;

			if (name.find("num_batches") == name.npos) {
				torch::Tensor ten1 = readTensorFromPt("res/mdhead-weights/" + name + ".pkl");
				assert(buffr.sizes() == ten1.sizes());
				assert(buffr.dtype() == ten1.dtype());
				assert(buffr.device() == ten1.device());
				archiveO.write(name, ten1, true);
				std::cout << buffr.sizes() << ":" << ten1.sizes() << std::endl;
				std::cout << buffr.dtype() << ":" << ten1.dtype() << std::endl;
			}
			else {
				std::cout << "使用原始张量！" << std::endl;
				archiveO.write(name, buffr, true);
				std::cout << buffr.dtype() << "=" << buffr.sizes() << std::endl;

			}
			//std::cout << std::endl;

		}
	}
	archiveO.save_to("res/temp");
	torch::serialize::InputArchive archiveI;
	archiveI.load_from("res/temp");
	torch::NoGradGuard no_grad;

	std::cout << "加载参数" << std::endl;
	auto params_in = this->model->named_parameters(true /*recurse*/);
	for (auto& val : params_in) {
		//std::cout << val.key() << std::endl;
		archiveI.read(val.key(), val.value(), /*is_buffer*/ false);
	}
	auto buffers_in = this->model->named_buffers(true /*recurse3--*/);
	if (buffers_in.size() > 0) {
		for (auto& val : buffers_in) {
			//std::cout << val.key() << std::endl;
			archiveI.read(val.key(), val.value(), /*is_buffer*/ true);
		}
	}
	std::cout << "加载参数完毕" << std::endl;


	//// 设备

	std::cout << "设备" << std::endl;
	if ((*this->args)["ifUseGPU"] == "true") {
		std::cout << "检查GPU情况..." << std::endl;
		if (torch::cuda::is_available()) {
			std::cout << "GPU就绪..." << std::endl;
			this->device = torch::kCUDA;
		}
		else {
			std::cout << "GPU未就绪..." << std::endl;
			this->device = torch::kCPU;
		}
	}
	else {
		this->device = torch::kCPU;
	}
	std::cout << "送入设备：" << this->device << std::endl;
	this->model->to(this->device);
	this->model->eval();
}

infer_MDHead::~infer_MDHead() {
	//delete (this->args);
	;
}


Tensor infer_MDHead::inference(Tensor& flo_ten, Tensor& fmap1_ten) {

	Tensor out_ten = this->model->forward(flo_ten, fmap1_ten); // [1,2,640,640]
	
	// 求取最大值对应索引，两种方法
	//out_ten = torch::argmin(out_ten, 1); //-> [1,1,640,640]  好慢啊
	out_ten = out_ten * 100;
	out_ten = torch::softmax(out_ten, 1);
	out_ten = out_ten.index({ 0,0 }).unsqueeze(0).unsqueeze(0);
	
	return out_ten;
}