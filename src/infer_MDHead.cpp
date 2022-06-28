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
		std::cout << name << std::endl;
		auto param = params[name];
		std::cout << "sizes = " << param.sizes() << std::endl;
		std::cout << std::endl;

		torch::Tensor ten1 = readTensorFromPt("res/mdhead-weights/" + name + ".pkl");
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

		torch::Tensor ten1 = readTensorFromPt("res/mdhead-weights/" + name + ".pkl");
		assert(buffr.sizes() == ten1.sizes());
		assert(buffr.dtype() == ten1.dtype());
		assert(buffr.device() == ten1.device());
		archiveO.write(name, ten1, true);
	}
	archiveO.save_to("res/temp");
	torch::serialize::InputArchive archiveI;
	archiveI.load_from("res/temp");
	torch::NoGradGuard no_grad;
	auto params_in = this->model->named_parameters(true /*recurse*/);
	auto buffers_in = this->model->named_buffers(true /*recurse3--*/);
	for (auto& val : params_in)
		archiveI.read(val.key(), val.value(), /*is_buffer*/ false);
	for (auto& val : buffers_in)
		archiveI.read(val.key(), val.value(), /*is_buffer*/ true);


	//// 设备
	if ((*this->args)["ifUseGPU"] == "true") {
		if (torch::cuda::is_available())
			this->device = torch::kCUDA;
	}
	else {
		this->device = torch::kCPU;
	}
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