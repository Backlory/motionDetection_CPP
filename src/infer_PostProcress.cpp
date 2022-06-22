#include "infer_PostProcress.h"

infer_PostProcress::infer_PostProcress(std::map<string, string>* pargs) {
	std::cout << "infer_PostProcress initializing..." << std::endl;

	this->args = pargs;
}

infer_PostProcress::~infer_PostProcress() {
	//delete (this->args);
}