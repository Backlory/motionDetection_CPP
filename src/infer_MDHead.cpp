#include "infer_MDHead.h"

infer_MDHead::infer_MDHead(std::map<string, string>* pargs) {
	std::cout << "infer_MDHead initializing..." << std::endl;
	this->args = pargs;
}

infer_MDHead::~infer_MDHead() {
	//delete (this->args);
}