#pragma once
#include <map>
#include <string>

#include<infer_HomoSwitcher.h>
#include<infer_RegionProposal.h>
#include<infer_OpticalFlow.h>
#include<infer_MDHead.h>
#include<infer_PostProcress.h>

using std::map;
using std::string;


class infer_all
{
public:
	infer_all();
	~infer_all();
	void step(Mat& , Mat& ,
		Mat&, Mat&, Mat& , Mat& ,
		Mat& , Mat& ,
		double& , string& , float& , Mat& );

private:
	map<string, string> args;

	infer_HomoSwitcher *infer_H;
	infer_RegionProposal *infer_R;
	infer_OpticalFlow *infer_O;
	infer_MDHead *infer_M;
	infer_PostProcress *infer_P;
};

