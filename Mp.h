#ifdef UAVtracker
#else
#define UAVtracker _declspec(dllimport)
#endif
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>
#include<iostream>
using namespace cv;
class UAVtracker Mp
{
	bool dd;//0是水平，1是垂直
	Mat stateMp;
	Mat transitionmatrix;
	float h11;
	float h1;
	float h01;
	float h0;
	Point2f lastpt;
	float lastdirection;
	float currentdirection;
	float predictdiretion;
public:
	
	Mp();
	void init(Point2f& initpt,bool d);
	void tmupdate(Point2f& currentpt);
	void tmupdate(Point2f& pt1, Point2f& pt2,Point2f& currentpt);
	float output();//0 for left,1 for right，0.5 for forbid
};
