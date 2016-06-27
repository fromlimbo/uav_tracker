#ifdef UAVtracker
#else
#define UAVtracker _declspec(dllimport)
#endif
#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>
using namespace cv;
const int stateNum = 4;
const int measureNum = 2;
class UAVtracker kalman
{
	KalmanFilter KF;
	Mat measurement = Mat::zeros(measureNum, 1, CV_32F);
public:
	Point2f centerpt;
	kalman();
	void kalmaninit(Point2f& initpoint);
	void kalmanpredict(Point2f& measurept, Point2f& predictpt);
};