#ifdef UAVtracker
#else
#define UAVtracker _declspec(dllimport)
#endif
#include <opencv2/opencv.hpp>
#pragma once

void UAVtracker drawBox(cv::Mat& image, CvRect box, cv::Scalar color = cvScalarAll(255), int thick=1); 

void UAVtracker drawPoints(cv::Mat& image, std::vector<cv::Point2f> points,cv::Scalar color=cv::Scalar::all(255));

cv::Mat createMask(const cv::Mat& image, CvRect box);

float UAVtracker median(std::vector<float> v);

std::vector<int> UAVtracker index_shuffle(int begin,int end);

