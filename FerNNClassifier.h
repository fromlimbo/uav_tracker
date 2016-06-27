/*
 * FerNNClassifier.h
 *
 *  Created on: Jun 14, 2011
 *      Author: alantrrs
 */
#ifdef UAVtracker
#else
#define UAVtracker _declspec(dllimport)
#endif
#include <opencv2/opencv.hpp>
#include <stdio.h>

class UAVtracker FerNNClassifier{
private:
	//下面这些参数通过程序开始运行时读入parameters.yml文件进行初始化  
  float thr_fern;//0.6
  int structSize;
  int nstructs;
  float valid;
  float ncc_thesame;
  float thr_nn;//0.65
  int acum;
public:
  //Parameters
	cv::Mat pmodel;//存放目标模型pEx的mat
  float thr_nn_valid;//当前最近邻分类器阈值的有效值，〉=前一帧的值 0.7
  bool model_flag=true;//是否添加到目标模型的标志位
  void read(const cv::FileNode& file);
  void prepare(const std::vector<cv::Size>& scales);
  void getFeatures(const cv::Mat& image,const int& scale_idx,std::vector<int>& fern);
  void update(const std::vector<int>& fern, int C, int N);
  float measure_forest(std::vector<int> fern);
  void trainF(const std::vector<std::pair<std::vector<int>,int> >& ferns,int resample);
  void trainNN(const std::vector<cv::Mat>& nn_examples, const std::vector<cv::Mat>& nn_images);
  void NNConf(const cv::Mat& example,std::vector<int>& isin,float& rsconf,float& csconf,float& msconf);
//  void NNConf1(const cv::Mat& example, std::vector<int>& isin, float& rsconf, float& csconf);
  void evaluateTh(const std::vector<std::pair<std::vector<int>,int> >& nXT,const std::vector<cv::Mat>& nExT);
  void show();
  //Ferns Members
  int getNumStructs(){return nstructs;}
  float getFernTh(){return thr_fern;}
  float getNNTh(){return thr_nn;}
  struct Feature
      {
          uchar x1, y1, x2, y2;
          Feature() : x1(0), y1(0), x2(0), y2(0) {}	//冒号后面的代表初始化
          Feature(int _x1, int _y1, int _x2, int _y2)
          : x1((uchar)_x1), y1((uchar)_y1), x2((uchar)_x2), y2((uchar)_y2)
          {}
          bool operator ()(const cv::Mat& patch) const
			  //二维单通道元素可以用Mat::at(i, j)访问，i是行序号，j是列序号  
			  //返回的patch图像片在(y1,x1)和(y2, x2)点的像素比较值，返回0或者1
          { return patch.at<uchar>(y1,x1) > patch.at<uchar>(y2, x2); }
      };
  //Ferns（蕨类植物：有根、茎、叶之分，不具花）features 特征组？ 
  std::vector<std::vector<Feature> > features; //Ferns features (one std::vector for each scale)
  std::vector< std::vector<int> > nCounter; //negative counter
  std::vector< std::vector<int> > pCounter; //positive counter
  std::vector< std::vector<float> > posteriors; //Ferns posteriors//二维vector
  float thrN; //Negative threshold
  float thrP;  //Positive thershold
  //NN Members
  std::vector<cv::Mat> pEx; //NN positive examples
  std::vector<cv::Mat> nEx; //NN negative examples
};
