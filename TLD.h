#ifdef UAVtracker
#else
#define UAVtracker _declspec(dllimport)
#endif
#include <opencv2/opencv.hpp>
#include "tld_utils.h"
#include "LKTracker.h"
#include "FerNNClassifier.h"
#include <fstream>
#include"PatchGenerator.h"
#include<opencv2/tracking.hpp>
//#include <opencv2/legacy.hpp>
//#include<opencv2/features2d/features2d.hpp>
//#include<opencv2/opencv.hpp>
#include"kalman.h"
#include"Mp.h"
//Bounding Boxes
struct BoundingBox : public cv::Rect {
  BoundingBox(){}
  BoundingBox(cv::Rect r): cv::Rect(r){}
public:
  float overlap;        //Overlap with current Bounding Box
  int sidx;             //scale index
};

//Detection structure
struct DetStruct {
    std::vector<int> bb;
    std::vector<std::vector<int> > patt;
    std::vector<float> conf1;
    std::vector<float> conf2;
    std::vector<std::vector<int> > isin;
    std::vector<cv::Mat> patch;
  };
//Temporal structure临时结构
  struct TempStruct {
    std::vector<std::vector<int> > patt;
    std::vector<float> conf;
  };

struct OComparator{//比较两者的重合度
  OComparator(const std::vector<BoundingBox>& _grid):grid(_grid){}
  std::vector<BoundingBox> grid;
  bool operator()(int idx1,int idx2){
    return grid[idx1].overlap > grid[idx2].overlap;
  }
};
struct CComparator{//比较两者置信度
	CComparator(const std::vector<float>& _conf) :conf(_conf){}
	std::vector<float> conf;
	bool operator()(int idx1, int idx2){
		return conf[idx1]> conf[idx2];
	}
};

class UAVtracker TLD{
private:
  PatchGenerator generator;//用来对图像区域仿射变换
  FerNNClassifier classifier;
  LKTracker tracker;
  //下面这些参数通过程序开始运行时读入parameters.yml文件进行初始化  
  ///Parameters  
  int bbox_step;
  int min_win;
  int patch_size;

  //initial parameters for positive examples  
  //从第一帧得到的目标的bounding box中（文件读取或者用户框定），经过几何变换得  
  //到 num_closest_init * num_warps_init 200个正样本  
  int num_closest_init;  //最近邻窗口数 10  
  int num_warps_init;  //几何变换数目 20  
  int noise_init;
  float angle_init;
  float shift_init;
  float scale_init;

  ////从跟踪得到的目标的bounding box中，经过几何变换更新正样本（添加到在线模型？）  
  //update parameters for positive examples  
  int num_closest_update;
  int num_warps_update;
  int noise_update;
  float angle_update;
  float shift_update;
  float scale_update;

  //parameters for negative examples  
  float bad_overlap;
  float bad_patches;

  ///Variables  
  //Integral Images  积分图像，用以计算2bitBP特征（类似于haar特征的计算）  
  //Mat最大的优势跟STL很相似，都是对内存进行动态的管理，不需要之前用户手动的管理内存  
  cv::Mat iisum;
  cv::Mat iisqsum;
  cv::Mat currentframe;/*********************************/
  float var;

  //Training data  
  //std::pair主要的作用是将两个数据组合成一个数据，两个数据可以是同一类型或者不同类型。  
  //pair实质上是一个结构体，其主要的两个成员变量是first和second，这两个变量可以直接使用。  
  //在这里用来表示样本，first成员为 features 特征点数组，second成员为 labels 样本类别标签  
  std::vector<std::pair<std::vector<int>, int> > pX; //positive ferns <features,labels=1> 集合分类器 正样本  
  std::vector<std::pair<std::vector<int>, int> > nX; // negative ferns <features,labels=0> 集合分类器 负样本  
  cv::Mat pEx;  //positive NN example    最近邻分类器正样本 只有一个
  std::vector<cv::Mat> nEx; //negative NN examples  最近邻分类器负样本 多个

  //Test data   
  std::vector<std::pair<std::vector<int>, int> > nXT; //negative data to Test  
  std::vector<cv::Mat> nExT; //negative NN examples to Test  

  //Last frame data  
  BoundingBox lastbox;
  bool lastvalid;
  float lastconf;

  //Current frame data  
  //Tracker data  
  bool tracked;
  BoundingBox tbb;
  Rect2d kcfbox;
  bool tvalid;
  float tconf;
  Ptr<cv::Tracker> kcf;
  //Detector data 
  TempStruct tmp;
  DetStruct dt;
  std::vector<BoundingBox> dbb;
  std::vector<bool> dvalid;   //检测有效性？？  
  std::vector<float> dconf;  //检测确信度？？  
  bool detected;

  //Bounding Boxes
  std::vector<BoundingBox> grid;
  std::vector<BoundingBox> gridoftracker;/***************************/
  int numl = 0;
  int numr = 0;
  int numj = 0;
  std::vector<int> gridscalnum;
  std::vector<cv::Size> scales;
  std::vector<int> good_boxes; //indexes of bboxes with overlap > 0.6
  std::vector<int> bad_boxes; //indexes of bboxes with overlap < 0.2
  BoundingBox bbhull; // hull of good_boxes 窗口外的边框
  BoundingBox best_box; // maximum overlapping bbox
  //kalman滤波器
  kalman Kalman;
  //Mp滤波器
	 Mp Mp1;//水平方向
  Mp Mp2;//垂直方向
  std::vector<int> kalman_boxes;//kalman滤波确定的大概区域的图像块的编号
public:
  //Constructors
  TLD();
  TLD(const cv::FileNode& file);
  void read(const cv::FileNode& file);
  //Methods
  void init(cv::Mat& frame1,const cv::Rect &box, FILE* bb_file);
  void generatePositiveData(const cv::Mat& frame, int num_warps);
  void generateNegativeData(const cv::Mat& frame);
  void processFrame(const cv::Mat& frame,const cv::Mat& img1,const cv::Mat& img2,std::vector<cv::Point2f>& points1,std::vector<cv::Point2f>& points2,
      BoundingBox& bbnext,bool& lastboxfound, bool tl,bool tk,bool tm,FILE* bb_file);
  void track(const cv::Mat& img1, const cv::Mat& img2,std::vector<cv::Point2f>& points1,std::vector<cv::Point2f>& points2, cv::Point2f& pt1,cv::Point2f& pt2);
  void detect(const cv::Mat& frame,bool lastboxfound,bool tk,bool tm,float spdirection ,float czdirection);
  void detect1(const cv::Mat& frame,Point2f& predictpt,float spdirection,float czdirection,bool lastboxfound, bool tk, bool tm);
  void detect2(const cv::Mat& frame, Point2f& predictpt, float spdirection, float czdirection, bool lastboxfound, bool tk, bool tm);
  void detect3(const cv::Mat& frame,int detections);
  void detect4(const cv::Mat& frame,int detections);
  void clusterConf(const std::vector<BoundingBox>& dbb,const std::vector<float>& dconf,std::vector<BoundingBox>& cbb,std::vector<float>& cconf);
  void evaluate();
  void learn(const cv::Mat& img,bool tk);
  //Tools
  void buildGrid(const cv::Mat& img, const cv::Rect& box);
  void changeGrid(BoundingBox& tobb);
  float bbOverlap(const BoundingBox& box1,const BoundingBox& box2);
  void getOverlappingBoxes(const cv::Rect& box1,int num_closest);
  void getOverlappingBoxes(const cv::Rect& box1, int num_closest,bool tk);
  void getBBHull();
  void getPattern(const cv::Mat& img, cv::Mat& pattern,cv::Scalar& mean,cv::Scalar& stdev);
  void bbPoints(std::vector<cv::Point2f>& points, const BoundingBox& bb);
  void bbPredict(const std::vector<cv::Point2f>& points1,const std::vector<cv::Point2f>& points2,
  const BoundingBox& bb1,BoundingBox& bb2);
  double getVar(const BoundingBox& box,const cv::Mat& sum,const cv::Mat& sqsum);
  bool bbComp(const BoundingBox& bb1,const BoundingBox& bb2);
  int clusterBB(const std::vector<BoundingBox>& dbb,std::vector<int>& indexes);
  void trainTh(Mat&img);
  
};
int max(int a,int b);
