/*
 * TLD.cpp
 *
 *  Created on: Jun 9, 2011
 *      Author: alantrrs
 */
#define UAVtracker _declspec(dllexport)
#include "TLD.h"
#include <stdio.h>
#include<time.h>
using namespace cv;
using namespace std;
//float SCALES[] = { 0.57870, 0.69444, 0.83333, 1, 1.20000, 1.44000, 1.72800 };
float SCALES[] = { 0.16151, 0.19381, 0.23257, 0.27908, 0.33490, 0.40188, 0.48225,
0.57870, 0.69444, 0.83333, 1, 1.20000, 1.44000, 1.72800,
2.07360, 2.48832, 2.98598, 3.58318, 4.29982, 5.15978, 6.19174 }; 
//float SCALES[] = { 0.33490, 0.40188, 0.48225,
//0.57870, 0.69444, 0.83333, 1, 1.20000, 1.44000, 1.72800,
//2.07360, 2.48832, 2.98598, 3.58318, 4.29982};
TLD::TLD()
{
}
TLD::TLD(const FileNode& file){
  read(file);
}

void TLD::read(const FileNode& file){
  ///Bounding Box Parameters
  min_win = (int)file["min_win"];//15
  ///Genarator Parameters
  //initial parameters for positive examples
  patch_size = (int)file["patch_size"];//15
  num_closest_init = (int)file["num_closest_init"];//10
  num_warps_init = (int)file["num_warps_init"];//20
  noise_init = (int)file["noise_init"];//5
  angle_init = (float)file["angle_init"];//20
  shift_init = (float)file["shift_init"];//0.02
  scale_init = (float)file["scale_init"];//0.02
  //update parameters for positive examples
  num_closest_update = (int)file["num_closest_update"];//10
  num_warps_update = (int)file["num_warps_update"];//10
  noise_update = (int)file["noise_update"];//5
  angle_update = (float)file["angle_update"];//10
  shift_update = (float)file["shift_update"];//0.02
  scale_update = (float)file["scale_update"];//0.02
  //parameters for negative examples
  bad_overlap = (float)file["overlap"];//0.2
  bad_patches = (int)file["num_patches"];//100 x:288 y:36 w:25 h:42
  classifier.read(file);
}

void TLD::init( Mat& frame1,const Rect& box,FILE* bb_file){
	kcf = Tracker::create("KCF");
	kcfbox = Rect2d(box.x, box.y, box.width, box.height);
	kcf->init(frame1, kcfbox);
	cvtColor(frame1, frame1, CV_RGB2GRAY);
  //bb_file = fopen("bounding_boxes.txt","w");
  //Get Bounding Boxes
	//此函数根据传入的box（目标边界框）在传入的图像frame1中构建全部的扫描窗口，并计算重叠度，重叠度定义为两个box的交集与并集之比
	clock_t t1 = clock();
    buildGrid(frame1,box);
	clock_t t2 = clock();
	//cout << "创建grid耗时" << t2 - t1 << "ms" << endl;
   // printf("Created %d bounding boxes\n",(int)grid.size());//vector的成员size()用于获取向量元素的个数  
	/*for (int i = 0; i < gridscalnum.size(); i++)
		cout << "第" << i + 1 << "种尺度个数=" << gridscalnum[i] << endl;*/
	frame1.copyTo(currentframe);
  ///Preparation
  //allocation
 //积分图像，用以计算2bitBP特征（类似于haar特征的计算）  
 //Mat的创建，方式有两种：1.调用create（行，列，类型）2.Mat（行，列，类型（值））。
  iisum.create(frame1.rows+1,frame1.cols+1,CV_32F);//float类型
  iisqsum.create(frame1.rows+1,frame1.cols+1,CV_64F);//double类型
  //Detector data中定义：std::vector<float> dconf; 
  //vector 的reserve增加了vector的capacity，但是它的size没有改变！而resize改变了vector  
  //的capacity同时也增加了它的size！reserve是容器预留空间，但在空间内不真正创建元素对象，  
  //所以在没有添加新的对象之前，不能引用容器内的元素。  
  //不管是调用resize还是reserve，二者对容器原有的元素都没有影响。  
  //myVec.reserve( 100 );     // 新元素还没有构造, 此时不能用[]访问元素  
  //myVec.resize( 100 );      // 用元素的默认构造函数构造了100个新的元素，可以直接操作新元素  
  dconf.reserve(100);  //确信度
  dbb.reserve(100);	//有效性
  bbox_step =7;

  // 以下在Detector data中定义的容器都给其分配grid.size()大小（这个是一幅图像中全部的扫描窗口个数）的容量
  //Detector data中定义TempStruct tmp;    
  //tmp.conf.reserve(grid.size()); 
  tmp.conf = vector<float>(grid.size());	//
  tmp.patt = vector<vector<int> >(grid.size(),vector<int>(10,0));
  //tmp.patt.reserve(grid.size());
  dt.bb.reserve(grid.size());
  good_boxes.reserve(grid.size());
  bad_boxes.reserve(grid.size());
  kalman_boxes.reserve(grid.size());

  //TLD中定义：cv::Mat pEx;  //positive NN example 大小为15*15图像片  
  pEx.create(patch_size,patch_size,CV_64F);

  //Init Generator
  //TLD中定义：cv::PatchGenerator generator;  //PatchGenerator类用来对图像区域进行仿射变换  
  /*
  cv::PatchGenerator::PatchGenerator (
  double     _backgroundMin,
  double     _backgroundMax,
  double     _noiseRange,
  bool     _randomBlur = true,
  double     _lambdaMin = 0.6,
  double     _lambdaMax = 1.5,
  double     _thetaMin = -CV_PI,
  double     _thetaMax = CV_PI,
  double     _phiMin = -CV_PI,
  double     _phiMax = CV_PI
  )
  一般的用法是先初始化一个PatchGenerator的实例，然后RNG一个随机因子，再调用（）运算符产生一个变换后的正样本。
  */
  //]
  generator = PatchGenerator (0,0,noise_init,true,1-scale_init,1+scale_init,-angle_init*CV_PI/180,angle_init*CV_PI/180,-angle_init*CV_PI/180,angle_init*CV_PI/180);

  //此函数根据传入的box（目标边界框），在整帧图像中的全部窗口中寻找与该box距离最小（即最相似，  
  //重叠度最大）的num_closest_init个窗口，然后把这些窗口 归入good_boxes容器  
  //同时，把重叠度小于0.2的，归入 bad_boxes 容器  
  //首先根据overlap的比例信息选出重复区域比例大于60%并且前num_closet_init= 10个的最接近box的RectBox，  
  //相当于对RectBox进行筛选。并通过BBhull函数得到这些RectBox的最大边界。  
   t1 = clock();
  getOverlappingBoxes(box, num_closest_init);
   t2 = clock();
  //cout << "Overlapping       time            is     " << t2 - t1 << "ms!!!!!" << endl;
  //初始化时永远都是10个good_boxes//存的是它们在60000多个里的序号

  // printf("Found %d good boxes, %d bad boxes\n",(int)good_boxes.size(),(int)bad_boxes.size());
 // printf("Best Box: %d %d %d %d\n",best_box.x,best_box.y,best_box.width,best_box.height);
  //printf("Bounding box hull: %d %d %d %d\n",bbhull.x,bbhull.y,bbhull.width,bbhull.height);
  //Correct Bounding Box
  lastbox=best_box;
  lastconf=1;
  lastvalid=true;
  //Print
 // fprintf(bb_file,"%d,%d,%d,%d,%f\n",lastbox.x,lastbox.y,lastbox.br().x,lastbox.br().y,lastconf);
  //Prepare Classifier
  //scales容器里是所有扫描窗口的尺度，由buildGrid()函数初始化  
  classifier.prepare(scales);

  ///Generate Data
  // Generate positive data
  //从good_boxes里生成10*20个仿射正样本
  generatePositiveData(frame1,num_warps_init);

  // Set variance threshold
  Scalar stdev, mean;//标准差、均值
  //统计best_box的均值和标准差  
  ////例如需要提取图像A的某个ROI（感兴趣区域，由矩形框）的话，用Mat类的B=img(ROI)即可提取  
  //frame1(best_box)就表示在frame1中提取best_box区域（目标区域）的图像片  
  meanStdDev(frame1(best_box),mean,stdev);

  //利用积分图像去计算每个待检测窗口的方差  
  //cvIntegral( const CvArr* image, CvArr* sum, CvArr* sqsum=NULL, CvArr* tilted_sum=NULL );  
  //计算积分图像，输入图像，sum积分图像, W+1×H+1，sqsum对象素值平方的积分图像，tilted_sum旋转45度的积分图像  
  //利用积分图像，可以计算在某象素的上－右方的或者旋转的矩形区域中进行求和、求均值以及标准方差的计算，  
  //并且保证运算的复杂度为O(1)。  
  integral(frame1,iisum,iisqsum);

  //级联分类器模块一：方差检测模块
  //利用积分图计算每个待检测窗口的方差，方差小于var阈值（目标patch方差的50%）的，  
  //则认为其含有前景目标方差；var 为标准差的平方  
  var = pow(stdev.val[0],2)*0.5; //getVar(best_box,iisum,iisqsum);best_box的标准差 *0.5
 // cout << "variance: " << var << endl;
  //check variance
  //getVar函数通过积分图像计算输入的best_box的方差  
  double vr =  getVar(best_box,iisum,iisqsum)*0.5;//固定算法，照抄吧
  //cout << "check variance: " << vr << endl;
  //它俩理论上应该相等


  // Generate negative data
  generateNegativeData(frame1);
  //Split Negative Ferns into Training and Testing sets (they are already shuffled)
  int half = (int)nX.size()*0.5f;//f代表浮点型
  //vector::assign函数将区间[start, end)中的值赋值给当前的vector.  
  //将一半的负样本集 作为 测试集  
  nXT.assign(nX.begin()+half,nX.end());//nXT：集合分类器的负样本测试集
  nX.resize(half);                     //nX: 集合分类器的负样本（训练集）
  ///Split Negative NN Examples into Training and Testing sets
  //将负样本放进训练集和测试集  
  half = (int)nEx.size()*0.5f;
  nExT.assign(nEx.begin()+half,nEx.end());//nExT:最近邻分类器的负样本测试集
  nEx.resize(half);                       //nExT:最近邻分类器的负样本（训练集）

  //Merge Negative Data with Positive Data and shuffle it
  //合并正负样本并混乱
  vector<pair<vector<int>,int> > ferns_data(nX.size()+pX.size());
  vector<int> idx = index_shuffle(0,ferns_data.size());
  int a=0;
  for (int i=0;i<pX.size();i++){
      ferns_data[idx[a]] = pX[i];
      a++;
  }
  for (int i=0;i<nX.size();i++){
      ferns_data[idx[a]] = nX[i];
      a++;
  }

  //Data already have been shuffled, just putting it in the same vector
  vector<cv::Mat> nn_data(nEx.size()+1);//pEx=1
  vector<cv::Mat> nn_image(nEx.size() + 1);//pEx=1
  nn_data[0] = pEx;
  nn_image[0] = frame1(lastbox);
  for (int i=0;i<nEx.size();i++){
      nn_data[i+1]= nEx[i];
		 int idx = bad_boxes[i];
	nn_image[i+1] = frame1(grid[idx]);
  }
  ///Training
  t1 = clock();
  classifier.trainF(ferns_data,2); //bootstrap = 2
  t2 = clock();
  //cout << "训练Fern分类器用了" << t2 - t1 << "ms" << endl;
  classifier.trainNN(nn_data,nn_image);
  ///Threshold Evaluation on testing sets
  classifier.evaluateTh(nXT,nExT);
  //kalman滤波器的初始化
  Point2f initpoint;
  initpoint.x = (float)lastbox.x + (float)lastbox.width / 2.f;
  initpoint.y = (float)lastbox.y + (float)lastbox.height / 2.f;
  //cout << initpoint.x <<endl<< initpoint.y << endl;
  Kalman.kalmaninit(initpoint);
  Mp1.init(initpoint, 0);
  Mp2.init(initpoint, 1);

}

/* Generate Positive data
 * Inputs:
 * - good_boxes (bbP)
 * - best_box (bbP0)
 * - frame (im0)
 * Outputs:
 * - Positive fern features (pX)
 * - Positive NN examples (pEx)
 */
void TLD::generatePositiveData(const Mat& frame, int num_warps){
	/*
	CvScalar定义可存放1—4个数值的数值，常用来存储像素，其结构体如下：
	typedef struct CvScalar
	{
	double val[4];
	}CvScalar;
	如果使用的图像是1通道的，则s.val[0]中存储数据
	如果使用的图像是3通道的，则s.val[0]，s.val[1]，s.val[2]中存储数据
	*/
  Scalar mean;
  Scalar stdev;
  //此函数将frame图像best_box区域的图像片归一化为均值为0的15*15大小的patch，存在pEx正样本中，pEX是为了后面的最近邻分类器做准备的  
  getPattern(frame(lastbox),pEx,mean,stdev);
  //Get Fern features on warped patches
  Mat img;
  Mat warped;//弯曲 
  GaussianBlur(frame,img,Size(9,9),1.5);

  //在img图像中截取bbhull信息（bbhull是包含了位置和大小的矩形框）的图像赋给warped  
  //例如需要提取图像A的某个ROI（感兴趣区域，由矩形框）的话，用Mat类的B=img(ROI)即可提取  
  warped = img(bbhull);//warped是img bbhull区域的浅拷贝，它俩是共享存储空间的，于是warped一变，img也就会变。
  RNG& rng = theRNG();
  Point2f pt(bbhull.x + (bbhull.width - 1)*0.5f, bbhull.y + (bbhull.height - 1)*0.5f); //取矩形包围框中心的坐标  int i(2) 
  
  //nstructs树木（由一个特征组构建，每组特征代表图像块的不同视图表示）的个数  
  //fern[nstructs] nstructs棵树的森林的数组？？  
  vector<int> fern(classifier.getNumStructs());
  pX.clear();
  Mat patch;
  //pX为处理后的RectBox最大边界处理后的像素信息，pEx最近邻的RectBox的Pattern，bbP0为最近邻的RectBox。  
  if (pX.capacity()<num_warps*good_boxes.size())//20*10
    pX.reserve(num_warps*good_boxes.size());//pX正样本个数为 仿射变换个数 * good_box的个数，故需分配至少这么大的空间  200
  int idx;
  clock_t t1 = clock();
  for (int i=0;i<num_warps;i++)//20种几何变换 //我已改成10种
  {
	  
	  if (i>0)
	  {
		  generator(frame, pt, warped, bbhull.size(), rng); //PatchGenerator类用来对图像区域进行仿射变换，先RNG一个随机因子，再调用（）运算符产生一个变换后的正样本。   //对frame的选择区域进行仿射变换，将仿射变换结果保存到warped。
		/* imshow("1", warped);
		  waitKey(0);*/
	  }
		
	  for (int b = 0; b < good_boxes.size(); b++)
	  {
		  idx = good_boxes[b];//good_boxes容器保存的是 grid 的索引 
		  //Rect region(grid[idx].x-bbhull.x,grid[idx].y-bbhull.y,grid[idx].width,grid[idx].height);
		  patch = img(grid[idx]);//把img的 grid[idx] 区域（也就是bounding box重叠度高的）这一块图像片提取出来  
		  //getFeatures函数得到输入的patch的用于树的节点，也就是特征组的特征fern（13位的二进制代码）  
		  classifier.getFeatures(patch, grid[idx].sidx, fern);//把10个good_boxes的图像输入 得到相应尺度box覆盖的图像的 随机两个点的真假特征值给fern，一共有10株fern树，每株树里的13个特征值由一个十进制数值代替13位的二进制数来表示。
		  pX.push_back(make_pair(fern, 1));//这样就得到一个good_box正样本的10株*13个特征点对的特征值，这样的正样本要来200个！
		  /*cout << "第" << b << "box" << "的";
		  for (int j = 0; j < pX[b].first.size();j++)
		  cout <<"第"<<j<<"棵树的"<<"特征值"<<hex<< pX[b].first[j]<<" ";
		  cout << endl;*/
	  }
	 /* cout << endl;*/
	   //generator(warped, Point2f(warped.rows*0.5f, warped.cols*0.5f), warped, bbhull.size(), rng);
  }
  clock_t t2 = clock();
 // cout << "10种几何变换时间为" << t2 - t1 << "ms" << endl;

  //printf("%d个good_boxes生成了随机森林正样本%d个\n", good_boxes.size(),(int)pX.size());
}

//先对最接近box的RectBox区域得到其patch ,然后将像素信息转换为Pattern，  
//具体的说就是归一化RectBox对应的patch的size（放缩至patch_size = 15*15），将2维的矩阵变成一维的向量信息，  
//然后将向量信息均值设为0，调整为zero mean and unit variance（ZMUV）  
//Output: resized Zero-Mean patch  
void TLD::getPattern(const Mat& img, Mat& pattern,Scalar& mean,Scalar& stdev){
  //Output: resized Zero-Mean patch
  resize(img,pattern,Size(patch_size,patch_size));//将img放缩至patch_size = 15*15，存到pattern中  
  //计算这个15*15矩阵的均值和标准差
  meanStdDev(pattern,mean,stdev);
  pattern.convertTo(pattern,CV_32F);
  pattern = pattern-mean.val[0];//opencv中Mat的运算符有重载， Mat可以 + Mat; + Scalar; + int / float / double 都可以  
  //将矩阵所有元素减去其均值，也就是把patch的均值设为零  
 
}

void TLD::generateNegativeData(const Mat& frame){
/* Inputs:
 * - Image
 * - bad_boxes (Boxes far from the bounding box)
 * - variance (pEx variance)
 * Outputs
 * - Negative fern features (nX)
 * - Negative NN examples (nEx)
 */
	//由于之前重叠度小于0.2的，都归入 bad_boxes了，所以数量挺多，下面的函数用于打乱顺序，也就是为了  
	//后面随机选择bad_boxes  
  random_shuffle(bad_boxes.begin(),bad_boxes.end());//Random shuffle bad_boxes indexes
  int idx;
  //Get Fern Features of the boxes with big variance (calculated using integral images)
  int a=0;//负样本数
  //int num = std::min((int)bad_boxes.size(),(int)bad_patches*100); //limits the size of bad_boxes to try
  //printf("negative data generation started.\n");
  vector<int> fern(classifier.getNumStructs());
  nX.reserve(bad_boxes.size());
  Mat patch;
  for (int j=0;j<bad_boxes.size()*0.1;j++)
  {
      idx = bad_boxes[j];
	  if (getVar(grid[idx], iisum, iisqsum)<var*0.5f) //把方差较大的bad_boxes加入负样本 
            continue;
      patch =  frame(grid[idx]);
	  classifier.getFeatures(patch,grid[idx].sidx,fern);
      nX.push_back(make_pair(fern,0));//得到负样本  
      a++;
  }
  //printf("%d个bad_boxes前十分之一生成了随机森林（蕨）负样本%d个\n ", bad_boxes.size(),a);
  //random_shuffle(bad_boxes.begin(),bad_boxes.begin()+bad_patches);//Randomly selects 'bad_patches' and get the patterns for NN;
  Scalar dum1, dum2;
  nEx=vector<Mat>(bad_patches);//100
  for (int i=0;i<bad_patches;i++)
  {
      idx=bad_boxes[i];
	  patch = frame(grid[idx]);
	  //具体的说就是归一化RectBox对应的patch的size（放缩至patch_size = 15*15）  
	  //由于负样本不需要均值和方差，所以就定义dum，将其舍弃  
      getPattern(patch,nEx[i],dum1,dum2);
  }
  //printf("最近邻分类器负样本: %d个\n",(int)nEx.size());
}

//该函数通过积分图像计算输入的box的方差 
double TLD::getVar(const BoundingBox& box,const Mat& sum,const Mat& sqsum)
{
  double brs = sum.at<int>(box.y+box.height,box.x+box.width);
  double bls = sum.at<int>(box.y+box.height,box.x);
  double trs = sum.at<int>(box.y,box.x+box.width);
  double tls = sum.at<int>(box.y,box.x);
  double brsq = sqsum.at<double>(box.y+box.height,box.x+box.width);
  double blsq = sqsum.at<double>(box.y+box.height,box.x);
  double trsq = sqsum.at<double>(box.y,box.x+box.width);
  double tlsq = sqsum.at<double>(box.y,box.x);
  double mean = (brs+tls-trs-bls)/((double)box.area());
  double sqmean = (brsq+tlsq-trsq-blsq)/((double)box.area());
  return sqmean-mean*mean;
}

void TLD::processFrame(const cv::Mat& frame, const cv::Mat& img1, const cv::Mat& img2, vector<Point2f>& points1, vector<Point2f>& points2, BoundingBox& bbnext, bool& lastboxfound, bool tl, bool tk, bool tm, FILE* bb_file)
{
	vector<BoundingBox>cbb;//聚类之后的bounding box  
	vector<float>cconf;
	int confident_detections = 0;//小D的结果聚类之后，分数比小T高的数目  
	int didx; //detection index  
	/// 1.Track  
	Point2f trackpt1 = { 0 };//前后向最好跟踪点，用以更新马尔可夫模型
	Point2f trackpt2 = { 0 };
	if (lastboxfound&&tl)//前一帧目标出现过，我们才跟踪，否则只能检测了  
	{
		clock_t t1 = clock();
		
		//Rect2d tbb1 = Rect2d(tbb.x, tbb.y, tbb.width, tbb.height);
		if (tm)
		{
			tracked = kcf->update(frame, kcfbox);
			if (tracked)
			{
				//kcfbox = kcfbox&Rect2d(0, 0, img2.rows, img2.cols);
				tbb.x = kcfbox.x;
				tbb.y = kcfbox.y;
				tbb.width = kcfbox.width;
				tbb.height = kcfbox.height;

				Mat pattern;
				Scalar mean, stdev;
				//tbb = tbb&Rect(0, 0, img2.rows, img2.cols);
				tbb.x = max(tbb.x, 0);
				tbb.y = max(tbb.y, 0);
				tbb.width = min(min(img2.cols - tbb.x, tbb.width), min(tbb.width, tbb.br().x));
				tbb.height = min(min(img2.rows - tbb.y, tbb.height), min(tbb.height, tbb.br().y));
				if (tbb.width < 5 || tbb.height < 5)
				{
					tracked = false;

				}
				else
				{
					//Mat track = img2(tbb);
					getPattern(img2(tbb), pattern, mean, stdev);
					vector<int> isin;
					float dummy, conf;
					//计算图像片pattern到在线模型M的保守相似度  
					classifier.NNConf(pattern, isin, conf, dummy, tconf); //Conservative Similarity tconf是用csconf 用最近邻分类器的Conservative Similarity【5.2】作为跟踪目标的得分即可，后面要用这个得分和检测器进行比较。
					tvalid = lastvalid;
					//保守相似度大于阈值，则评估跟踪有效 
					cout << tconf << endl;
					if (tconf > 0.5)
						tvalid = true;//判定轨迹是否有效，从而决定是否要增加正样本，标志位tvalid【论文5.6.2 P-Expert】 
					if (tconf < 0.5)
						tvalid = false;
					/*if (tconf < 0.35)
						tracked = false;*/
					clock_t t2 = clock();
				}
				//cout << "track用了" << t2 - t1 << "ms" << endl;
			}
			//cout << tvalid << endl;
		}
		else
		{
			track(img1, img2, points1, points2, trackpt1, trackpt2);//网格均匀撒点（均匀采样），在lastbox中共产生最多10*10=100个特征点，存于points1
		}
	}
	
	else //lastfound=false
	{
		tracked = false;
			//tvalid = false;
	}
	
		///Detect
		/*if (tracked){
			printf("%f\t%f\n", trackpt1.x, trackpt1.y);
			printf("%f\t%f\n", trackpt2.x, trackpt2.y);
			}*/
		float spdirection = 1.f;
		float czdirection = 1.f;
		//cout << tm<<endl;
		//cout << tracked<<endl;
		/*if (tm&&tracked)
		{
			if (trackpt1.x < trackpt2.x)
				spdirection = 1.f;
			else if (trackpt1.x > trackpt2.x)
				spdirection = 0.f;
			else
				spdirection = 0.5;
			if (trackpt1.y < trackpt2.y)
				czdirection = 1.f;
			else if (trackpt1.y > trackpt2.y)
				czdirection = 0.f;
			else
				czdirection = 0.5f;
		}
		if (tm&&!tracked)
		{
			spdirection = Mp1.output();
			czdirection = Mp2.output();
		}*/
		clock_t t1 = clock(); 
			detect(img2, lastboxfound, tk, tm, spdirection, czdirection);
			if (tconf > 0.55&&tracked == true)
				detected = false;

	/*	if (tconf < 0.5 || tracked ==false)
		detect(img2, lastboxfound, tk, tm, spdirection, czdirection);
		else
		{
			detected = false;
		}
		*/
		clock_t t2 = clock();
	
		//cout << "detect用了" << t2 - t1 << "ms" << endl;
		///Integration 
		///Integration   综合模块  
		//TLD只跟踪单目标，所以综合模块综合跟踪器跟踪到的单个目标和检测器检测到的多个目标，然后只输出保守相似度最大的一个目标  
		t1 = clock();
		if (tracked)
		{
			bbnext = tbb;
			lastconf = tconf;//表示相关相似度的阈值
			lastvalid = tvalid;
			// printf("Tracked\n");
			if (detected)
			{ //通过 重叠度 对检测器检测到的目标bounding box进行聚类，每个类其重叠度小于0.5 //   if Detected
				clusterConf(dbb, dconf, cbb, cconf);                       //   cluster detections
				//  printf("找到 %d 个聚类\n", (int)cbb.size());
				for (int i = 0; i < cbb.size(); i++)
				{    //找到与跟踪器跟踪到的box距离比较远的类（检测器检测到的box），而且它的相关相似度比跟踪器的要大  
					if (bbOverlap(tbb, cbb[i]) < 0.8 && cconf[i] > tconf){  //  Get index of a clusters that is far from tracker and are more confident than the tracker
						confident_detections++;
						didx = i; //detection index
						bbnext = cbb[i];
						tconf = cconf[i];
					}
				}
				cout << "confident_detections="<<confident_detections << endl;
				lastconf = tconf;
				if (confident_detections == 1)
				{

						kcfbox = Rect2d(bbnext.x, bbnext.y, bbnext.width, bbnext.height);
							kcf = Tracker::create("KCF");
							kcf->init(frame, kcfbox);
							cout << "reinintialize tracker" << endl;
							lastvalid = true;
				}
				//如果只有一个满足上述条件的box，那么就用这个目标box来重新初始化跟踪器（也就是用检测器的结果去纠正跟踪器）

				//if (confident_detections == 1)//跟踪到了但没跟好
				//{                                //if there is ONE such a cluster, re-initialize the tracker
				//	printf("Found a better match..reinitializing tracking\n");
				//	bbnext = cbb[didx];
				//	if (tm)
				//	{
				//		float conf;
				//		Mat pattern;
				//		Scalar mean, stdev;
				//		bbnext = bbnext&Rect(0, 0, img2.cols, img2.rows);
				//		getPattern(img2(bbnext), pattern, mean, stdev);
				//		vector<int> isin;
				//		float dummy;
				//		//计算图像片pattern到在线模型M的保守相似度  
				//		classifier.NNConf(pattern, isin, dummy, conf); //Conservative Similarity tconf是用csconf 用最近邻分类器的Conservative Similarity【5.2】作为跟踪目标的得分即可，后面要用这个得分和检测器进行比较
				//		//保守相似度大于阈值，则评估跟踪有效 
				//		if (conf > classifier.thr_nn_valid)//重置的窗口和目标模型相似
				//		{
				//			lastboxfound = true;//判定轨迹是否有效，从而决定是否要增加正样本，标志位tvalid【论文5.6.2 P-Expert】 

				//		}
				//		else
				//		{
				//			lastboxfound = true;
				//			//bbnext = Rect((int)Kalman.centerpt.x, (int)Kalman.centerpt.y, lastbox.width, lastbox.height);
				//			bbnext = lastbox;
				//		}

				//		kcfbox = Rect2d(bbnext.x, bbnext.y, bbnext.width, bbnext.height);
				//		kcf = Tracker::create("KCF");
				//		kcf->init(frame, kcfbox);
				//	}
				//	lastconf = cconf[didx];
				//	lastvalid = false;
		
				//}
				//else//confident_detections != 1
				//{
				//	// printf("找到%d个置信度较高的聚类 \n", confident_detections);
				//	int cx = 0, cy = 0, cw = 0, ch = 0;
				//	int close_detections = 0;
				//	for (int i = 0; i < dbb.size(); i++)
				//	{ //找到检测器检测到的box与跟踪器预测到的box距离很近（重叠度大于0.7）的box，对其坐标和大小进行累加  
				//		if (bbOverlap(tbb, dbb[i]) > 0.7)
				//		{ // Get mean of close detections
				//			cx += dbb[i].x;
				//			cy += dbb[i].y;
				//			cw += dbb[i].width;
				//			ch += dbb[i].height;
				//			close_detections++; //记录最近邻box的个数  
				//			// printf("与跟踪器重叠度大于0.7的检测器窗口: %d %d %d %d\n", dbb[i].x, dbb[i].y, dbb[i].width, dbb[i].height);
				//		}
				//	}
				//	if (close_detections > 1)
				//	{//目标bounding box，但是跟踪器的权值较大  close_detections一般比10小
				//		//bbnext.x = cvRound((float)(10 * tbb.x + cx) / (float)(10 + close_detections));   // weighted average trackers trajectory with the close detections
				//		//bbnext.y = cvRound((float)(10 * tbb.y + cy) / (float)(10 + close_detections));
				//		//bbnext.width = cvRound((float)(10 * tbb.width + cw) / (float)(10 + close_detections));
				//		//bbnext.height = cvRound((float)(10 * tbb.height + ch) / (float)(10 + close_detections));
				//		// printf("Tracker bb: %d %d %d %d\n", tbb.x, tbb.y, tbb.width, tbb.height);
				//		// printf("Average bb: %d %d %d %d\n", bbnext.x, bbnext.y, bbnext.width, bbnext.height);
				//		//  printf(" %d个权重检测器窗口与跟踪器融合\n", close_detections);
				//		/*
				//		kcfbox = Rect2d(bbnext.x, bbnext.y, bbnext.width, bbnext.height);
				//		kcf = Tracker::create("KCF");
				//		kcf->init(frame, kcfbox);
				//		*/
				//		lastvalid = false;
				//	}
				//	else
				//	{
				//		// printf("%d close detections were found\n", close_detections);

				//	}
				//}
			}
			else
			{
				
				lastconf = tconf;//表示相关相似度的阈值
				lastvalid = true;//表示保守相似度的阈值
				
				/* lastconf = 1;
				 lastvalid = false;
				 lastboxfound = true;*/
				 //cout << "没有检测到相似目标！！！！使用跟踪器窗口" << endl;
			}
		}

		else{ //   If NOT tracking
			printf("Not tracking..\n");
			lastboxfound = false;
			lastvalid = false;
			if (detected)
			{  //如果跟踪器没有跟踪到目标，但是检测器检测到了一些可能的目标box，那么同样对其进行聚类，但只是简单的  
				//将聚类的cbb[0]作为新的跟踪目标box（不比较相似度了？？还是里面已经排好序了？？），重新初始化跟踪器 //  and detector is defined
				clusterConf(dbb, dconf, cbb, cconf);   //  cluster detections
				// printf("Found %d clusters\n",(int)cbb.size());
				if (cconf.size() == 1);

				{
					bbnext = cbb[0];
					lastconf = cconf[0];
					printf("Confident detection..reinitializing tracker\n");
					lastboxfound = true;
				}

			}
		}
		t2 = clock();
		// cout << "综合模块耗时" << t2 - t1 << "ms" << endl;
		lastbox = bbnext;
		if (tm)
		{
			if (!tracked&&lastboxfound)
			{
				cout << "lx12345566" << endl;
				kcfbox = Rect2d(bbnext.x, bbnext.y, bbnext.width, bbnext.height);
				kcf = Tracker::create("KCF");
				kcf->init(frame, kcfbox);
			}
		}
		/*
		if (!tracked)
		{
			cout <<" lx" << endl;
			//kcfbox = Rect2d(bbnext.x, bbnext.y, bbnext.width, bbnext.height);
			kcf = Tracker::create("KCF");
			kcf->init(frame, lastbox);
		}
		*/
		/* if (lastboxfound)
		   fprintf(bb_file,"%d,%d,%d,%d,%f\n",lastbox.x,lastbox.y,lastbox.br().x,lastbox.br().y,lastconf);
		   else
		   fprintf(bb_file,"NaN,NaN,NaN,NaN,NaN\n");*/
		t1 = clock();
		cout << lastvalid << endl;
		if (lastvalid && tl)
			learn(img2, tk);
		t2 = clock();
		// cout << "学习模块耗时" << t2 - t1 << "ms"<<endl;
		Point2f currentpt;
		currentpt.x = (float)lastbox.x + (float)lastbox.width / 2.f;
		currentpt.y = (float)lastbox.y + (float)lastbox.height / 2.f;
		if (tm)//更新马尔可夫模型
		{
			if (tracked)	//如果跟踪模块成功额，用最好的跟踪点来更新，否则用目标框的中心点来更新
			{
				Mp1.tmupdate(trackpt1, trackpt2, currentpt);
				Mp2.tmupdate(trackpt1, trackpt2, currentpt);
			}
			else
			{
				Mp1.tmupdate(currentpt);
				Mp2.tmupdate(currentpt);
			}
		}
	}


void TLD::track(const Mat& img1, const Mat& img2,vector<Point2f>& points1,vector<Point2f>& points2,Point2f& pt1,Point2f& pt2){
  /*Inputs:
   * -current frame(img2), last frame(img1), last Bbox(bbox_f[0]).
   *Outputs:
   *- Confidence(tconf), Predicted bounding box(tbb),Validity(tvalid), points2 (for display purposes only)
   */
  //Generate points
	//网格均匀撒点（均匀采样），在lastbox中共产生最多10*10=100个特征点，存于points1  
  bbPoints(points1,lastbox);	//初始化时latsbox=best_box
  if (points1.size()<1){
     // printf("BB= %d %d %d %d,光流跟踪点没有生成\n",lastbox.x,lastbox.y,lastbox.width,lastbox.height);
      tvalid=false;
      tracked=false;
      return;
  }
  vector<Point2f> points = points1;

  //Frame-to-frame tracking with forward-backward error cheking  
  //trackf2f函数完成：跟踪、计算FB error和匹配相似度sim，然后筛选出 FB_error[i] <= median(FB_error) 和   
  //sim_error[i] > median(sim_error) 的特征点（跟踪结果不好的特征点），剩下的是不到50%的特征点,成功返回的true，失败返回的是flase  
  tracked = tracker.trackf2f(img1,img2,points,points2,pt1,pt2);
  if (tracked)
  {
      //Bounding box prediction
	  //利用剩下的这不到一半的跟踪点输入来预测bounding box在当前帧的位置和大小 tbb 
      bbPredict(points,points2,lastbox,tbb);
	  //跟踪失败检测：如果FB error的中值大于10个像素（经验值），或者预测到的当前box的位置移出图像，则  
	  //认为跟踪错误，此时不返回bounding box；Rect::br()返回的是右下角的坐标  
	  //getFB()返回的是FB error的中值  
      if (tracker.getFB()>10 || tbb.x>img2.cols ||  tbb.y>img2.rows || tbb.br().x < 1 || tbb.br().y <1){
          tvalid =false; //too unstable prediction or bounding box out of image
          tracked = false;
         // printf("太不靠谱的光流预测FB error=%f\n",tracker.getFB());//Too unstable predictions FB error=%f\n
          return;
      }
      //Estimate Confidence and Validity
	  //评估跟踪确信度和有效性  
      Mat pattern;
      Scalar mean, stdev;
      BoundingBox bb;
      bb.x = max(tbb.x,0);
      bb.y = max(tbb.y,0);
      bb.width = min(min(img2.cols-tbb.x,tbb.width),min(tbb.width,tbb.br().x));//可能目标只有一部分在视野中，另外的在视频框外
      bb.height = min(min(img2.rows-tbb.y,tbb.height),min(tbb.height,tbb.br().y));
	  //归一化img2(bb)对应的patch的size（放缩至patch_size = 15*15），存入pattern 
      getPattern(img2(bb),pattern,mean,stdev);
      vector<int> isin;
      float dummy,conf;
	  //计算图像片pattern到在线模型M的保守相似度  
      classifier.NNConf(pattern,isin,conf,dummy,tconf); //Conservative Similarity tconf是用csconf 用最近邻分类器的Conservative Similarity【5.2】作为跟踪目标的得分即可，后面要用这个得分和检测器进行比较。
      tvalid = lastvalid;
	  //保守相似度大于阈值，则评估跟踪有效 
      if (tconf>classifier.thr_nn_valid){
          tvalid =true;//判定轨迹是否有效，从而决定是否要增加正样本，标志位tvalid【论文5.6.2 P-Expert】 
      }
  }
  else
    printf("没有点被跟踪到\n");

}

//网格均匀撒点，box共10*10=100个特征点  
void TLD::bbPoints(vector<cv::Point2f>& points,const BoundingBox& bb){
  int max_pts=10;
  int margin_h=0;//采样边界  
  int margin_v=0;
  int stepx = ceil((bb.width-2*margin_h)/max_pts);//ceil返回大于或者等于指定表达式的最小整数  
  int stepy = ceil((bb.height-2*margin_v)/max_pts);
  //网格均匀撒点，box共10*10=100个特征点  
  for (int y=bb.y+margin_v;y<bb.y+bb.height-margin_v;y+=stepy){
      for (int x=bb.x+margin_h;x<bb.x+bb.width-margin_h;x+=stepx){
          points.push_back(Point2f(x,y));
      }
  }
}

//利用剩下的这不到一半的跟踪点输入来预测bounding box在当前帧的位置和大小  
void TLD::bbPredict(const vector<cv::Point2f>& points1,const vector<cv::Point2f>& points2,
                    const BoundingBox& bb1,BoundingBox& bb2)    
{
  int npoints = (int)points1.size();
  vector<float> xoff(npoints);//位移  
  vector<float> yoff(npoints);
 // printf("光流跟踪到 : %d个点\n",npoints);
  for (int i=0;i<npoints;i++){//计算每个特征点在两帧之间的位移 
      xoff[i]=points2[i].x-points1[i].x;
      yoff[i]=points2[i].y-points1[i].y;
  }
  float dx = median(xoff);
  float dy = median(yoff);
  float s;
  //计算bounding box尺度scale的变化：通过计算 当前特征点相互间的距离 与 先前（上一帧）特征点相互间的距离 的  
  //比值，以比值的中值作为尺度的变化因子  
  if (npoints>1){
      vector<float> d;
      d.reserve(npoints*(npoints-1)/2);
      for (int i=0;i<npoints;i++){
          for (int j=i+1;j<npoints;j++){
			  //计算 当前特征点相互间的距离 与 先前（上一帧）特征点相互间的距离 的比值（位移用绝对值）  
              d.push_back(norm(points2[i]-points2[j])/norm(points1[i]-points1[j]));
          }
      }
      s = median(d);
  }
  else {
      s = 1.0;
  }
  float s1 = 0.5*(s-1)*bb1.width;// top-left 坐标的偏移(s1,s2) 
  float s2 = 0.5*(s-1)*bb1.height;
  //printf("光流跟踪尺度变化s= %f 宽偏移s1= %f高偏移 s2= %f \n",s,s1,s2);
  //得到当前bounding box的位置与大小信息  
  //当前box的x坐标 = 前一帧box的x坐标 + 全部特征点位移的中值（可理解为box移动近似的位移） - 当前box宽的一半
  bb2.x = round( bb1.x + dx -s1);
  bb2.y = round( bb1.y + dy -s2);
  bb2.width = round(bb1.width*s);
  bb2.height = round(bb1.height*s);
  //printf("跟踪器预测到的方框 bb: %d %d %d %d\n",bb2.x,bb2.y,bb2.width,bb2.height);
  //changeGrid(bb2);
}
void TLD::changeGrid(BoundingBox& tobb)
{
	BoundingBox tb;
	gridoftracker.reserve(grid.size());
	for (int i = 0; i < grid.size(); i++)
	{
		tb = grid[i];
		tb.x = min(max(grid[i].x + tobb.x - lastbox.x,1),currentframe.cols);
		tb.y = max(min(grid[i].x + tobb.y - lastbox.y,currentframe.rows),1);
		gridoftracker.push_back(tb);
	}
	
}
void TLD::detect(const cv::Mat& frame,bool lastboxfound,bool tk,bool tm,float spdirection,float czdirection){
  //cleaning
  dbb.clear();
  dconf.clear();
  dt.bb.clear();//检测的结果，一个目标一个bounding box 
  kalman_boxes.clear();
  double t = (double)getTickCount();
  
  Mat img(frame.rows,frame.cols,CV_8U);
  integral(frame,iisum,iisqsum);
  GaussianBlur(frame,img,Size(9,9),1.5);
  int numtrees = classifier.getNumStructs();
  float fern_th = classifier.getFernTh();//getFernTh()返回thr_fern; 集合分类器的分类阈值,这个阈值是经过训练的
  vector <int> ferns(10);
  float conf;
  int a=0;
  Mat patch;
  
  //级联分类器模块一：方差检测模块，利用积分图计算每个待检测窗口的方差，方差大于var阈值（目标patch方差的50%）的，  
  //则认为其含有前景目标 
  RNG rng;
  int count = 0;
  double fangcha = 0;
 /* //cout << "grid.size=" << grid.size() << endl;
  float ratofscal = tbb.width*1.0f / (lastbox.width*1.0f);//tbb是track以后的目标框，lastbox是之前的
  int scalidx=0;
  float absdetasc = abs(ratofscal - SCALES[0]);//赋初值
  for (int i = 0; i < gridscalnum.size()-1; i++)	//gridscalnum容器中放的是第sc个尺度中有多少个扫描窗口，其size就是尺度数。。搞得这么玄乎
  {
	  if (abs(ratofscal - SCALES[i + 1]) <absdetasc)	//这应该是为了找出最接近于该次缩放的尺度吧。。
	  {
		  absdetasc = abs(ratofscal - SCALES[i + 1]);
		  //cout << "abs(ratofscal - SCALES[i])=" << abs(ratofscal - SCALES[i]) << endl;
		  scalidx = i+1;
	  }
		  
  }
  cout << "ratio of scale 跟踪器前后两帧尺度变化=" << ratofscal << endl;
  cout << "尺寸scalidx是" << scalidx << endl;
 
  
  for (; numj <= scalidx-2; numj++)	//numl是在这次缩放的尺度之前的所有扫描窗口数
  {
	  numl += gridscalnum[numj];
  }
  if (scalidx <= (gridscalnum.size()-1)/2)
  {
	  numr = numl + gridscalnum[numj]+gridscalnum[numj+1]+gridscalnum[numj+2]+gridscalnum[numj+3];
  }
  else
	  numr = grid.size();
  cout << "从第num=" << numl << "个grid开始检测器" << endl;
  */
  clock_t t1 = clock();
  if (lastboxfound&&tk)
  {
	  Point2f measurept ;
	  Point2f predictpt ;
	  measurept.x = (float)lastbox.x +(float) lastbox.width / 2.f;
	  measurept.y = (float)lastbox.y +(float) lastbox.height / 2.f;
	  Kalman.kalmanpredict(measurept, predictpt);
	// Mp1.tmupdate(measurept);
	// Mp2.tmupdate(measurept);
	 int varr =round(predictpt.x + 3.f*(float)lastbox.width/2.f);
	 int varu =round(predictpt.y + 3.f*(float)lastbox.height/2.f) ;
	 int varl =round(predictpt.x - 3.f*(float)lastbox.width/2.f);
	 int vard =round( predictpt.y - 3.f*(float)lastbox.height/2.f);
	 //printf("%d\t%d\t%d\t%d\n", varr, varl, varu, vard);
	 /*
	 if (tm)
	 {
		 
		 if (spdirection == 1.f)
			 varl =round(measurept.x - (float)lastbox.width / 2.f);
		 if (spdirection == 0.f)
			 varr = round(measurept.x + (float)lastbox.width / 2.f);
		 if (czdirection == 1.f)
			 vard = round(measurept.y - (float)lastbox.height / 2.f);
		 if (czdirection == 0.f)
			 varu =round(measurept.y +(float) lastbox.height / 2.f);
			 
	 }
	 */
	 
	 for (int i = 0; i < grid.size(); i++)
	 {
		 if (grid[i].x > varr/*predictpt.x +2 * lastbox.width*/ || grid[i].y >varu/* predictpt.y + 2 * lastbox.height*/ || grid[i].br().x < varl/*predictpt.x - 2 * lastbox.width*/ || grid[i].br().y < vard/*predictpt.y - 2 * lastbox.height*/);
		 else
		 {
			 kalman_boxes.push_back(i);
			 fangcha = getVar(grid[i], iisum, iisqsum);
			 if (fangcha >= var)////第一关：方差 best_box的标准差*0.5
			 {//计算每一个扫描窗口的方差  
				 a++;
				 //级联分类器模块二：集合分类器检测模块  
				 patch = img(grid[i]);
				 classifier.getFeatures(patch, grid[i].sidx, ferns);//得到该patch特征（13位的二进制代码）  
				 conf = classifier.measure_forest(ferns);//计算该特征值对应的后验概率累加值  
				 tmp.conf[i] = conf;
				 tmp.patt[i] = ferns;
				 //如果集合分类器的后验概率的平均值大于阈值fern_th（由训练得到），就认为含有前景目标  
				 if (conf > numtrees*fern_th){
					 /* printf("conf=%f,fern=", conf);
					 for (int i = 0; i < 10; i++)
					 {
					 cout << ferns[i]<<" ";
					 }
					 cout << endl;*/
					 dt.bb.push_back(i);//将通过以上两个检测模块的扫描窗口记录在detect structure中  
				 }
			 }
			 else
				 tmp.conf[i] = 0.0;
		 }
	 }
  }
  else{
	  numl = 0;
	  numr = grid.size();
	  for (int i = numl; i < numr; i++)
	  {
		 
			  fangcha = getVar(grid[i], iisum, iisqsum);
			  if (fangcha >= var)////第一关：方差 best_box的标准差*0.5
			  {//计算每一个扫描窗口的方差  
				  a++;
				  //级联分类器模块二：集合分类器检测模块  
				  patch = img(grid[i]);
				  classifier.getFeatures(patch, grid[i].sidx, ferns);//得到该patch特征（13位的二进制代码）  
				  conf = classifier.measure_forest(ferns);//计算该特征值对应的后验概率累加值  
				  tmp.conf[i] = conf;
				  tmp.patt[i] = ferns;
				  //如果集合分类器的后验概率的平均值大于阈值fern_th（由训练得到），就认为含有前景目标  
				  if (conf > numtrees*fern_th){
					 
					  dt.bb.push_back(i);//将通过以上两个检测模块的扫描窗口记录在detect structure中  
				  }
			  }
			  else
				  tmp.conf[i] = 0.0;

	  }
  }
	  
  clock_t t2 = clock();
  //cout << " grid检测器的方差分类器与随机森林分类器耗时" << t2 - t1 << "ms" << endl;
  int detections = dt.bb.size();
  //printf("%d Bounding boxes 通过了方差分类器\n",a);
 // printf("%d Initial detection 通过了随机森林分类器\n",detections);
  //如果通过以上两个检测模块的扫描窗口数大于100个，则只取后验概率大的前100个 
  if (detections>15){
      nth_element(dt.bb.begin(),dt.bb.begin()+15,dt.bb.end(),CComparator(tmp.conf));
      dt.bb.resize(15);
      detections=15;
  }
 // printf("听过的窗口数为%d", detections);
//  for (int i=0;i<detections;i++){
//        drawBox(img,grid[dt.bb[i]]);
//    }
//  imshow("detections",img);
  if (detections==0){
	  //printf("没有图像块通过随机森林");
        detected=false;
        return;
      }
 // printf("Fern detector made %d detections ",detections);
 // t=(double)getTickCount()-t;
 // printf("in %gms\n", t*1000/getTickFrequency());
                                                                       //  Initialize detection structure
  dt.patt = vector<vector<int> >(detections,vector<int>(10,0));        //  Corresponding codes of the Ensemble Classifier
  dt.conf1 = vector<float>(detections);                                //  Relative Similarity (for final nearest neighbour classifier)
  dt.conf2 =vector<float>(detections);                                 //  Conservative Similarity (for integration with tracker)
  dt.isin = vector<vector<int> >(detections,vector<int>(3,-1));        //  Detected (isin=1) or rejected (isin=0) by nearest neighbour classifier
  dt.patch = vector<Mat>(detections,Mat(patch_size,patch_size,CV_32F));//  Corresponding patches
  int idx;
  Scalar mean, stdev;
  float dummy;
  //级联分类器模块三：最近邻分类器检测模块 
  t1 = clock();
  float nn_th = classifier.getNNTh();
  for (int i=0;i<detections;i++){                                         //  for every remaining detection
      idx=dt.bb[i];                                                       //  Get the detected bounding box index
	  patch = frame(grid[idx]);
      getPattern(patch,dt.patch[i],mean,stdev);                //  Get pattern within bounding box
      classifier.NNConf(dt.patch[i],dt.isin[i],dt.conf1[i],dummy,dt.conf2[i]);  //  Evaluate nearest neighbour classifier
      dt.patt[i]=tmp.patt[idx];
      //printf("Testing feature %d, conf:%f isin:(%d|%d|%d)\n",i,dt.conf1[i],dt.isin[i][0],dt.isin[i][1],dt.isin[i][2]);
      if (dt.conf1[i]>nn_th){                                               //  idx = dt.conf1 > tld.model.thr_nn; % get all indexes that made it through the nearest neighbour
          dbb.push_back(grid[idx]);                                         //  BB    = dt.bb(:,idx); % bounding boxes
          dconf.push_back(dt.conf2[i]);                                     //  Conf  = dt.conf2(:,idx); % conservative confidences
      }
  }      
  t2 = clock();
  Mat test;
  //cout << "检测器的最近邻分类器耗时" << t2 - t1 << "ms" << endl;
  //打印检测到的可能存在目标的扫描窗口数（可以通过三个级联检测器的） //  end
  if (dbb.size()>0){
      //printf("检测找到 %d 个最近邻匹配块.\n",(int)dbb.size());
      detected=true;
	  for (int i = 0; i < dbb.size(); i++)
	  {
		  test = frame(dbb[i]);
	  }
  }
  else{
      printf("没找到最近邻匹配块.\n");
      detected=false;
  }
}

void TLD::evaluate(){
}

void TLD::learn(const Mat& img,bool tk){
  //printf("[Learning] ");
  ///Check consistency
  BoundingBox bb;
  bb.x = max(lastbox.x,0);
  bb.y = max(lastbox.y,0);
  bb.width = min(min(img.cols-lastbox.x,lastbox.width),min(lastbox.width,lastbox.br().x));
  bb.height = min(min(img.rows-lastbox.y,lastbox.height),min(lastbox.height,lastbox.br().y));
  Scalar mean, stdev;
  Mat pattern;
  //归一化img(bb)对应的patch的size（放缩至patch_size = 15*15），存入pattern  
  getPattern(img(bb),pattern,mean,stdev);
  vector<int> isin;
  float dummy, conf,mconf;
  classifier.NNConf(pattern,isin,conf,dummy,mconf);
  /*if (conf > 0.5&&conf < 0.65)
  {
	  cout << "123" << endl;
  }*/
  if (conf<0.5) { //如果相似度太小了，就不训练 
      //printf("rsconf太小，不训练了\n");
      lastvalid =false;
      return;
  }
  if (pow(stdev.val[0],2)<var){//如果方差太小了，也不训练 
      printf("方差太小，不训练了\n");
      lastvalid=false;
      return;
  }
  if(isin[2]==1){//如果被被识别为负样本，也不训练  
      //printf("图像块属于负样本，不训练了\n");
      lastvalid=false;
      return;
  }
/// Data generation
  //clock_t t1 = clock();
  clock_t t1 = clock();
  for (int i=0;i<grid.size();i++){
      grid[i].overlap = bbOverlap(lastbox,grid[i]);
  }
  //clock_t t2 = clock();
  //cout << "overlap Time is " << t2 - t1 << "ms" << endl;
 // for (int i = 0;i<dt.bb.size())
  vector<pair<vector<int>,int> > fern_examples;
  good_boxes.clear();
  bad_boxes.clear();
 
// printf("%d\t%d\n", grid.size(), kalman_boxes.size());
 if(tk)
	 getOverlappingBoxes(lastbox, num_closest_update,tk);//利用kalman滤波减少了容器的区域，减少了容器的数量
 else
  getOverlappingBoxes(lastbox,num_closest_update);
// classifier.pmodel = img(best_box);
 // printf("%d\t%d\n", good_boxes.size(), bad_boxes.size());
  if (good_boxes.size()>0)
    generatePositiveData(img,num_warps_update);//用仿射模型产生正样本（类似于第一帧的方法，但只产生10*10=100个）  
  else{
    lastvalid = false;
    printf("No good boxes..Not training");
    return;
  }
  fern_examples.reserve(pX.size()+bad_boxes.size());
  fern_examples.assign(pX.begin(),pX.end());
  int idx;
  for (int i=0;i<bad_boxes.size();i++){
      idx=bad_boxes[i];
	  if (tmp.conf[idx] >= 1){ //加入负样本，相似度大于1？？相似度不是出于0和1之间吗？//在detect函数中看到这个temp.conf是后验概率相加得到的，所以>=1代表这些负样本比较容易混淆
          fern_examples.push_back(make_pair(tmp.patt[idx],0));
      }
  }
  //最近邻分类器
  vector<Mat> nn_examples;
  vector<Mat>nn_images;
  nn_examples.reserve(dt.bb.size()+1);
  nn_examples.push_back(pEx);
  nn_images.reserve(dt.bb.size() + 1);
  nn_images.push_back(img(lastbox));
  for (int i=0;i<dt.bb.size();i++){
      idx = dt.bb[i];
	  if (bbOverlap(lastbox, grid[idx]) < 0.1)
	  {
		  nn_examples.push_back(dt.patch[i]);
		  nn_images.push_back(img(grid[idx]));
	  }
  }
  /// Classifiers update
  /// 分类器训练  
 // printf("%d\t%d\n", fern_examples.size(), nn_examples.size());
  classifier.trainF(fern_examples,2);
  clock_t t2 = clock();
  //cout << "generate good and bad boxes time is " << t2 - t1 << "ms" << endl;
  classifier.trainNN(nn_examples,nn_images);
 // classifier.show();//把正样本库（在线模型）包含的所有正样本显示在窗口上  
  //trainTh(img(lastbox));
  //cout << "learning"<< endl;
  cout << "Pex的个数为" << classifier.pEx.size() << endl;
  cout << "Nex的个数为" << classifier.nEx.size() << endl;
}

void TLD::buildGrid(const cv::Mat& img, const cv::Rect& box){
  const float SHIFT = 0.1;
 /* const float SCALES[] = {0.16151,0.19381,0.23257,0.27908,0.33490,0.40188,0.48225,
						  0.57870,0.69444,0.83333,1,1.20000,1.44000,1.72800,
						  2.07360,2.48832,2.98598,3.58318,4.29982,5.15978,6.19174};*/
 /* const float SCALES[] =  {0.90909,0.82645,0.75131,0.68301,0.62092,0.56447,0.51316,0.46651,0.42410,0.38554,1,
						   1.1,1.21,1.331,1.4641,1.61051,1.77156,1.94871,2.14359,2.35794,2.59374
						   };*/
  /*const float SCALES[] = {  0.33490, 0.40188, 0.48225,
	  0.57870, 0.69444, 0.83333, 1, 1.20000, 1.44000, 1.72800,
	  2.07360, 2.48832, 2.98598, 3.58318, 4.29982, 5.15978, 6.19174 };*/
 
  int width, height, min_bb_side;
  //Rect bbox;
  BoundingBox bbox;
  Size scale;
  int sc=0;
  for (int s=0;s<21;s++)
  {
    width = round(box.width*SCALES[s]);//round函数是四舍五入为整数
    height = round(box.height*SCALES[s]);
    min_bb_side = min(height,width);
    if (min_bb_side < min_win || width > img.cols || height > img.rows)
      continue;
    scale.width = width;
    scale.height = height;
    scales.push_back(scale);
	/*for (int y=1;y<img.rows-height;y+=round(SHIFT*min_bb_side))//这是原始程序的循环，扫描的是整个图像，效率比较低
	{
	for (int x=1;x<img.cols-width;x+=round(SHIFT*min_bb_side))
	{
	bbox.x = x;
	bbox.y = y;
	bbox.width = width;
	bbox.height = height;
	bbox.overlap = bbOverlap(bbox,BoundingBox(box));
	bbox.sidx = sc;
	grid.push_back(bbox);
	}
	}*/
	Rect area;//改进后的算法，是基于我们输入的boundingbox，即目标所在的图像块，扫描窗口包含目标图像块，提升了效率
	int num = 0;
	//area.x = max(1,box.x-3*box.width);
	//area.y = max(1, box.y - 3*box.height);
	area.x = 1;
	area.y = 1;
	for (int y = area.y; y < img.rows-height/*min(img.rows, box.y +  4* box.height)-height*/; y += round(SHIFT*min_bb_side))
	for (int x = area.x; x <img.cols-width/*min(img.cols, box.x+4* box.width) - width*/; x += round(SHIFT*min_bb_side))
	{
		bbox.x = x;
		bbox.y = y;
		bbox.width = width;
		bbox.height = height;
		bbox.overlap = bbOverlap(bbox, BoundingBox(box));//计算当前图像块与目标图像块的重叠率
		bbox.sidx = sc;
		grid.push_back(bbox);
		num++;
	}
	gridscalnum.push_back(num);//gridscalnum这个容器中放的是第sc个尺度有多少个扫描窗口
    sc++;
  }
}
int max(int a,int b)
{
	if (a > b)
		return a;
	else
		return b;
}

float TLD::bbOverlap(const BoundingBox& box1,const BoundingBox& box2)//重叠度即两个图像块重叠面积比上总面积
{
  if (box1.x > box2.x+box2.width) { return 0.0; }
  if (box1.y > box2.y+box2.height) { return 0.0; }
  if (box1.x+box1.width < box2.x) { return 0.0; }
  if (box1.y+box1.height < box2.y) { return 0.0; }

  float colInt =  min(box1.x+box1.width,box2.x+box2.width) - max(box1.x, box2.x);
  float rowInt =  min(box1.y+box1.height,box2.y+box2.height) - max(box1.y,box2.y);

  float intersection = colInt * rowInt;
  float area1 = box1.width*box1.height;
  float area2 = box2.width*box2.height;
  return intersection / (area1 + area2 - intersection);
}

void TLD::getOverlappingBoxes(const cv::Rect& box1, int num_closest) {
	float max_overlap = 0;
	// vector<int>::iterator it;
	
		for (int i = 0; i < grid.size(); i++)
		{
			if (grid[i].overlap > max_overlap) //找到最大重叠率的图像块
			{
				max_overlap = grid[i].overlap;
				best_box = grid[i];
			}
			if (grid[i].overlap > 0.7)//重叠率大于0.6的话就放进goodboxes这个容器中，注意放进去的只是这个图像块的编号
			{
				good_boxes.push_back(i);
			}
			else if (grid[i].overlap < bad_overlap) {
				bad_boxes.push_back(i);
			}
		}
		//Get the best num_closest (10) boxes and puts them in good_boxes
		if (good_boxes.size() > num_closest) {
			//for (it = good_boxes.begin(); it != good_boxes.end(); it++)
				//cout << *it << ends;
			//cout << endl;
			std::nth_element(good_boxes.begin(), good_boxes.begin() + num_closest, good_boxes.end(), OComparator(grid));
			//这是一个部分排序算法，goodboxes中放的是这些图像的序号，通过这些序号比较的是这些序号在grid中代表的值，部分排序是指
			//只注重第二个参数在序列中是正确的就可以，之前或者之后怎么排的无所谓，提升了效率，通过屏幕输出，很明显地看出其作用
			//for (it = good_boxes.begin(); it != good_boxes.end(); it++)
				//cout << *it << ends;
			//cout << endl;
			good_boxes.resize(num_closest);
			//	for (it = good_boxes.begin(); it != good_boxes.end(); it++)
					//cout << *it << ends;
				//cout << endl;
		}
		getBBHull();

}
void TLD::getOverlappingBoxes(const cv::Rect& box1, int num_closest,bool tk) {
	float max_overlap = 0;
	// vector<int>::iterator it;
	//printf("%d\t%d\n", grid.size(), kalman_boxes.size());
	int idx = 0;
	for (int i = 0; i < kalman_boxes.size(); i++)
	{
		//printf("%d\t%d", grid.size(), kalman_boxes.size());
		idx = kalman_boxes[i];
		//printf("%d\n", idx);
		if (idx<0)
			break;
		if (grid[idx].overlap > max_overlap) //找到最大重叠率的图像块
		{
			max_overlap = grid[idx].overlap;
			best_box = grid[idx];
		}
		if (grid[idx].overlap > 0.7)//重叠率大于0.6的话就放进goodboxes这个容器中，注意放进去的只是这个图像块的编号
		{
			good_boxes.push_back(idx);
		}
		else if (grid[idx].overlap < bad_overlap) {
			bad_boxes.push_back(idx);
		}
	}
	//Get the best num_closest (10) boxes and puts them in good_boxes
	if (good_boxes.size() > num_closest) {
		//for (it = good_boxes.begin(); it != good_boxes.end(); it++)
		//cout << *it << ends;
		//cout << endl;
		std::nth_element(good_boxes.begin(), good_boxes.begin() + num_closest, good_boxes.end(), OComparator(grid));
		//这是一个部分排序算法，goodboxes中放的是这些图像的序号，通过这些序号比较的是这些序号在grid中代表的值，部分排序是指
		//只注重第二个参数在序列中是正确的就可以，之前或者之后怎么排的无所谓，提升了效率，通过屏幕输出，很明显地看出其作用
		//for (it = good_boxes.begin(); it != good_boxes.end(); it++)
		//cout << *it << ends;
		//cout << endl;
		good_boxes.resize(num_closest);
		//	for (it = good_boxes.begin(); it != good_boxes.end(); it++)
		//cout << *it << ends;
		//cout << endl;
	}
	getBBHull();

}
//这个实际上是10个goodboxes的并集的边框
void TLD::getBBHull(){
  int x1=INT_MAX, x2=0;//初始化
  int y1=INT_MAX, y2=0;
  int idx;
  for (int i=0;i<good_boxes.size();i++){
      idx= good_boxes[i];
      x1=min(grid[idx].x,x1);
      y1=min(grid[idx].y,y1);
      x2=max(grid[idx].x+grid[idx].width,x2);
      y2=max(grid[idx].y+grid[idx].height,y2);
  }
  bbhull.x = x1;		//bbhull这个boundingbox类中存放了我们选出的goodboxes的边界，即他们的并集的边框
  bbhull.y = y1;
  bbhull.width = x2-x1;
  bbhull.height = y2 -y1;
}

bool bbcomp(const BoundingBox& b1,const BoundingBox& b2){
  TLD t;
    if (t.bbOverlap(b1,b2)<0.5)
      return false;
    else
      return true;
}
int TLD::clusterBB(const vector<BoundingBox>& dbb,vector<int>& indexes){
  //FIXME: Conditional jump or move depends on uninitialised value(s)
  const int c = dbb.size();
  //1. Build proximity matrix
  Mat D(c,c,CV_32F);
  float d;
  for (int i=0;i<c;i++){
      for (int j=i+1;j<c;j++){
        d = 1-bbOverlap(dbb[i],dbb[j]);
        D.at<float>(i,j) = d;
        D.at<float>(j,i) = d;
      }
  }
  //2. Initialize disjoint clustering
  float *L = new float[c - 1]; //Level
  int **nodes = new int *[c - 1];
  for (int i = 0; i < 2; i++)
	  nodes[i] = new int[c - 1];
  int *belongs = new int[c];
  //记得在函数末释放分配的内存
  delete[] L;
  L = NULL;
  for (int i = 0; i < 2; ++i)
  {
	  delete[] nodes[i];
	  nodes[i] = NULL;
  }
  delete[]nodes;
  nodes = NULL;
  delete[] belongs;
  belongs = NULL;
 int m=c;
 for (int i=0;i<c;i++){
    belongs[i]=i;
 }
 for (int it=0;it<c-1;it++){
 //3. Find nearest neighbor
     float min_d = 1;
     int node_a, node_b;
     for (int i=0;i<D.rows;i++){
         for (int j=i+1;j<D.cols;j++){
             if (D.at<float>(i,j)<min_d && belongs[i]!=belongs[j]){
                 min_d = D.at<float>(i,j);
                 node_a = i;
                 node_b = j;
             }
         }
     }
     if (min_d>0.5){
         int max_idx =0;
         bool visited;
         for (int j=0;j<c;j++){
             visited = false;
             for(int i=0;i<2*c-1;i++){
                 if (belongs[j]==i){
                     indexes[j]=max_idx;
                     visited = true;
                 }
             }
             if (visited)
               max_idx++;
         }
         return max_idx;
     }

 //4. Merge clusters and assign level
     L[m]=min_d;
     nodes[it][0] = belongs[node_a];
     nodes[it][1] = belongs[node_b];
     for (int k=0;k<c;k++){
         if (belongs[k]==belongs[node_a] || belongs[k]==belongs[node_b])
           belongs[k]=m;
     }
     m++;
 }
 return 1;

}

void TLD::clusterConf(const vector<BoundingBox>& dbb,const vector<float>& dconf,vector<BoundingBox>& cbb,vector<float>& cconf){
  int numbb =dbb.size();
  vector<int> T;
  float space_thr = 0.5;
  int c=1;
  switch (numbb){
  case 1:
    cbb=vector<BoundingBox>(1,dbb[0]);
    cconf=vector<float>(1,dconf[0]);
    return;
    break;
  case 2:
    T =vector<int>(2,0);
    if (1-bbOverlap(dbb[0],dbb[1])>space_thr){
      T[1]=1;
      c=2;
    }
    break;
  default:
    T = vector<int>(numbb,0);
    c = partition(dbb,T,(*bbcomp));
    //c = clusterBB(dbb,T);
    break;
  }
  cconf=vector<float>(c);
  cbb=vector<BoundingBox>(c);
 // printf("Cluster indexes: ");
  BoundingBox bx;
  for (int i=0;i<c;i++){
      float cnf=0;
      int N=0,mx=0,my=0,mw=0,mh=0;
      for (int j=0;j<T.size();j++){
          if (T[j]==i){
              //printf("%d ",i);
              cnf=cnf+dconf[j];
              mx=mx+dbb[j].x;
              my=my+dbb[j].y;
              mw=mw+dbb[j].width;
              mh=mh+dbb[j].height;
              N++;
          }
      }
      if (N>0){
          cconf[i]=cnf/N;
          bx.x=cvRound(mx/N);
          bx.y=cvRound(my/N);
          bx.width=cvRound(mw/N);
          bx.height=cvRound(mh/N);
          cbb[i]=bx;
      }
  }
  //printf("\n");
}
//void TLD::trainTh(Mat&img)
//{
//
//	Mat pattern;
//	Scalar mean, stdev;
//	getPattern(img, pattern, mean, stdev);
//	vector<int> isin;
//	float dummy;
//	float conf;
//	//计算图像片pattern到在线模型M的保守相似度  
//	classifier.NNConf1(pattern, isin, dummy, conf);
//	classifier.thr_nn_valid = conf;
//}
