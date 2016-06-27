#include <opencv2/opencv.hpp>
#include "tld_utils.h"
#include <iostream>
#include <sstream>
#include "TLD.h"
#include <stdio.h>
#include<time.h>
//#include <opencv2/highgui/highgui.hpp>
using namespace cv;
using namespace std;
//Global variables
Rect box;
bool drawing_box = false;
bool gotBB = false;
bool tl = true;
bool tk = false;	//卡尔曼开关
//bool tk = false;
//bool tma = false;	//方向预测开关
bool rep = false;
bool fromfile=false;
string video;

void readBB(char* file){
  ifstream bb_file (file);
  string line;
  getline(bb_file,line);
  istringstream linestream(line);
  string x1,y1,x2,y2;
  getline (linestream,x1, ',');
  getline (linestream,y1, ',');
  getline (linestream,x2, ',');
  getline (linestream,y2, ',');
  int x = atoi(x1.c_str());// = (int)file["bb_x"];
  int y = atoi(y1.c_str());// = (int)file["bb_y"];
  int w = atoi(x2.c_str())-x;// = (int)file["bb_w"];
  int h = atoi(y2.c_str())-y;// = (int)file["bb_h"];
  box = Rect(x,y,w,h);
}
//bounding box mouse callback
void mouseHandler(int event, int x, int y, int flags, void *param){
  switch( event ){
  case CV_EVENT_MOUSEMOVE:
    if (drawing_box){
        box.width = x-box.x;
        box.height = y-box.y;
    }
    break;
  case CV_EVENT_LBUTTONDOWN:
    drawing_box = true;
    box = Rect( x, y, 0, 0 );
    break;
  case CV_EVENT_LBUTTONUP:
    drawing_box = false;
    if( box.width < 0 ){
        box.x += box.width;
        box.width *= -1;
    }
    if( box.height < 0 ){
        box.y += box.height;
        box.height *= -1;
    }
    gotBB = true;
    break;
  }
}

void print_help(char** argv){
  printf("use:\n     %s -p /path/parameters.yml\n",argv[0]);
  printf("-s    source video\n-b        bounding box file\n-tl  track and learn\n-r     repeat\n");
}

void read_options(int argc, char** argv,VideoCapture& capture,FileStorage &fs){
  for (int i=0;i<argc;i++)
  {
      if (strcmp(argv[i],"-b")==0)
	  {
          if (argc>i){
              readBB(argv[i+1]);
              gotBB = true;
          }
          else
            print_help(argv);
      }
      if (strcmp(argv[i],"-s")==0)
	  {
          if (argc>i)
		  {
              video = string(argv[i+1]);
              capture.open(video);
              fromfile = true;
          }
          else
            print_help(argv);

      }
      if (strcmp(argv[i],"-p")==0)
	  {
          if (argc>i)
		  {
              fs.open(argv[i+1], FileStorage::READ);
          }
          else
            print_help(argv);
      }
      if (strcmp(argv[i],"-no_tl")==0)
	  {
          tl = false;
      }
	  if (strcmp(argv[i], "-no_tk") == 0)
	  {
		  tk = false;
	  }
      if (strcmp(argv[i],"-r")==0)
	  {
          rep = true;
      }
  }
}

int main(int argc, char * argv[]){//int argc, char * argv[]     "-p" "parameters.yml" "-s" 0 "-r"
  VideoCapture capture;
 //capture.open(0);
  FileStorage fs;
  fs.open("parameters.yml", FileStorage::READ);
  //Read options
  //int a = 0;
  /*
  do {
	  cout << "open kalman prediction,1 for yes,0 for no" << endl;
	  cin >> a;
	  if (a == 1)
		  tk = true;
	  else if (a == 0)
		  tk = false;
  } while (a != 0 && a != 1);
  do {
	  cout << "open direction prediction,1 for yes,0 for no" << endl;
	  cin >> a;
	  if (a == 1)
		  tma = true;
	  else if (a == 0)
		  tma = false;
  } while (a != 0 && a != 1);
  */
  char str[30];
  do {
	  cout << "please input the name of video" << endl;
	  cin >> str;
	  video = string(str);
	  if(capture.open(video))
	  fromfile = true;
  } while (fromfile == false);
 // read_options(argc,argv,capture,fs);
  //Init camera
  if (!capture.isOpened())
  {
	cout << "capture device failed to open!" << endl;
    return 1;
  }
  //Register mouse callback to draw the bounding box
 // VideoWriter writer("VideoTest1.avi", CV_FOURCC('M', 'J', 'P', 'G'), 25.0, Size(320, 240));
 /* if (writer.isOpened())
  {
	  printf("123");
  }
  */
  cvNamedWindow("TLD",CV_WINDOW_AUTOSIZE);
  cvSetMouseCallback( "TLD", mouseHandler, NULL );
  //TLD framework
  TLD tld;
  //Read parameters file
  tld.read(fs.getFirstTopLevelNode());
  Mat frame;
  Mat last_gray;
  Mat first;
  if (fromfile){
      capture >> frame;
      cvtColor(frame, last_gray, CV_RGB2GRAY);
      frame.copyTo(first);
  }else{
      capture.set(CV_CAP_PROP_FRAME_WIDTH,340);
      capture.set(CV_CAP_PROP_FRAME_HEIGHT,240);
  }

  ///Initialization
//GETBOUNDINGBOX:
  while (!gotBB)
  {
	  while (!gotBB)
	  {
		  if (!fromfile){
			  capture >> frame;
		  }
		  else
			  first.copyTo(frame);

		  cvtColor(frame, last_gray, CV_RGB2GRAY);
		  drawBox(frame, box);
		  imshow("TLD", frame);
		  if (cvWaitKey(33) == 'q')
			  return 0;
	  }
	  if (min(box.width, box.height) < (int)fs.getFirstTopLevelNode()["min_win"])
	  {
		  cout << "Bounding box too small, try again." << endl;
		  gotBB = false;
	  }
  }
  /*
  if (min(box.width,box.height)<(int)fs.getFirstTopLevelNode()["min_win"]){
      cout << "Bounding box too small, try again." << endl;
      gotBB = false;
      goto GETBOUNDINGBOX;
  }
  */
  //Remove callback
 
  cvSetMouseCallback( "TLD", NULL, NULL );
 // printf("Initial Bounding Box = x:%d y:%d h:%d w:%d\n",box.x,box.y,box.width,box.height);
  //Output file
  FILE  *bb_file = fopen("missingframe.txt","w");
  //TLD initialization
  tld.init(first,box,bb_file);
  char name1[30];
  char name2[30];
  ///Run-time
  Mat current_gray;
  BoundingBox pbox;
  vector<Point2f> pts1;//光流前向跟踪点
  vector<Point2f> pts2;
  bool status=true;//status lastboxfound
  int frames = 1;
  int detections = 1;
  int t=0;
  Mat current;
//REPEAT:
  //Mat frame_temp;

  while(capture.read(frame)){
	  
	 /* if (!capture.read(frame))
		  break;*/
	  clock_t t1 = clock();
    //get frame
    cvtColor(frame, current_gray, CV_RGB2GRAY);
    //Process Frame
	clock_t t3 = clock();
    tld.processFrame(frame,last_gray,current_gray,pts1,pts2,pbox,status,tl,tk,bb_file);//跟踪，检测，综合，学习四个模块
	clock_t t4 = clock();
	//printf("运行%dms\n", t4 - t3);
	//Draw Points
    if (status){//跟踪成功，这里指的是我们最后通过综合模块输出了一个框，称为输出成功
      //drawPoints(frame,pts1);
      //drawPoints(frame,pts2,Scalar(255,255,0));
		frame.copyTo(current);
      drawBox(frame,pbox);
      detections++;
    }
    //Display
   imshow("TLD", frame);
    //swap points and images
	//writer << frame;
	//clock_t t4 = clock();
	//printf("%dms\n", t4 - t3);
   //waitKey(100);
    swap(last_gray,current_gray);
    pts1.clear();
    pts2.clear();
    frames++;
    //printf("Detection rate: %d/%d\n",detections,frames);
    if (cvWaitKey(2) == 'q')
      break;
	clock_t t2 = clock();
	//cout << "Time is " << t2 - t1 << "ms"<<endl;
	if(status)
	t += (t2 - t1);
	sprintf(name1, "%s%d%s", "frame", frames, ".png");
	sprintf(name2, "%s%d%s", "target", frames, ".png");
	//imwrite(name1, frame);
	//cout << status << endl;
	if (status)
	{
		//if (pbox.x > 0 && pbox.y > 0 && pbox.br().x < frame.cols&&pbox.br().y < frame.rows)
		//imwrite(name2, current(pbox));
	}
	else
		fprintf(bb_file, "%d\n", frames);
  }
  /*
  if (rep){
    rep = false;
    tl = false;
    fclose(bb_file);
    bb_file = fopen("final_detector.txt","w");
    //capture.set(CV_CAP_PROP_POS_AVI_RATIO,0);
    capture.release();
    capture.open(video);
    goto REPEAT;
  }
  */
  printf("平均每一帧所用的时间为%dms\n", t / detections);
  printf("Detection rate: %d/%d\n", detections, frames);
  fclose(bb_file);
  capture.release();
  system("pause");
  return 0;
}
