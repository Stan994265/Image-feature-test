#include "tic_toc.h"
#include "feature_compare.h"

using namespace std;
using namespace cv;
using namespace xfeatures2d;

static string Dataset_order = "../dataset/test.txt";
static string Result_file = "../result/result.txt";

Mat img_1,img_2;
Mat dstImage1, dstImage2;
vector<KeyPoint> keypoints_1, keypoints_2; //remember to clear in loop
vector <KeyPoint> RR_KP1, RR_KP2;

vector<DMatch> feature_extract(Feature_compare &tmp)
{
  if (tmp.feature_type==0||tmp.feature_type==1){
    tmp.normal_detector->detect ( img_1,keypoints_1);
    tmp.normal_detector->detect ( img_2,keypoints_2);
    tmp.descriptor->compute ( img_1, keypoints_1, dstImage1 );
    tmp.descriptor->compute ( img_2, keypoints_2, dstImage2 );
  }
  if (tmp.feature_type==2||tmp.feature_type==3){
    tmp.extra_detector->detectAndCompute(img_1, Mat(), keypoints_1, dstImage1);
    tmp.extra_detector->detectAndCompute(img_2, Mat(), keypoints_2, dstImage2); 
  }
  if (tmp.match_type==FL_){
    dstImage1.convertTo(dstImage1,CV_32F);
    dstImage2.convertTo(dstImage2,CV_32F);
  }
  vector<DMatch> matches;
  tmp.matcher->match ( dstImage1, dstImage2, matches , Mat());
  return matches;
}

void read_data(int index1,int index2){
  boost::format fmt_read("%s%i%s");
  string Data1=(fmt_read %"../dataset/" %(index1) %".png").str();
  string Data2=(fmt_read %"../dataset/" %(index2) %".png").str();
  img_1 = imread (Data1,CV_LOAD_IMAGE_COLOR );
  img_2 = imread (Data2,CV_LOAD_IMAGE_COLOR );
}

vector<DMatch> rough_match(vector<DMatch> &match1){
    vector<DMatch> match2;
    double min_dist=1000, max_dist=0;
    for ( int i = 0; i < dstImage1.rows; i++ )
    {
        double dist = match1[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }  
    cout<<"Max_dist="<<max_dist<<endl;
    cout<<"Min_dist="<<min_dist<<endl;
    for ( int i = 0; i < dstImage1.rows; i++ )
    {
        if ( match1[i].distance <= max ( 2*min_dist, 3.0 ) )//you can change this parameter
        {
            match2.push_back ( match1[i] );
        }
    }
    return match2;
}

vector<DMatch> ransac(vector<DMatch> &match3){
    vector<DMatch> m_Matches(match3);
    vector <KeyPoint> RAN_KP1, RAN_KP2;
    for (size_t i = 0; i < match3.size(); i++){
	RAN_KP1.push_back(keypoints_1[match3[i].queryIdx]);
	RAN_KP2.push_back(keypoints_2[match3[i].trainIdx]);
    }
    vector <Point2f> p01, p02;
    KeyPoint::convert(RAN_KP1,p01);
    KeyPoint::convert(RAN_KP2,p02);
    vector<uchar> RansacStatus;
    Mat Fundamental = findFundamentalMat(p01, p02, RansacStatus, FM_RANSAC);
    //Mat Fundamental = cvFindFundamentalMat(p01, p02, RansacStatus, CV_FM_RANSAC);
    int index = 0;
    vector<DMatch> match4;
    for (size_t i = 0; i < match3.size(); i++){
	if (RansacStatus[i] != 0){
	  RR_KP1.push_back(RAN_KP1[i]);
	  RR_KP2.push_back(RAN_KP2[i]);
	  m_Matches[i].queryIdx = index;
	  m_Matches[i].trainIdx = index;
	  match4.push_back(m_Matches[i]);
	  index++;
	}
    }
    return match4;
}


int main ( int argc, char** argv )
{
    ofstream result;
    result.open(Result_file,fstream::app | fstream::out);
    ifstream data_order;
    data_order.open(Dataset_order.c_str());
    std::string data_line;
    int data1,data2;      
    Feature_compare test(ORB_,BF_);//if use SURF/SIFT BF,you need GPU supported
    while (std::getline(data_order, data_line) && !data_line.empty()) {
    std::istringstream Data(data_line);
    Data >> data1 >>data2;     
    read_data(data1,data2);
    vector<DMatch> matches;//all match
    vector<DMatch> good_matches;//after rough
    vector<DMatch> RR_matches;//after ransac
    //---------------Step1---------------------/
    TicToc time;
    time.tic ();
    matches=feature_extract(test);
//     test.matcher->match ( dstImage1, dstImage2, matches , Mat());
    cout<<"-------------step1-----------"<<endl;
    cout << "all_matches.size:" <<matches.size()<< endl;
    cout << "After feature extract & basic match cost: " << time.toc() << " ms" << endl; 
    //---------------step2---------------------/
    good_matches=rough_match(matches);    
    int ptCount = good_matches.size();
    if (ptCount <= 5)
      {	
	cout << "Don't find enough match points" << endl;	
	result<<"fail!"<<endl;	
	continue;	
      }
    cout<<"-------------step2-----------"<<endl;
    cout << "good_matches.size:" <<good_matches.size()<< endl;
    cout << "After rough cost: " << time.toc() << " ms" << std::endl;
    //---------------step3---------------------/
    RR_matches=ransac(good_matches);
    double T=time.toc();
    cout<<"-------------step3-----------"<<endl;
    cout << "RR_matches.size:" <<RR_matches.size()<< endl;
    cout << "After ransac cost: " << time.toc() << " ms" << std::endl;
    
    
    //---------------step4---------------------/
    Mat img_RR_matches;
    Mat img_match;
    Mat img_goodmatch;
    Mat outimg1;
    drawKeypoints( img_1, keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
    drawMatches ( img_1, keypoints_1, img_2, keypoints_2, matches, img_match );
    drawMatches ( img_1, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatch );
    drawMatches(img_1, RR_KP1, img_2, RR_KP2, RR_matches, img_RR_matches,CV_RGB(0, 255, 255));      
    //---------------step5---------------------/
    boost::format fmt_save("%s%i%s");
    string feature_result=(fmt_save %"../result/Feature" %(data1) %".jpg").str();
    string All_matching=(fmt_save %"../result/All_matching" %(data1) %".jpg").str();
    string After_Pick=(fmt_save %"../result/After_Pick" %(data1) %".jpg").str();
    string After_RANSAC=(fmt_save %"../result/After_RANSAC" %(data1) %".jpg").str();

    //imshow("Feature",outimg1);
    //imshow ( "All_matching", img_match );
    //imshow ( "After Pick", img_goodmatch );
    //imshow("After RANSAC",img_RR_matches);
    imwrite(feature_result, outimg1);
    imwrite(All_matching, img_match);
    imwrite(After_Pick, img_goodmatch);
    imwrite(After_RANSAC, img_RR_matches);
    //waitKey(0);
    //---------------step6---------------------/
    vector <Point2f> P1, P2;
    for( size_t i = 0; i < RR_matches.size(); i++ ){
        P1.push_back( RR_KP1[ RR_matches[i].queryIdx ].pt );
        P2.push_back( RR_KP2[ RR_matches[i].trainIdx ].pt );
      }
    Mat homography = findHomography(P1,P2,RANSAC);
    cout<<"-------------result-----------"<<endl;
    cout<<"H Matrix="<<homography<<endl;
    float repeatability=0.0;
    int correspCount=0;
    evaluateFeatureDetector(img_1,img_2,homography,&RR_KP1,&RR_KP2,repeatability,correspCount);
    cout<<"repeat="<<repeatability<<endl;
    cout<<"corresp="<<correspCount<<endl;
    result<<RR_matches.size()<<" "<<T<<" "<<repeatability<<" "<<correspCount<<endl;
    
    keypoints_1.clear();keypoints_2.clear();
    RR_KP1.clear(); RR_KP2.clear();
}
    data_order.close();
    result.close();
    return 0;
}
