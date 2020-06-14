#ifndef _FEATURE_COMPARE_H
#define _FEATURE_COMPARE_H

#include <iostream>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iomanip>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <boost/format.hpp> 
#include "tic_toc.h"

#define GFTT_ 0
#define ORB_ 1
#define SURF_ 2
#define SIFT_ 3
#define BF_ "BruteForce-Hamming"
#define FL_ "FlannBased"

using namespace std;
using namespace cv;
using namespace xfeatures2d;

class Feature_compare {
public:
  Feature_compare(const int type1,const string type2);
  ~Feature_compare(){};
  
  const int feature_type;const string match_type; 
  Ptr<FeatureDetector>  normal_detector;  //for orb&gftt
  Ptr<Feature2D> extra_detector;          //for sift&surf
  Ptr<DescriptorExtractor> descriptor;    //for orb&gftt
  Ptr<DescriptorMatcher> matcher;         //for all
  inline void init(const int type1,const string type2);
  
};
//Mat Image1,Mat Image2,Mat img1,Mat img2,vector<KeyPoint> keypoints1,vector<KeyPoint> keypoints2
Feature_compare::Feature_compare(const int type1,const string type2)
:feature_type(type1),match_type(type2)
{
   init(type1,type2);
  
}

inline void Feature_compare::init(const int type1,const string type2)
{
  
  if(type1==0){
   normal_detector = GFTTDetector::create(500);
   descriptor = BriefDescriptorExtractor::create(); 
//    descriptor = BRISK::create();
   matcher  = DescriptorMatcher::create ( type2 );
  }
  if(type1==1){
   normal_detector = ORB::create(500);
   descriptor = ORB::create(); 
   matcher  = DescriptorMatcher::create ( type2 );
  }
  if(type1==2){
   extra_detector = SURF::create(500);
//     Ptr<Feature2D> detector = xfeatures2d::SURF::create();
//     Ptr<SurfFeatureDetector> detector = SurfFeatureDetector::create(1000);
   matcher = DescriptorMatcher::create(type2);
  }
  if(type1==3){
   extra_detector = SIFT::create(500);
//     Ptr<SiftFeatureDetector> detector = SiftFeatureDetector::create(1000);
   matcher = DescriptorMatcher::create(type2);
  }
}




#endif






