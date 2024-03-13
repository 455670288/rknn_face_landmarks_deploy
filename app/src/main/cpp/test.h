#include <iostream>
#include <vector>
#include <string>
#include "opencv2/opencv.hpp"

struct  return_d_m
{
    cv::Mat dst;
    cv::Mat matri;
};

struct landmarks_output{
    bool status;
    float* face_landmark;
};


struct face_pre //人脸检测前处理
{
    cv::Mat det_img; //模型输入
    float det_scale; //缩放比例
};

struct bbox_landmark  //存放检测框的位置坐标
{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;

    bbox_landmark() : x1(0), y1(0), x2(0), y2(0),score(0) {} //默认初始化方式，传入关键点模块用
    bbox_landmark(float _x1, float _y1, float _x2, float _y2, float _score) : x1(_x1), y1(_y1), x2(_x2), y2(_y2),score(_score) {} //构造函数
};