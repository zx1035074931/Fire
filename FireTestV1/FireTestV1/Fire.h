#pragma once
#ifndef _FIRE
#define _FIRE

#include<string>
#include <opencv2/opencv.hpp>  
#include <opencv.hpp>
#include <fstream>
#include "stdio.h"
#include <iostream>
#include <math.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

class Fire
{
private:
	//质量检测
	//double colour(Mat &img);
	//double fuzzy_degree(Mat &src);
	//八大特征
	float Cen_hei(vector<Point>contour);
	double Factor(vector<Point>contour);
	float R_t(Mat fst, Mat dst);
	float Rec(vector<Point>contour);
	double Rou(vector<Point>contour);
	double Are_dy(double area1, double area2);
	double Rou_dy(double length1, double length2);
	double Hei_dy(double height1, double height2);
	bool R_max(Mat fst, Mat dst);//没用到

	Mat picCheckColorHSV(Mat &inImg);//检测疑似火焰区域
	Mat videoCheckColorHSV(Mat &inImg);//没用到
									   //
	double date[9] = { 0 };//存每一帧的数据
	void picprogram(Mat pic);//原本代码中，图片要进行的程序
	void fillHole(const cv::Mat srcimage, cv::Mat &dstimage);//虫洞填充
	vector<vector<vector<double>>> weightlocal;//建立5*5*9的double类型vector存储权重数据
	double G_reblur(Mat src);//模糊度
	double color_std(Mat &img);//色偏
	int blurlevel(double blur_mos);//模糊度等级
	int colorlevel(double color_mos);//色偏等级
	bool feather;//存储红色特征
	bool getresult(int color_grade, int blur_grade, double th);//特征值与权重相乘，并于阈值比较
	Mat fireHSV;//疑似火焰区域

	Mat conv(Mat &in);//卷积
	struct post
	{
		float std;
		int i;
		int j;
	};
	bool comp(const Fire::post & a, const Fire::post & b);

public:
	Fire(String Add, vector<vector<vector<double>>> weight);

	//~Fire();
};

#endif