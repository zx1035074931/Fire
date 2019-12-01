#pragma once
#ifndef _FIRE
#define _FIRE
#include<string>
#include <opencv2/opencv.hpp>  
#include <opencv.hpp>
#include <fstream>
#include <cstdio>
#include <iostream>
#include <cmath>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include<vector>
using namespace cv;
using namespace std;
class Fire
{
private:
	void picprogram(Mat pic);//预处理，原本代码中，图片要进行的程序	
	void AreaExtract(Mat inImg);//检测疑似火焰区域
	Mat picCheckColorHSV(Mat &inImg);//检测疑似火焰区域
	Mat fireHSV;//疑似火焰区域
	//质量检测
	double G_reblur(Mat src);//模糊度
	double color_std(Mat &img);//色偏
	int blurlevel(double blur_mos);//模糊度等级
	int colorlevel(double color_mos);//色偏等级
	//八大特征
	//bool R_max(Mat fst, Mat dst);//未启用
	//float R_t(Mat suspic);//红色分量
	float R_t(Mat fst);
	float R_t(Mat fst, Mat dst);
	float Rec(vector<Point>contour);//矩形度
	double Factor(vector<Point>contour);//圆形度
	float Cen_hei(vector<Point>contour);//质心高度比
	double Rou(vector<Point>contour);//边缘粗糙度
	double change_dy(double para1, double para2);//面积变化率，边缘抖动变化率，高度变化率
	double date[9] = { 0 };//存每一帧的数据
	void fillHole(const cv::Mat srcimage, cv::Mat &dstimage);//虫洞填充
	vector<vector<vector<double>>> weightlocal;//建立5*5*9的double类型vector存储权重数据
	bool feather;//存储红色特征
	bool getresult(int color_grade, int blur_grade, double th);//特征值与权重相乘，并于阈值比较
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