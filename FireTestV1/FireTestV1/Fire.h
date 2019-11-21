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
	//�������
	//double colour(Mat &img);
	//double fuzzy_degree(Mat &src);
	//�˴�����
	float Cen_hei(vector<Point>contour);
	double Factor(vector<Point>contour);
	float R_t(Mat fst, Mat dst);
	float Rec(vector<Point>contour);
	double Rou(vector<Point>contour);
	double Are_dy(double area1, double area2);
	double Rou_dy(double length1, double length2);
	double Hei_dy(double height1, double height2);
	bool R_max(Mat fst, Mat dst);//û�õ�

	Mat picCheckColorHSV(Mat &inImg);//������ƻ�������
	Mat videoCheckColorHSV(Mat &inImg);//û�õ�
									   //
	double date[9] = { 0 };//��ÿһ֡������
	void picprogram(Mat pic);//ԭ�������У�ͼƬҪ���еĳ���
	void fillHole(const cv::Mat srcimage, cv::Mat &dstimage);//�涴���
	vector<vector<vector<double>>> weightlocal;//����5*5*9��double����vector�洢Ȩ������
	double G_reblur(Mat src);//ģ����
	double color_std(Mat &img);//ɫƫ
	int blurlevel(double blur_mos);//ģ���ȵȼ�
	int colorlevel(double color_mos);//ɫƫ�ȼ�
	bool feather;//�洢��ɫ����
	bool getresult(int color_grade, int blur_grade, double th);//����ֵ��Ȩ����ˣ�������ֵ�Ƚ�
	Mat fireHSV;//���ƻ�������

	Mat conv(Mat &in);//���
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