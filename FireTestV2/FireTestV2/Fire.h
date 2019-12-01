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
	void picprogram(Mat pic);//Ԥ����ԭ�������У�ͼƬҪ���еĳ���	
	void AreaExtract(Mat inImg);//������ƻ�������
	Mat picCheckColorHSV(Mat &inImg);//������ƻ�������
	Mat fireHSV;//���ƻ�������
	//�������
	double G_reblur(Mat src);//ģ����
	double color_std(Mat &img);//ɫƫ
	int blurlevel(double blur_mos);//ģ���ȵȼ�
	int colorlevel(double color_mos);//ɫƫ�ȼ�
	//�˴�����
	//bool R_max(Mat fst, Mat dst);//δ����
	//float R_t(Mat suspic);//��ɫ����
	float R_t(Mat fst);
	float R_t(Mat fst, Mat dst);
	float Rec(vector<Point>contour);//���ζ�
	double Factor(vector<Point>contour);//Բ�ζ�
	float Cen_hei(vector<Point>contour);//���ĸ߶ȱ�
	double Rou(vector<Point>contour);//��Ե�ֲڶ�
	double change_dy(double para1, double para2);//����仯�ʣ���Ե�����仯�ʣ��߶ȱ仯��
	double date[9] = { 0 };//��ÿһ֡������
	void fillHole(const cv::Mat srcimage, cv::Mat &dstimage);//�涴���
	vector<vector<vector<double>>> weightlocal;//����5*5*9��double����vector�洢Ȩ������
	bool feather;//�洢��ɫ����
	bool getresult(int color_grade, int blur_grade, double th);//����ֵ��Ȩ����ˣ�������ֵ�Ƚ�
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