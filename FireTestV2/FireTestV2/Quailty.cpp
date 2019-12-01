#include "stdio.h"
#include <iostream>
#include <math.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include "Fire.h"
using namespace cv;
using namespace std;
using std::placeholders::_1;
using std::placeholders::_2;
/*-------------------模糊度测试-------------------
--------------------------------------------------
计算相邻像素之间差值的变化,如果待测图像相邻像素之间的差值
比参考图像相邻像素之间的差值大，则原始图像较清晰。相反则原始图像较模糊。*/
double Fire::G_reblur(Mat src)
{
 //rgb2gray
	Mat in;
	cvtColor(src, in, CV_BGR2GRAY);
	Mat mat_c;
	// 只有32FC1, 32FC2, 64FC1, 64FC2才支持乘法
	in.convertTo(mat_c, CV_64F, 1.0, 0);
	//横向的滤波结果
	Mat h = conv(mat_c);
	/*for (int a = 79; a < 84; a++)
	printf(" %f", h.at<double>(49, a));
	printf("\n");*/
	//纵向滤波
	Mat z;
	Mat m1;
	Mat m2;
	transpose(mat_c, m1);
	m2 = conv(m1);
	transpose(m2, z);
	//测试图像的边缘检测
	Mat mat_b;
	Mat grad;
	Mat grad_x, grad_y;//x和y方向上的梯度
	Mat abs_grad_x, abs_grad_y;
	//Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
	//获取梯度信息
	Sobel(mat_c, grad_x, CV_64F, 1, 0, 3, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);
	//Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
	Sobel(mat_c, grad_y, CV_64F, 0, 1, 3, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
	//计算并获取图像中所有大小为side*side,步长为step的区域的方差和位置，存于vector roi 中
	int side = 8;
	int	step = 4;
	Mat roi_area, mat_mean, mat_stddev;//roi_area为各个大小为像素点8*8的小区域
	vector<post> roi;
	for (int i1 = 0; i1 < (grad.rows + 1 - side);)
	{
		for (int j1 = 0; j1 < (grad.cols + 1 - side);)
		{
			roi_area = grad(Rect(j1, i1, side, side));
			meanStdDev(roi_area, mat_mean, mat_stddev);
			double m, s;
			post p;
			m = mat_mean.at<double>(0, 0);
			s = mat_stddev.at<double>(0, 0);
			p.i = i1;
			p.j = j1;
			p.std = s;
			roi.push_back(p);
			j1 += step;
		}
		i1 += step;
	}
	//对roi中的方差排序
	auto bound_comp = bind(&Fire::comp, this, _1, _2);
	sort(roi.begin(), roi.end(), bound_comp);
	//计算N个区域的模糊程度
	int n = 64;
	double blur_reb[64];
	double blur_greb = 0;
	Mat f, b_ver, b_hor;//f代表原始图像，b代表滤波后的图像
	Mat d_fver(side - 1, side - 1, CV_64F);
	Mat d_bver(side - 1, side - 1, CV_64F);
	Mat d_fhor(side - 1, side - 1, CV_64F);
	Mat d_bhor(side - 1, side - 1, CV_64F);
	Mat d_vver(side - 1, side - 1, CV_64F);
	Mat d_vhor(side - 1, side - 1, CV_64F);
	for (int i2 = 0; i2 < 64; i2++)
	{
		double r1 = roi[i2].i;
		double c1 = roi[i2].j;
		f = mat_c(Rect(c1, r1, side, side));
		b_ver = h(Rect(c1, r1, side, side));//h为横向滤波结果
		b_hor = z(Rect(c1, r1, side, side));//z为纵向滤波结果
		double s_fver = 0.0;
		double s_vver = 0.0;
		double s_fhor = 0.0;
		double s_vhor = 0.0;
		for (int i3 = 0; i3 < side - 1; i3++)
		{
			for (int j3 = 0; j3 < side - 1; j3++)
			{
				d_fver.at<double >(i3, j3) = abs(f.at<double >(i3, j3 + 1) - f.at<double >(i3, j3));
				//printf("%d", d_fver.at<uchar>(i3, j3));
				d_bver.at<double >(i3, j3) = abs(b_ver.at<double >(i3, j3 + 1) - b_ver.at<double >(i3, j3));
				d_fhor.at<double >(i3, j3) = abs(f.at<double >(i3 + 1, j3) - f.at<double >(i3, j3));
				d_bhor.at<double >(i3, j3) = abs(b_hor.at<double >(i3 + 1, j3) - b_hor.at<double >(i3, j3));
				d_vver.at<double >(i3, j3) = (d_fver.at<double >(i3, j3) - d_bver.at<double>(i3, j3) > 0) ? (d_fver.at<double>(i3, j3) - d_bver.at<double>(i3, j3)) : 0;
				d_vhor.at<double >(i3, j3) = (d_fhor.at<double >(i3, j3) - d_bhor.at<double>(i3, j3) > 0) ? (d_fhor.at<double>(i3, j3) - d_bhor.at<double>(i3, j3)) : 0;
				s_fver += d_fver.at<double>(i3, j3);
				s_vver += d_vver.at<double>(i3, j3);
				s_fhor += d_fhor.at<double>(i3, j3);
				s_vhor += d_vhor.at<double>(i3, j3);
			}
		}
		if ((s_fver == 0) || (s_fhor == 0))
		{
			blur_reb[i2] = 0;
			//cout << "blur_reb[i2] is:" << blur_reb[i2] << endl;
		}
		else
		{
			double b_fver = double(s_fver - s_vver) / s_fver;
			double b_fhor = double(s_fhor - s_vhor) / s_fhor;
			double tep = (b_fver > b_fhor) ? b_fver : b_fhor;
			blur_reb[i2] = 1 - tep;
			//cout << "blur_reb[i2] is:"<<blur_reb[i2] << endl;
		}
	}
	for (int i4 = 0; i4 < n; i4++)
	{
		blur_greb += blur_reb[i4] / n;
	}
	cout << "模糊度" << blur_greb << endl;
	return blur_greb;
}
/*-------------------色偏测试--------------------
------------------------------------------------*/
double Fire::color_std(Mat &img)
{
	Mat hsv_img;
	Mat h, s, i;
	cvtColor(img, hsv_img, CV_BGR2HSV);
	Mat HSV[3];
	split(hsv_img, HSV);
	h = HSV[0];
	s = HSV[1];
	i = HSV[2];
	h = 255 * h / 180;    //将h结果转换为matlab那边所表示的h通道结果
						  //绘制src直方图
	Mat dstHist;  //定义存储直方图变量
	int dims = 1;  //需要统计的特征数目(只统计灰度值)
	float hranges[] = { 0, 256 };  //范围[0,255)//{0，256}？？
	const float* ranges[] = { hranges };
	int bins = 256;
	int channels = 0;
	calcHist(&h, 1, &channels, Mat(), dstHist, dims, &bins, ranges);
	Mat drawImage = Mat::zeros(Size(256, 256), CV_8UC3);
	float hh[256];
	for (int i = 0; i < 256; i++)
	{
		hh[i] = dstHist.at<float>(i);
	}
	double sum = 0;
	double avg;
	for (int j = 0; j<256; j++)
	{
		sum += hh[j];
	}
	avg = sum / 256;
	double temp = 0;
	double spow = 0;
	for (int k = 0; k<256; k++)
	{
		spow += (hh[k] - avg) * (hh[k] - avg);//方差
	}
	double std = spow / 256;   //matlab对应为n-1
	double stdstd = sqrt(std);    //开方求标准差
	Scalar mean;
	Scalar stddev;
	meanStdDev(h, mean, stddev);
	double mean_pxl = mean.val[0];
	double stddev_pxl = stddev.val[0];
	Mat zeroMatrix = Mat::zeros(h.size(), CV_8UC1);
	MatND zeroM_hist;
	calcHist(&zeroMatrix, 1, &channels, Mat(), zeroM_hist, dims, &bins, ranges);
	double h3[256];
	for (int i2 = 0; i2 < 256; i2++)
	{
		h3[i2] = zeroM_hist.at<float>(i2);
	}
	double sum2 = 0;
	double avg2;
	for (int j2 = 0; j2<256; j2++)
	{
		sum2 += h3[j2];
	}
	avg2 = sum2 / 256;
	double temp2 = 0;
	double spow2 = 0;
	for (int k2 = 0; k2 < 256; k2++)
	{
		spow2 += (h3[k2] - avg2) * (h3[k2] - avg2);
	}
	double std2 = spow2 / 256;
	double stdstd2 = sqrt(std2);  //开方求标准差
	double rate1 = 1 - stdstd / stdstd2;
	Mat zeroMatrix2 = Mat::zeros(h.size(), CV_8UC1);
	double num = 0;
	for (int i3 = 0; i3 < img.rows; i3++)
	{
		for (int j3 = 0; j3<img.cols; j3++)
		{
			if (num < (img.rows * img.cols) / 2)
			{
				zeroMatrix2.at<uchar>(i3, j3) = 255;
				num++;
			}
		}
	}
	Scalar mean1;  //直接计算标准差
	Scalar stddev1;
	meanStdDev(zeroMatrix2, mean1, stddev1);
	double mean_px2 = mean1.val[0];
	double stddev_px2 = stddev1.val[0];
	double stddev_pxlf = stddev_pxl;
	double stddev_px2f = stddev_px2;
	double rate2 = stddev_pxlf / stddev_px2f;
	if (rate2 > 1)
	{
		rate2 = 1;
	}
	double rate = rate1 + rate2;
	double final_std = rate / 2;
	cout << "色偏" << final_std << endl;
	return final_std;
}
int Fire::blurlevel(double blur_mos)
{//模糊度等级
	int blur_level;
	if ((0 <= blur_mos) && (blur_mos<0.2))
		blur_level = 1;
	else if ((0.2 <= blur_mos) && (blur_mos<0.4))
		blur_level = 2;
	else if ((0.4 <= blur_mos) && (blur_mos<0.6))
		blur_level = 3;
	else if ((0.6 <= blur_mos) && (blur_mos<0.8))
		blur_level = 4;
	else
		blur_level = 5;
	return blur_level;
}
int Fire::colorlevel(double color_mos)
{//色偏等级
	int color_level;
	if ((0 <= color_mos) && (color_mos<0.2))
		color_level = 1;
	else if ((0.2 <= color_mos) && (color_mos<0.4))
		color_level = 2;
	else if ((0.4 <= color_mos) && (color_mos<0.6))
		color_level = 3;
	else if ((0.6 <= color_mos) && (color_mos<0.8))
		color_level = 4;
	else
		color_level = 5;
	return color_level;
}
