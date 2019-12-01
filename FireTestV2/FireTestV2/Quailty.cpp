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
/*-------------------ģ���Ȳ���-------------------
--------------------------------------------------
������������֮���ֵ�ı仯,�������ͼ����������֮��Ĳ�ֵ
�Ȳο�ͼ����������֮��Ĳ�ֵ����ԭʼͼ����������෴��ԭʼͼ���ģ����*/
double Fire::G_reblur(Mat src)
{
 //rgb2gray
	Mat in;
	cvtColor(src, in, CV_BGR2GRAY);
	Mat mat_c;
	// ֻ��32FC1, 32FC2, 64FC1, 64FC2��֧�ֳ˷�
	in.convertTo(mat_c, CV_64F, 1.0, 0);
	//������˲����
	Mat h = conv(mat_c);
	/*for (int a = 79; a < 84; a++)
	printf(" %f", h.at<double>(49, a));
	printf("\n");*/
	//�����˲�
	Mat z;
	Mat m1;
	Mat m2;
	transpose(mat_c, m1);
	m2 = conv(m1);
	transpose(m2, z);
	//����ͼ��ı�Ե���
	Mat mat_b;
	Mat grad;
	Mat grad_x, grad_y;//x��y�����ϵ��ݶ�
	Mat abs_grad_x, abs_grad_y;
	//Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
	//��ȡ�ݶ���Ϣ
	Sobel(mat_c, grad_x, CV_64F, 1, 0, 3, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);
	//Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
	Sobel(mat_c, grad_y, CV_64F, 0, 1, 3, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
	//���㲢��ȡͼ�������д�СΪside*side,����Ϊstep������ķ����λ�ã�����vector roi ��
	int side = 8;
	int	step = 4;
	Mat roi_area, mat_mean, mat_stddev;//roi_areaΪ������СΪ���ص�8*8��С����
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
	//��roi�еķ�������
	auto bound_comp = bind(&Fire::comp, this, _1, _2);
	sort(roi.begin(), roi.end(), bound_comp);
	//����N�������ģ���̶�
	int n = 64;
	double blur_reb[64];
	double blur_greb = 0;
	Mat f, b_ver, b_hor;//f����ԭʼͼ��b�����˲����ͼ��
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
		b_ver = h(Rect(c1, r1, side, side));//hΪ�����˲����
		b_hor = z(Rect(c1, r1, side, side));//zΪ�����˲����
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
	cout << "ģ����" << blur_greb << endl;
	return blur_greb;
}
/*-------------------ɫƫ����--------------------
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
	h = 255 * h / 180;    //��h���ת��Ϊmatlab�Ǳ�����ʾ��hͨ�����
						  //����srcֱ��ͼ
	Mat dstHist;  //����洢ֱ��ͼ����
	int dims = 1;  //��Ҫͳ�Ƶ�������Ŀ(ֻͳ�ƻҶ�ֵ)
	float hranges[] = { 0, 256 };  //��Χ[0,255)//{0��256}����
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
		spow += (hh[k] - avg) * (hh[k] - avg);//����
	}
	double std = spow / 256;   //matlab��ӦΪn-1
	double stdstd = sqrt(std);    //�������׼��
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
	double stdstd2 = sqrt(std2);  //�������׼��
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
	Scalar mean1;  //ֱ�Ӽ����׼��
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
	cout << "ɫƫ" << final_std << endl;
	return final_std;
}
int Fire::blurlevel(double blur_mos)
{//ģ���ȵȼ�
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
{//ɫƫ�ȼ�
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
