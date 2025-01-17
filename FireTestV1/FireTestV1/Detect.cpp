#include "Fire.h"

//auto bound_comp = bind(&Fire::comp, this, _1, _2);
//sort(roi.begin(), roi.end(), bound_comp);
using namespace std;
using std::placeholders::_1;
using std::placeholders::_2;

void draw_circle(Mat &frame)//画圈函数
{
	circle(frame, Point(cvRound(frame.cols / 2), cvRound(frame.rows / 2)), 50, Scalar(0, 0, 255), 2, 8);
}

Fire::Fire(String video_address, vector<vector<vector<double>>> weight)
{
	Mat frame,frame1;
	Mat videodst;
	Mat picdst;
	double ConArea, ConLength;
	double last_area;
	double cur_area;
	double last_height;
	double cur_height;
	double last_length;
	double cur_length;
	vector<int> x, y;
	Point p;
	int s,num = 0;
	//---------------------------------------------------
	int time=1;//帧抽样间隔，单位：秒,手动1输入
	//-------------------------------------------------
	weightlocal = weight;//权重数组

	int blur_grade = 0;//模糊度等级
	int color_grade = 0;//色偏等级
	bool result;//存储当前帧是否有火焰
	//bool stop = false;
	double thre = 10;//存储火焰阈值，大于该值有火焰

	VideoCapture capture(video_address);//获取当前帧
	if (!capture.isOpened())
		cout << "视频无法继续读取" << endl;

	//double totalFrameNumber = capture.get(CV_CAP_PROP_FRAME_COUNT);//获取总帧数
	double rate = capture.get(CV_CAP_PROP_FPS);//获取帧率
	//cout << "总帧为:" << totalFrameNumber << endl;
	cout << "帧率为:" << rate << endl;
	cout << "帧抽样周期为:" << time << "s" << endl;

	while (1)
	{
		num++;
		if (!capture.read(frame1))
		{
			cout << "未能继续读取视频" << endl;
			break;
		}

		if (num % cvRound(time*rate) == 0)//实现帧抽样的核心语句
			frame = frame1.clone();
		else
			continue;

		cur_area = 0;
		cur_length = 0;
		cur_height = 0;
		//draw_circle(frame);
		imshow("Frame", frame);
		cout << "正在读取第" << num << "帧" << endl;

		//waitKey(int delay=0)当delay ≤ 0时会永远等待；当delay>0时会等待delay毫秒
		//当时间结束前没有按键按下时，返回值为-1；否则返回按键

		blur_grade = blurlevel(G_reblur(frame));//模糊度等级
		color_grade = colorlevel(color_std(frame));//色偏等级
		 //cout << "色偏等级为" << color_grade << endl;
		//cout << "模糊度等级为" << blur_grade << endl;
		picprogram(frame);

		videodst = videoCheckColorHSV(frame);
		//videodst = fireHSV;//同用一张火焰疑似区域

		fillHole(videodst, videodst);//虫洞填充
		Mat element = getStructuringElement(MORPH_RECT, Size(7, 7));
		//进行形态学操作  
		morphologyEx(videodst, videodst, MORPH_CLOSE, element);
		fillHole(videodst, videodst);

		vector<vector<Point>> contours;
		findContours(videodst, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);//寻找边缘

		for (int i = 0; i < (int)contours.size(); i++)
		{
			ConArea = contourArea(contours[i], false);
			ConLength = arcLength(contours[i], false);
			cur_area += ConArea;
			cur_length += ConLength;
			s = contours[i].size();
			if (s > 0)
			{
				for (int j = 0; j < s; j++)
				{
					p = contours[i].at(j);
					x.push_back(p.x);
					y.push_back(p.y);
				}
			}
		}
		vector<int>::iterator buttom = max_element(y.begin(), y.end());
		vector<int>::iterator top = min_element(y.begin(), y.end());
		//cur_height = *buttom - *top + 1;//??
		if (num == cvRound(time*rate))//num为第一抽样帧的帧数
		{
			last_height = cur_height;
			last_area = cur_area;
			last_length = cur_area;
		}
		else if (date[0] == 0)
		{
			cout << "当前帧不存在火焰" << endl;
			num +=1;
			frame.release();
			continue;
		}
		else
		{
			cur_height = *buttom - *top + 1;//??
			date[5] = Hei_dy(last_height, cur_height);
			last_height = cur_height;//将当前帧数据赋为上一帧
			cout << "高度变化" << date[5] << endl;
			date[6] = Are_dy(last_area, cur_area);
			cout << "面积变化率" << date[6] << endl;
			date[7] = Rou_dy(last_length, cur_length);
			cout << "边缘抖动" << date[7] << endl;
			last_area = cur_area;
			last_length = cur_length;
		}
		result = getresult(color_grade, blur_grade, thre);//当前帧火焰判断
		if (result)
			cout << "当前帧存在火焰" << endl;
		else
			cout << "当前帧不存在火焰" << endl;

		x.clear();
		y.clear();

		//if (cv::waitKey(100) >= 0)
		//break;
		//如果要imshow，取消这两行注释
		//frame.release();
		cout << endl;

		//waitKey(int delay=0)当delay ≤ 0时会永远等待；当delay>0时会等待delay毫秒
		//int c = waitKey(delay);
		//if ((char)c == 27)//按下ESC或者到达指定的结束帧后退出读取视频
		//{
		//	stop = true;
		//}
		////按下按键后会停留在当前帧，等待下一次按键
		//if (c >= 0)
		//{
		//	waitKey(0);
		//}
		//num++;
	}
	capture.release();
	//cv::destroyAllWindows();
	//waitKey(0);
	return;
}

//Fire::~Fire(){}

//八大特征
float Fire::Cen_hei(vector<Point> contour)//物体的质心高度比
{
	Point p;
	int s;
	s = contour.size();
	vector<int> x, y;
	if (s > 0)
	{
		for (int i = 0; i < s; i++)
		{
			p = contour.at(i);

			x.push_back(p.x);

			y.push_back(p.y);
		}
	}
	vector<int>::iterator buttom = max_element(y.begin(), y.end());
	vector<int>::iterator top = min_element(y.begin(), y.end());
	int height = *buttom - *top + 1;
	Moments mu;
	Point2f mc;
	mu = moments(contour, false);
	mc = Point2d(mu.m10 / mu.m00, mu.m01 / mu.m00);//质心坐标
	double cen = mc.y;
	float ratio_ch = (*buttom - cen) / height;
	return ratio_ch;
}
double Fire::Factor(vector<Point> contour)//圆形度
{
	double factor = (contourArea(contour) * 4 * CV_PI) /
		(pow(arcLength(contour, true), 2));
	return factor;
}
bool Fire::R_max(Mat fst, Mat dst)
{//红色比例特征
	Mat multiRGB[3];
	int a = fst.channels();
	int counter = countNonZero(dst);

	uchar* p;
	uchar* p0;
	uchar* p1;
	uchar* p2;
	for (int i = 0; i < fst.rows; i++)
	{
		p = dst.ptr<uchar>(i);
		for (int j = 0; j < fst.cols; j++)
		{
			if (p[j] == 0)
				fst.at<Vec3b>(i, j) = { 0,0,0 };
		}
	}
	float B = 0;
	float G = 0;
	float R = 0;
	split(fst, multiRGB); //将图片拆分成R,G,B,三通道的颜色  
	for (int i = 0; i < fst.rows; i++)
	{
		p0 = multiRGB[0].ptr<uchar>(i);
		p1 = multiRGB[1].ptr<uchar>(i);
		p2 = multiRGB[2].ptr<uchar>(i);
		for (int j = 0; j < fst.cols; j++)
		{
			B = B + p0[j]; //计算区域内像素R,G,B总和
			G = G + p1[j];
			R = R + p2[j];
		}
	}
	float B_ave = B / counter;
	float G_ave = G / counter;
	float R_ave = R / counter;

	cout << "B" << B_ave << endl;
	cout << "G" << G_ave << endl;
	cout << "R" << R_ave << endl;
	
	if (R_ave >= G_ave && R_ave >= B_ave)
		return true;
	else
		return false;
}
float Fire::R_t(Mat fst, Mat dst)
{//红色分量
	Mat multiRGB[3];
	int a = fst.channels();
	int counter = countNonZero(dst);
	if (counter == 0)
		return 0;
	uchar* p;
	uchar* p0;
	uchar* p1;
	uchar* p2;

	for (int i = 0; i < fst.rows; i++)
	{
		p = dst.ptr<uchar>(i);
		for (int j = 0; j < fst.cols; j++)
		{
			if (p[j] == 0)
				fst.at<Vec3b>(i, j) = { 0,0,0 };
		}
	}
	float B = 0, G = 0, R = 0;

	split(fst, multiRGB); //将图片拆分成R,G,B,三通道的颜色  
						  //cout << "b是否连续" << multiRGB[0].isContinuous() << endl;
						  //cout << "g是否连续" << multiRGB[1].isContinuous() << endl;
						  //cout << "r是否连续" << multiRGB[2].isContinuous() << endl;
	for (int i = 0; i < fst.rows; i++)
	{
		p0 = multiRGB[0].ptr<uchar>(i);
		p1 = multiRGB[1].ptr<uchar>(i);
		p2 = multiRGB[2].ptr<uchar>(i);
		for (int j = 0; j < fst.cols; j++)
		{

			B = B + p0[j]; //计算区域内像素R,G,B总和
			G = G + p1[j];
			R = R + p2[j];
		}
	}
	float B_ave = B / counter;
	float G_ave = G / counter;
	float R_ave = R / counter;
	cout << "B" << B_ave << endl;
	cout << "G" << G_ave << endl;
	cout << "R" << R_ave << endl;
	multiRGB[0].release();
	multiRGB[1].release();
	multiRGB[2].release();
	if (R_ave >= G_ave && R_ave >= B_ave)
		return R_ave;
	else
		return 0;
}
float Fire::Rec(vector<Point> contour)
{//矩形度
	int area = contourArea(contour);

	RotatedRect minRect = minAreaRect(contour);
	Size2f size = minRect.size;
	float rectarea = size.area();
	float rec = area / rectarea;
	return rec;
}
double Fire::Rou(vector<Point> contour)
{//边缘粗糙度
	vector<Point> hull;
	convexHull(Mat(contour), hull, false);

	double hullLength, dstLength, hullsize;
	hullLength = arcLength(hull, false);
	hullsize = contourArea(hull, false);
	dstLength = arcLength(contour, false);
	double roughness;
	roughness = dstLength / hullLength;

	return roughness;
}
double Fire::Are_dy(double area1, double area2)
{//面积变化率
	double area3 = abs(area1 - area2);
	double max_area = max(area1, area2);
	double area_dy = area3 / max_area;
	return area_dy;
}
double Fire::Rou_dy(double length1, double length2)
{//边缘抖动变化率
	double length3 = abs(length1 - length2);
	double max_length = max(length1, length2);
	double rou_dy = length3 / max_length;
	return rou_dy;
}
double Fire::Hei_dy(double height1, double height2)
{//高度变化率
	double height3 = abs(height1 - height2);
	double max_height = max(height1, height2);
	double hei_dy = height3 / max_height;
	return hei_dy;
}

//图像预处理
void Fire::picprogram(Mat pic)
{
	//静态
	Mat oriImg = pic;
	//imshow("oriImg", oriImg);
	fireHSV = picCheckColorHSV(oriImg);//疑似火焰区域
	//imshow("HSVbefore", fireHSV);
	//processiamge(fireHSV);画圈
	fillHole(fireHSV, fireHSV);
	date[0] = R_t(oriImg, fireHSV);
	oriImg.release();
	if (date[0] != 0)
		cout << "该区域红色比例分量最大，符合该特征判别" << endl;
	else
	{
		cout << "该区域红色比例分量不是最大，不符合该特征判别" << endl;
		return;
	}
	cout << "区域红色特征参量为" << date[0] << endl;
	imshow("picarea", fireHSV);//显示疑似火焰区域
	waitKey(200);
	vector<vector<Point>> contours;
	findContours(fireHSV, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	//
	for (int i = 0; i < (int)contours.size(); i++)
	{
		cout << "第" << i + 1 << "块区域" << endl;
		date[1] = Factor(contours[i]);
		cout << "区域圆形度为" << date[1] << endl;
		date[2] = Rec(contours[i]);
		cout << "区域矩形度为" << date[2] << endl;
		date[3] = Rou(contours[i]);
		cout << "区域边缘粗糙程度为" << date[3] << endl;
		date[4] = Cen_hei(contours[i]);
		cout << "区域质心高度比为" << date[4] << endl;
	}
}
Mat Fire::picCheckColorHSV(Mat &inImg)
{//火焰区域提取
	Mat fireImg;
	fireImg.create(inImg.size(), CV_8UC1);
	Mat imgHSV;
	imgHSV.create(inImg.size(), CV_8UC1);
	cvtColor(inImg, imgHSV, COLOR_BGR2HSV);//转为HSV 
	int a = imgHSV.channels();
	Mat multiHSV[3];
	split(imgHSV, multiHSV);
	//
	uchar* p0;
	uchar* p1;
	uchar* p2;
	for (int i = 0; i < inImg.rows; i++)
	{
		p0 = multiHSV[0].ptr<uchar>(i);
		p1 = multiHSV[1].ptr<uchar>(i);
		p2 = multiHSV[2].ptr<uchar>(i);
		for (int j = 0; j < inImg.cols; j++)
		{
			float H, S, V;
			H = p0[j];
			S = p1[j];
			V = p2[j];
			if (H > 15 && H < 100 && S>20 && S <= 150 && V>100 && V <= 255)
			{
				fireImg.at<uchar>(i, j) = 255;
			}
			else
			{
				fireImg.at<uchar>(i, j) = 0;
			}
		}
	}
	multiHSV[0].release();
	multiHSV[1].release();
	multiHSV[2].release();
	//erode(fireImg, fireImg, Mat(3, 3, CV_8UC1));  
	//GaussianBlur(fireImg, fireImg, Size(5, 5), 0, 0);  
	medianBlur(fireImg, fireImg, 7);
	dilate(fireImg, fireImg, Mat(5, 5, CV_8UC1));
	Size size1 = inImg.size();
	double area1 = size1.area();
	//cout << "sjsj:" << area1 << endl;
	double contourarea;
	vector<vector<Point>> contours, contours2;
	findContours(fireImg, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	for (int i = 0; i < (int)contours.size(); i++)
	{
		contourarea = contourArea(contours[i], false);
		if (contourarea < (area1 / 1000))
			contours2.push_back(contours[i]);
	}
	drawContours(fireImg, contours2, -1, Scalar(0), CV_FILLED);
	return fireImg;
}
Mat Fire::videoCheckColorHSV(Mat & inImg)
{//没用到
	Mat fireImg;
	fireImg.create(inImg.size(), CV_8UC1);
	Mat imgHSV;
	imgHSV.create(inImg.size(), CV_8UC1);
	cvtColor(inImg, imgHSV, COLOR_BGR2HSV);//转为HSV 
	int a = imgHSV.channels();
	Mat multiHSV[3];
	split(imgHSV, multiHSV);
	imgHSV.release();
	uchar* p0;
	uchar* p1;
	uchar* p2;
	for (int i = 0; i < inImg.rows; i++)
	{
		p0 = multiHSV[0].ptr<uchar>(i);
		p1 = multiHSV[1].ptr<uchar>(i);
		p2 = multiHSV[2].ptr<uchar>(i);
		for (int j = 0; j < inImg.cols; j++)
		{
			float H, S, V;
			H = p0[j];
			S = p1[j];
			V = p2[j];
			if (H > 15 && H < 100 && S>20 && S <= 150 && V>100 && V <= 255)
			{
				fireImg.at<uchar>(i, j) = 255;
			}
			else
			{
				fireImg.at<uchar>(i, j) = 0;
			}
		}

	}

	multiHSV[0].release();
	multiHSV[1].release();
	multiHSV[2].release();

	medianBlur(fireImg, fireImg, 7);
	return fireImg;
}
void Fire::fillHole(const cv::Mat srcimage, cv::Mat & dstimage)
{
	Size m_Size = srcimage.size();
	Mat temimage = Mat::zeros(m_Size.height + 2, m_Size.width + 2, srcimage.type());//延展图像    
	srcimage.copyTo(temimage(Range(1, m_Size.height + 1), Range(1, m_Size.width + 1)));
	floodFill(temimage, Point(0, 0), Scalar(255));
	Mat cutImg;//裁剪延展的图像    
	temimage(Range(1, m_Size.height + 1), Range(1, m_Size.width + 1)).copyTo(cutImg);
	dstimage = srcimage | (~cutImg);
}

bool Fire::getresult(int color_grade, int blur_grade, double th)
{//最终判断
	double result = 0;
	for (int ii = 0; ii < 8; ii++)
	{
		result += date[ii] * weightlocal[color_grade][blur_grade][ii];
	}
	if (result >= th)
		return true;
	else
		return false;
}

Mat Fire::conv(Mat & in)
{
	Mat out = cv::Mat::zeros(in.rows, in.cols, CV_64FC1);

	Mat temp = cv::Mat::zeros(in.rows + 8, in.cols + 8, CV_64FC1);
	Mat temp2 = cv::Mat::zeros(in.rows + 8, in.cols + 8, CV_64FC1);
	for (int i = 4; i < (in.rows + 4); i++)//移位
	{
		for (int j = 4; j < (in.cols + 4); j++)
		{
			temp.at<double>(i, j) = in.at<double>(i - 4, j - 4);
		}
	}

	double avg = 0;

	for (int i1 = 4; i1 < (in.rows + 4); i1++)//卷积
	{
		for (int j1 = 4; j1 < (in.cols + 4); j1++)
		{
			for (int k = j1 - 4; k < j1 + 5; k++)
			{
				avg += (temp.at<double>(i1, k)) / 9;
			}
			temp2.at<double>(i1, j1) = avg;
			avg = 0;
		}
	}

	for (int i = 4; i < (in.rows + 4); i++)//回移
	{
		for (int j = 4; j < (in.cols + 4); j++)
		{
			out.at<double>(i - 4, j - 4) = temp2.at<double>(i, j);
		}
	}
	return out;
}
struct post
{
	float std;
	int i;
	int j;
};

bool Fire::comp(const post & a, const post & b)
{
	return a.std > b.std;
}