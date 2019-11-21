#include "Fire.h"

//auto bound_comp = bind(&Fire::comp, this, _1, _2);
//sort(roi.begin(), roi.end(), bound_comp);
using namespace std;
using std::placeholders::_1;
using std::placeholders::_2;

void draw_circle(Mat &frame)//��Ȧ����
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
	int time=1;//֡�����������λ����,�ֶ�1����
	//-------------------------------------------------
	weightlocal = weight;//Ȩ������

	int blur_grade = 0;//ģ���ȵȼ�
	int color_grade = 0;//ɫƫ�ȼ�
	bool result;//�洢��ǰ֡�Ƿ��л���
	//bool stop = false;
	double thre = 10;//�洢������ֵ�����ڸ�ֵ�л���

	VideoCapture capture(video_address);//��ȡ��ǰ֡
	if (!capture.isOpened())
		cout << "��Ƶ�޷�������ȡ" << endl;

	//double totalFrameNumber = capture.get(CV_CAP_PROP_FRAME_COUNT);//��ȡ��֡��
	double rate = capture.get(CV_CAP_PROP_FPS);//��ȡ֡��
	//cout << "��֡Ϊ:" << totalFrameNumber << endl;
	cout << "֡��Ϊ:" << rate << endl;
	cout << "֡��������Ϊ:" << time << "s" << endl;

	while (1)
	{
		num++;
		if (!capture.read(frame1))
		{
			cout << "δ�ܼ�����ȡ��Ƶ" << endl;
			break;
		}

		if (num % cvRound(time*rate) == 0)//ʵ��֡�����ĺ������
			frame = frame1.clone();
		else
			continue;

		cur_area = 0;
		cur_length = 0;
		cur_height = 0;
		//draw_circle(frame);
		imshow("Frame", frame);
		cout << "���ڶ�ȡ��" << num << "֡" << endl;

		//waitKey(int delay=0)��delay �� 0ʱ����Զ�ȴ�����delay>0ʱ��ȴ�delay����
		//��ʱ�����ǰû�а�������ʱ������ֵΪ-1�����򷵻ذ���

		blur_grade = blurlevel(G_reblur(frame));//ģ���ȵȼ�
		color_grade = colorlevel(color_std(frame));//ɫƫ�ȼ�
		 //cout << "ɫƫ�ȼ�Ϊ" << color_grade << endl;
		//cout << "ģ���ȵȼ�Ϊ" << blur_grade << endl;
		picprogram(frame);

		videodst = videoCheckColorHSV(frame);
		//videodst = fireHSV;//ͬ��һ�Ż�����������

		fillHole(videodst, videodst);//�涴���
		Mat element = getStructuringElement(MORPH_RECT, Size(7, 7));
		//������̬ѧ����  
		morphologyEx(videodst, videodst, MORPH_CLOSE, element);
		fillHole(videodst, videodst);

		vector<vector<Point>> contours;
		findContours(videodst, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);//Ѱ�ұ�Ե

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
		if (num == cvRound(time*rate))//numΪ��һ����֡��֡��
		{
			last_height = cur_height;
			last_area = cur_area;
			last_length = cur_area;
		}
		else if (date[0] == 0)
		{
			cout << "��ǰ֡�����ڻ���" << endl;
			num +=1;
			frame.release();
			continue;
		}
		else
		{
			cur_height = *buttom - *top + 1;//??
			date[5] = Hei_dy(last_height, cur_height);
			last_height = cur_height;//����ǰ֡���ݸ�Ϊ��һ֡
			cout << "�߶ȱ仯" << date[5] << endl;
			date[6] = Are_dy(last_area, cur_area);
			cout << "����仯��" << date[6] << endl;
			date[7] = Rou_dy(last_length, cur_length);
			cout << "��Ե����" << date[7] << endl;
			last_area = cur_area;
			last_length = cur_length;
		}
		result = getresult(color_grade, blur_grade, thre);//��ǰ֡�����ж�
		if (result)
			cout << "��ǰ֡���ڻ���" << endl;
		else
			cout << "��ǰ֡�����ڻ���" << endl;

		x.clear();
		y.clear();

		//if (cv::waitKey(100) >= 0)
		//break;
		//���Ҫimshow��ȡ��������ע��
		//frame.release();
		cout << endl;

		//waitKey(int delay=0)��delay �� 0ʱ����Զ�ȴ�����delay>0ʱ��ȴ�delay����
		//int c = waitKey(delay);
		//if ((char)c == 27)//����ESC���ߵ���ָ���Ľ���֡���˳���ȡ��Ƶ
		//{
		//	stop = true;
		//}
		////���°������ͣ���ڵ�ǰ֡���ȴ���һ�ΰ���
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

//�˴�����
float Fire::Cen_hei(vector<Point> contour)//��������ĸ߶ȱ�
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
	mc = Point2d(mu.m10 / mu.m00, mu.m01 / mu.m00);//��������
	double cen = mc.y;
	float ratio_ch = (*buttom - cen) / height;
	return ratio_ch;
}
double Fire::Factor(vector<Point> contour)//Բ�ζ�
{
	double factor = (contourArea(contour) * 4 * CV_PI) /
		(pow(arcLength(contour, true), 2));
	return factor;
}
bool Fire::R_max(Mat fst, Mat dst)
{//��ɫ��������
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
	split(fst, multiRGB); //��ͼƬ��ֳ�R,G,B,��ͨ������ɫ  
	for (int i = 0; i < fst.rows; i++)
	{
		p0 = multiRGB[0].ptr<uchar>(i);
		p1 = multiRGB[1].ptr<uchar>(i);
		p2 = multiRGB[2].ptr<uchar>(i);
		for (int j = 0; j < fst.cols; j++)
		{
			B = B + p0[j]; //��������������R,G,B�ܺ�
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
{//��ɫ����
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

	split(fst, multiRGB); //��ͼƬ��ֳ�R,G,B,��ͨ������ɫ  
						  //cout << "b�Ƿ�����" << multiRGB[0].isContinuous() << endl;
						  //cout << "g�Ƿ�����" << multiRGB[1].isContinuous() << endl;
						  //cout << "r�Ƿ�����" << multiRGB[2].isContinuous() << endl;
	for (int i = 0; i < fst.rows; i++)
	{
		p0 = multiRGB[0].ptr<uchar>(i);
		p1 = multiRGB[1].ptr<uchar>(i);
		p2 = multiRGB[2].ptr<uchar>(i);
		for (int j = 0; j < fst.cols; j++)
		{

			B = B + p0[j]; //��������������R,G,B�ܺ�
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
{//���ζ�
	int area = contourArea(contour);

	RotatedRect minRect = minAreaRect(contour);
	Size2f size = minRect.size;
	float rectarea = size.area();
	float rec = area / rectarea;
	return rec;
}
double Fire::Rou(vector<Point> contour)
{//��Ե�ֲڶ�
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
{//����仯��
	double area3 = abs(area1 - area2);
	double max_area = max(area1, area2);
	double area_dy = area3 / max_area;
	return area_dy;
}
double Fire::Rou_dy(double length1, double length2)
{//��Ե�����仯��
	double length3 = abs(length1 - length2);
	double max_length = max(length1, length2);
	double rou_dy = length3 / max_length;
	return rou_dy;
}
double Fire::Hei_dy(double height1, double height2)
{//�߶ȱ仯��
	double height3 = abs(height1 - height2);
	double max_height = max(height1, height2);
	double hei_dy = height3 / max_height;
	return hei_dy;
}

//ͼ��Ԥ����
void Fire::picprogram(Mat pic)
{
	//��̬
	Mat oriImg = pic;
	//imshow("oriImg", oriImg);
	fireHSV = picCheckColorHSV(oriImg);//���ƻ�������
	//imshow("HSVbefore", fireHSV);
	//processiamge(fireHSV);��Ȧ
	fillHole(fireHSV, fireHSV);
	date[0] = R_t(oriImg, fireHSV);
	oriImg.release();
	if (date[0] != 0)
		cout << "�������ɫ����������󣬷��ϸ������б�" << endl;
	else
	{
		cout << "�������ɫ��������������󣬲����ϸ������б�" << endl;
		return;
	}
	cout << "�����ɫ��������Ϊ" << date[0] << endl;
	imshow("picarea", fireHSV);//��ʾ���ƻ�������
	waitKey(200);
	vector<vector<Point>> contours;
	findContours(fireHSV, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	//
	for (int i = 0; i < (int)contours.size(); i++)
	{
		cout << "��" << i + 1 << "������" << endl;
		date[1] = Factor(contours[i]);
		cout << "����Բ�ζ�Ϊ" << date[1] << endl;
		date[2] = Rec(contours[i]);
		cout << "������ζ�Ϊ" << date[2] << endl;
		date[3] = Rou(contours[i]);
		cout << "�����Ե�ֲڳ̶�Ϊ" << date[3] << endl;
		date[4] = Cen_hei(contours[i]);
		cout << "�������ĸ߶ȱ�Ϊ" << date[4] << endl;
	}
}
Mat Fire::picCheckColorHSV(Mat &inImg)
{//����������ȡ
	Mat fireImg;
	fireImg.create(inImg.size(), CV_8UC1);
	Mat imgHSV;
	imgHSV.create(inImg.size(), CV_8UC1);
	cvtColor(inImg, imgHSV, COLOR_BGR2HSV);//תΪHSV 
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
{//û�õ�
	Mat fireImg;
	fireImg.create(inImg.size(), CV_8UC1);
	Mat imgHSV;
	imgHSV.create(inImg.size(), CV_8UC1);
	cvtColor(inImg, imgHSV, COLOR_BGR2HSV);//תΪHSV 
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
	Mat temimage = Mat::zeros(m_Size.height + 2, m_Size.width + 2, srcimage.type());//��չͼ��    
	srcimage.copyTo(temimage(Range(1, m_Size.height + 1), Range(1, m_Size.width + 1)));
	floodFill(temimage, Point(0, 0), Scalar(255));
	Mat cutImg;//�ü���չ��ͼ��    
	temimage(Range(1, m_Size.height + 1), Range(1, m_Size.width + 1)).copyTo(cutImg);
	dstimage = srcimage | (~cutImg);
}

bool Fire::getresult(int color_grade, int blur_grade, double th)
{//�����ж�
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
	for (int i = 4; i < (in.rows + 4); i++)//��λ
	{
		for (int j = 4; j < (in.cols + 4); j++)
		{
			temp.at<double>(i, j) = in.at<double>(i - 4, j - 4);
		}
	}

	double avg = 0;

	for (int i1 = 4; i1 < (in.rows + 4); i1++)//����
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

	for (int i = 4; i < (in.rows + 4); i++)//����
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