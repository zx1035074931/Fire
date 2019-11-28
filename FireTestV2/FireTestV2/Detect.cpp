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
	//---------------------------------------------------
	int time = 1;//֡�����������λ����,�ֶ�1����	 
	//-------------------------------------------------
	int s, num = 0;
	int blur_grade = 0;//ģ���ȵȼ�
	int color_grade = 0;//ɫƫ�ȼ�
	Mat frame,frame1, frame_color;
	Mat videodst;
	Mat picdst;
	double ConArea, ConLength;
	double last_area,cur_area;
	double last_height,cur_height;
	double last_length,cur_length;
	double thre = 10;//�洢������ֵ�����ڸ�ֵ�л���
	vector<int> x, y;
	Point p;
	weightlocal = weight;//Ȩ������
	bool result;//�洢��ǰ֡�Ƿ��л���
	//bool stop = false;

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
		draw_circle(frame);
		imshow("Frame", frame);
		cout << "���ڶ�ȡ��" << num << "֡" << endl;

		blur_grade = blurlevel(G_reblur(frame));//ģ���ȵȼ�
		color_grade = colorlevel(color_std(frame));//ɫƫ�ȼ�
		//cout << "ɫƫ�ȼ�Ϊ" << color_grade << endl;
		//cout << "ģ���ȵȼ�Ϊ" << blur_grade << endl;
		picprogram(frame); //frame��ɫ
		frame_color = frame;
		imshow("frame_color", frame_color);//��ɫ

		videodst = picCheckColorHSV(frame);
		imshow("picCheckColorHSV֮��ĵ�videodst", videodst);//�ڰ�

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
			ConArea = contourArea(contours[i], false);//contourArea��������е����
			ConLength = arcLength(contours[i], false);//����������ܳ�
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
			date[5] = change_dy(last_height, cur_height);//�߶ȱ仯
			last_height = cur_height;//����ǰ֡���ݸ�Ϊ��һ֡
			cout << "�߶ȱ仯" << date[5] << endl;
			date[6] = change_dy(last_area, cur_area);//����仯��
			cout << "����仯��" << date[6] << endl;
			date[7] = change_dy(last_length, cur_length);//��Ե����
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
	}
	capture.release();
	//cv::destroyAllWindows();
	//waitKey(0);
	return;
}
//Fire::~Fire(){}

//ͼ��Ԥ����
void Fire::picprogram(Mat pic)
{
	//��̬
	Mat oriImg = pic;
	//imshow("oriImg", oriImg);
	fireHSV = picCheckColorHSV(oriImg);//���ƻ�������
	//imshow("fireHSV", fireHSV);
	fillHole(fireHSV, fireHSV);
	//imshow("fireHSV", fireHSV);
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
	imshow("picprogram�е�picarea", fireHSV);//��ʾ���ƻ�������
	waitKey(200);
	vector<vector<Point>> contours;
	findContours(fireHSV, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	/*findContours���ڼ��ͼ���Ե
	����1������ͼ�񣺵�ͨ��ͼ����󣬿����ǻҶ�ͼ���������õ�һ���Ǿ���Canny��������˹�ȱ�Ե������Ӵ�����Ķ�ֵͼ��
	����2��contours����Ϊ��vector<vector<Point>> contours������һ��˫��������������ÿ��Ԫ�ر�����һ����������Point���ɵĵ�ļ��ϵ���������ÿһ��㼯����һ���������ж���������contours���ж���Ԫ�أ�
	����3��CV_RETR_LIST���������ļ���ģʽ��������е�����
	����4��CV_CHAIN_APPROX_NONE����������߽������������������㵽contours�����ڣ�
	*/
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
	Mat fireImg,imgHSV, multiHSV[3], multiRGB[3];
	//creat:����һ��ָ����С(Size)��ָ������type(CV_8UC1, CV_16SC1, CV_32FC3)��ͼ�����ľ�����,Matֻ�����˾���ͷ
	fireImg.create(inImg.size(), CV_8UC1);
	imgHSV.create(inImg.size(), CV_8UC1);
	cvtColor(inImg, imgHSV, COLOR_BGR2HSV);//cvtcolor():��ɫ�ռ�ת������
	int b = inImg.channels();
	int a = imgHSV.channels();
	split(inImg, multiRGB);
	split(imgHSV, multiHSV);

	uchar *p0, *p1, *p2, *q0, *q1, *q2;
	for (int i = 0; i < inImg.rows; i++)
	{
		q0 = multiRGB[0].ptr<uchar>(i);//B
		q1 = multiRGB[1].ptr<uchar>(i);//G
		q2 = multiRGB[2].ptr<uchar>(i);//R

		p0 = multiHSV[0].ptr<uchar>(i);//H
		p1 = multiHSV[1].ptr<uchar>(i);//S
		p2 = multiHSV[2].ptr<uchar>(i);//V

		for (int j = 0; j < inImg.cols; j++)
		{
			float H = p0[j], S = p1[j], V = p2[j];
			float B = q0[j], G = q1[j], R = q2[j];
			if (H > 15 && H < 100 && S>20 && S <= 150 && V>100 && V <= 255)
				continue;
			//fireImg.at<uchar>(i, j) = 255;//at: ȡ��ͼ����i��j�еĵ�,�˾佫���ƻ��������Ϊ��ɫ
			else
				fireImg.at<uchar>(i, j) = 0;// (R + G + B) / 3; //at: ȡ��ͼ����i��j�еĵ�,�˾佫�����ƻ��������Ϊ��ɫ
		}
	}
	multiRGB[0].release();
	multiRGB[1].release();
	multiRGB[2].release();

	multiHSV[0].release();
	multiHSV[1].release();
	multiHSV[2].release();
	//erode(fireImg, fireImg, Mat(3, 3, CV_8UC1));  
	//GaussianBlur(fireImg, fireImg, Size(5, 5), 0, 0);  
	medianBlur(fireImg, fireImg, 7);//medianBlur��ֵ�˲�
	dilate(fireImg, fireImg, Mat(5, 5, CV_8UC1));//dilate���ͺ���
	Size size1 = inImg.size();
	double area1 = size1.area(), contourarea;
	//cout << "sjsj:" << area1 << endl;
	vector<vector<Point>> contours, contours2;
	findContours(fireImg, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	for (int i = 0; i < (int)contours.size(); i++)//??
	{
		contourarea = contourArea(contours[i], false);
		if (contourarea < (area1 / 1000))
			contours2.push_back(contours[i]); //push_back:��vector��������Ϊ��vectorβ������һ������
	}
	drawContours(fireImg, contours2, -1, Scalar(0), CV_FILLED);//������
	imshow("picCheckColorHSV�Ľ��", fireImg);
		
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

//�˴�����
float Fire::R_t(Mat fst, Mat dst)//��ɫ����
{
	Mat multiRGB[3];
	int a = fst.channels();
	int counter = countNonZero(dst); //�Զ�ֵ��ͼ��ִ��countNonZero,�ɵ÷������ص���.
	if (counter == 0)
		return 0;
	uchar *p,*p0,*p1,*p2;

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
		p0 = multiRGB[0].ptr<uchar>(i);//B  V
		p1 = multiRGB[1].ptr<uchar>(i);//G  S
		p2 = multiRGB[2].ptr<uchar>(i);//R Խ�ӽ���ɫ�ĵط��ں�ɫͨ��Խ�ӽ���ɫ  H
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
	/*	��Ҫ����ɫ���������Ƿ����������������֮������
	if (R_ave >= G_ave && R_ave >= B_ave)
	return true;
	else
	return false;
	*/
	multiRGB[0].release();
	multiRGB[1].release();
	multiRGB[2].release();
	if (R_ave >= G_ave && R_ave >= B_ave)
		return R_ave;
	else
		return 0;
}

float Fire::Rec(vector<Point> contour)//���ζ�
{
	int area = contourArea(contour);
	RotatedRect minRect = minAreaRect(contour);
	Size2f size = minRect.size;
	float rectarea = size.area();
	float rec = area / rectarea;
	return rec;
}

double Fire::Factor(vector<Point> contour)//Բ�ζ�
{
	double factor = (contourArea(contour) * 4 * CV_PI) /
		(pow(arcLength(contour, true), 2));
	return factor;
}

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

double Fire::Rou(vector<Point> contour) // ��Ե�ֲڶ�
{
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

double Fire::change_dy(double para1, double para2)//����仯�ʣ���Ե�����仯�ʣ��߶ȱ仯��
{
	double para3 = abs(para1 - para2);
	double max_para = max(para1, para2);
	double para_dy = para3 / max_para;
	return para_dy;
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

	for (int i1 = 4; i1 < (in.rows + 4); i1++)//���
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