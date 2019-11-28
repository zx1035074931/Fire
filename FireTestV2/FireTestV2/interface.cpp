#include <opencv.hpp>
#include <fstream>
#include"Fire.h"

using namespace cv;
using namespace std;

vector<vector<vector<double>>> loaddata(String add);
//string address;
int main()
{
	/*
	��Ƶ��Դʾ����
	�������� ��rtmp://58.200.131.2:1935/livetv/gxtv
	�������� ��rtmp://58.200.131.2:1935/livetv/hunantv
	*/
	//http://devimages.apple.com.edgekey.net/streaming/examples/bipbop_4x3/gear2/prog_index.m3u8
	//string address("rtmp://58.200.131.2:1935/livetv/gxtv");//������Ƶ��ַ��ַ��httpЭ��,����Ҫ˫��б��
	string address("F:\\��ѧ����\\ʵ����Ŀ\\1.RZ005-�ɼ�����Զ�ʶ��\\3�������ز�\\Fire Videos\\1.mp4");//������Ƶ��ַ 
	
	string weightfileadd("F:\\��ѧ����\\ʵ����Ŀ\\1.RZ005-�ɼ�����Զ�ʶ��\\1��Ŀ��������\\a.txt");//Ȩ���ļ���ַ

	vector<vector<vector<double>>> weight(6, vector<vector<double>>(6, vector<double>(8)));
	weight = loaddata(weightfileadd);

	Fire fire1 = Fire(address, weight);
	waitKey(0);
}

vector<vector<vector<double>>> loaddata(String add)//����Ȩ���ļ�
{
	vector<vector<vector<double>>> weight(6, vector<vector<double>>(6, vector<double>(8)));//����5*5*8��double����vector�洢Ȩ������
	int i = 0;
	int j = 0;
	int k = 0;
	ifstream file;
	file.open(add, ios::in);//Ȩ���ļ���ַ
	if (file.fail())
	{
		cout << "ȱ��Ȩ���ļ�." << endl;
		file.close();
		cin.get();
		cin.get();
	}
	else
	{
		while (!file.eof() && k<8) //��ȡ���ݵ�����,file.eof()�ж��ļ��Ƿ�Ϊ��
		{
			for (i = 0; i < 5; i++)
			{
				for (j = 0; j < 5; j++)
					file >> weight[i][j][k];
			}
			k++;
		}
	}
	file.close(); //�ر��ļ�*/
	return weight;
}

/*
V02 ���£�������Ƶ�ó��Ŵ�������
v3-3 ���£�ֻ����һ��ͼƬ��ֻ����ͼ����HSV-->640*360 30fps 960*540 13fps����ҪӰ��fpsԭ��hsv�������ж��ٸ�����
v4-0 ���£�����һ����ע�ͣ�����ɫƫ��ģ���ȴ���
*/