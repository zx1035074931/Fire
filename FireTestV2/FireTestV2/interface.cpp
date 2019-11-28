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
	视频流源示例：
	广西卫视 ：rtmp://58.200.131.2:1935/livetv/gxtv
	湖南卫视 ：rtmp://58.200.131.2:1935/livetv/hunantv
	*/
	//http://devimages.apple.com.edgekey.net/streaming/examples/bipbop_4x3/gear2/prog_index.m3u8
	//string address("rtmp://58.200.131.2:1935/livetv/gxtv");//网络视频地址地址，http协议,不需要双反斜杠
	string address("F:\\大学工作\\实验项目\\1.RZ005-可见火点自动识别\\3数据与素材\\Fire Videos\\1.mp4");//本地视频地址 
	
	string weightfileadd("F:\\大学工作\\实验项目\\1.RZ005-可见火点自动识别\\1项目完整主体\\a.txt");//权重文件地址

	vector<vector<vector<double>>> weight(6, vector<vector<double>>(6, vector<double>(8)));
	weight = loaddata(weightfileadd);

	Fire fire1 = Fire(address, weight);
	waitKey(0);
}

vector<vector<vector<double>>> loaddata(String add)//加载权重文件
{
	vector<vector<vector<double>>> weight(6, vector<vector<double>>(6, vector<double>(8)));//建立5*5*8的double类型vector存储权重数据
	int i = 0;
	int j = 0;
	int k = 0;
	ifstream file;
	file.open(add, ios::in);//权重文件地址
	if (file.fail())
	{
		cout << "缺少权重文件." << endl;
		file.close();
		cin.get();
		cin.get();
	}
	else
	{
		while (!file.eof() && k<8) //读取数据到数组,file.eof()判断文件是否为空
		{
			for (i = 0; i < 5; i++)
			{
				for (j = 0; j < 5; j++)
					file >> weight[i][j][k];
			}
			k++;
		}
	}
	file.close(); //关闭文件*/
	return weight;
}

/*
V02 更新：输入视频得出九大特征。
v3-3 更新：只遍历一次图片，只进行图像检测HSV-->640*360 30fps 960*540 13fps（主要影响fps原因：hsv检测出来有多少个区域）
v4-0 更新：加入一定的注释；加入色偏与模糊度代码
*/