//#include "stdafx.h"
#include <opencv.hpp>

using namespace cv;
using namespace std;


float normL2(float * Hist1, float * Hist2, int size)
{
	float sum = 0;
	for (int i = 0; i < size; i++) {
		sum += (Hist1[i] - Hist2[i])*(Hist1[i] - Hist2[i]);
	}
	sum = sqrt(sum);
	return sum;
}

//手动实现 HOG (Histogram-of-Oriented-Gradients) 
int calcHOG(cv::Mat src, float * hist, int nAngle, int cellSize)
{

	if (cellSize > src.cols || cellSize > src.rows) {
		return -1;
	}

	//参数设置
	int nX = src.cols / cellSize;
	int nY = src.rows / cellSize;

	int binAngle = 360 / nAngle;

	//计算梯度及角度
	Mat gx, gy;
	Mat mag, angle;
	Sobel(src, gx, CV_32F, 1, 0, 1);
	Sobel(src, gy, CV_32F, 0, 1, 1);
	// x方向梯度，y方向梯度，梯度，角度，决定输出弧度or角度
	// 默认是弧度radians，通过最后一个参数可以选择角度degrees.
	cartToPolar(gx, gy, mag, angle, true);

	cv::Rect roi;
	roi.x = 0;
	roi.y = 0;
	roi.width = cellSize;
	roi.height = cellSize;

	for (int i = 0; i < nY; i++) {
		for (int j = 0; j < nX; j++) {

			cv::Mat roiMat;
			cv::Mat roiMag;
			cv::Mat roiAgl;

			roi.x = j * cellSize;
			roi.y = i * cellSize;

			//赋值图像
			roiMat = src(roi);
			roiMag = mag(roi);
			roiAgl = angle(roi);

			//当前cell第一个元素在数组中的位置
			int head = (i*nX + j)*nAngle;

			for (int n = 0; n < roiMat.rows; n++) {
				for (int m = 0; m < roiMat.cols; m++) {
					//计算角度在哪个bin，通过int自动取整实现
					int pos = (int)(roiAgl.at<float>(n, m) / binAngle);
					hist[head + pos] += roiMag.at<float>(n, m);
				}
			}

		}
	}

	return 0;
}

//手动实现，通过 HOG (Histogram-of-Oriented-Gradients)比较图像相似度
float compareImages(cv::Mat plMat)
{

	//读入图像
	cv::Mat refMat = imread("E:\\PIC\\template.png", 0);

	if (refMat.empty() || plMat.empty()) {
		std::cout << "failed to read image!:" << std::endl;
		return -1;
	}

	//参数设置
	int nAngle = 8;
	int blockSize = 16;
	int nX = refMat.cols / blockSize;
	int nY = refMat.rows / blockSize;

	int bins = nX * nY*nAngle;


	float * ref_hist = new float[bins];
	memset(ref_hist, 0, sizeof(float)*bins);
	float * pl_hist = new float[bins];
	memset(pl_hist, 0, sizeof(float)*bins);


	int reCode = 0;
	//计算三张输入图片的HOG
	reCode = calcHOG(refMat, ref_hist, nAngle, blockSize);
	reCode = calcHOG(plMat, pl_hist, nAngle, blockSize);

	//计算直方图距离
	float dis = normL2(ref_hist, pl_hist, bins);

	delete[] ref_hist;
	delete[] pl_hist;
	destroyAllWindows();

	return dis;
}


int main()
{
	cv::Mat srcMat = imread("E:\\PIC\\m.png", 0);
	cv::Mat frameMat = imread("E:\\PIC\\template.png", 0);
	cv::Mat resultMat;

	int x1 = srcMat.rows - frameMat.rows/2;
	int y1 = srcMat.cols - frameMat.cols/2;
	int x = frameMat.rows/2, y = frameMat.cols/2;
	int a = 0, b = 0;
	float dis1=0,dis2=0;

	for ( ;x < x1; x++)
	{
		for (; y< y1; y++)
		{
			Mat roi(srcMat, Rect(x, y, frameMat.rows, frameMat.cols));//选取部分图像一直不成功
			dis1=compareImages(roi);
			if (dis1 < dis2)
			{
				resultMat = roi;
				a = x;
				b = y;
			}
			dis2 = dis1;
		}
	}
	cv::Rect rect;
	rect.x = a;
	rect.y = b;
	rect.width = frameMat.rows;
	rect.height = frameMat.cols;
	rectangle(srcMat, rect, CV_RGB(255, 0, 0), 1, 8, 3);

	imshow("src", srcMat);
	waitKey(0);
	return 0;
}

