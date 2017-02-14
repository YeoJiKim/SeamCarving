#include<opencv2/opencv.hpp>
#include<stdio.h>

#include "Saliency.h" 

using namespace cv;
using namespace std;

#define NUM 20

void getGradientMap(Mat& image)
{
	Mat image_gray(image.rows, image.cols, CV_8U, Scalar(0));
	cvtColor(image, image_gray, CV_BGR2GRAY); //彩色图像转换为灰度图像

	Mat gradiant_H(image.rows, image.cols, CV_64F, Scalar(0));//水平梯度矩阵
	Mat gradiant_V(image.rows, image.cols, CV_64F, Scalar(0));//垂直梯度矩阵

	Mat kernel_H = (Mat_<double>(3, 3) << 1, 2, 1,
		0, 0, 0,
		-1, -2, -1); //求水平梯度所使用的卷积核（赋初始值）

	Mat kernel_V = (Mat_<double>(3, 3) << 1, 0, -1,
		2, 0, -2,
		1, 0, -1); //求垂直梯度所使用的卷积核（赋初始值）

	filter2D(image_gray, gradiant_H, gradiant_H.depth(), kernel_H);
	filter2D(image_gray, gradiant_V, gradiant_V.depth(), kernel_V);

	Mat gradMag_mat(image.rows, image.rows, CV_64F, Scalar(0));
	add(abs(gradiant_H), abs(gradiant_V), gradMag_mat);
	gradMag_mat.copyTo(image);
	
	
}

void getSaliencyMap(Mat& tmp)
{
	Saliency sal;
	vector<double> salmap(0);
	
	IplImage *img = &IplImage(tmp);
	if (!img) {
		cout << "failed to load image" << endl;
	}
	assert(img->nChannels == 3);

	vector<unsigned int >imgInput;
	vector<double> imgSal;
	//IplImage to vector  
	for (int h = 0; h<img->height; h++) {
		unsigned char*p = (unsigned char*)img->imageData + h*img->widthStep;
		for (int w = 0; w<img->width; w++) {
			unsigned int t = 0;
			t += *p++;
			t <<= 8;
			t += *p++;
			t <<= 8;
			t += *p++;
			imgInput.push_back(t);
		}
	}
	sal.GetSaliencyMap(imgInput, img->width, img->height, imgSal);
	//vector to IplImage  
	int index = 0;
	IplImage* imgout = cvCreateImage(cvGetSize(img), IPL_DEPTH_64F, 1);
	for (int h = 0; h<imgout->height; h++) {
		double*p = (double*)(imgout->imageData + h*imgout->widthStep);
		for (int w = 0; w<imgout->width; w++) {
			*p++ = imgSal[index++];
		}
	}

	tmp = cvarrToMat(imgout, true);
}
void getEnergyMap(Mat& tmp,Mat& Sal, Mat& Gra)
{
	Mat E(tmp.rows, tmp.cols, CV_64F, Scalar(0));
	
	for(int i=0;i<tmp.rows;i++)
		for (int j = 0; j < tmp.cols;j++)
		{
			
			E.at<double>(i, j) = fmax(Sal.at<double>(i, j), Gra.at<double>(i, j));
		}

	E.copyTo(tmp);
}

/*Mat dynamicProgramming(Mat& tmp, Mat& E, Mat& traceV)
{
	double min;
	min = E.at<double>(0, 0);
	int col=0;
	int index = col;
	for (int i = 0; i < tmp.rows - 1; i++)//从上到下,获取垂直方向上的seam
	{
		if(i==0)////////求第一行中的最小能量
	  { 
		for (int j = 0; j < tmp.cols; j++)////遍历第一行
		{
			if (min > E.at<double>(i, j))
			{
				min = E.at<double>(i, j);////找出第一行中的最小值
				col = j;////记录下是哪一列
				index = col;
			}
			
			
		}
		tmp.at<Vec3b>(i, index)[0] = 0;////将最小像素变成红色
		tmp.at<Vec3b>(i, index)[1] = 0;
		tmp.at<Vec3b>(i, index)[2] = 255;
	  }
		else      
		{
			 index = col;
			min = E.at<double>(i, col);////1到tmp.rows-1行
			if (col == 0)///如果是第一列
			{
				///只需和第二列像素比较，找出最小值
				if (min > E.at<double>(i, col + 1))
				{
					index = col + 1;
				}
			}
			else if (col == tmp.cols)////如果是最后一列
			{
				///只需和倒二列像素比较，找出最小值
				if (min > E.at<double>(i, col - 1))
				{
					index = col- 1;
				}
			}
			else////如果是中间列
			{/////
				if (min > E.at<double>(i, col + 1))
				{
					index = col + 1;
				}
				if (min > E.at<double>(i, col - 1))
				{
					index = col - 1;
				}				
			}
			min = E.at<double>(i, index);
			col = index;
			tmp.at<Vec3b>(i, index)[0] = 0;
			tmp.at<Vec3b>(i, index)[1] = 0;
			tmp.at<Vec3b>(i, index)[2] = 255;
		}

		traceV.at<double>(i, 0) = index;
	}

	Mat out(tmp.rows, tmp.cols, CV_64F, Scalar(0));
	tmp.copyTo(out);
	return out;
	
}*/
Mat dynamicProgramming(Mat& tmpSumE, Mat& E, Mat& traceV)
{
	E.copyTo(tmpSumE);
	double min;
	int index;
	for (int i = 1; i < E.rows; i++)
	{
		for (int j = 0; j < E.cols; j++)
		{
			min = tmpSumE.at<double>(i - 1, j);
			index = 0;
			if (j == 0)
			{
				if (min > tmpSumE.at<double>(i - 1, j + 1))
				{
					min = tmpSumE.at<double>(i - 1, j + 1);
					index = 1;
				}
			}
			else if (j == E.cols - 1)
			{
				if (min >tmpSumE.at<double>(i - 1, j - 1))
				{
					min= tmpSumE.at<double>(i - 1, j - 1);
					index = -1;
				}
			}
			else
			{
				if (min > tmpSumE.at<double>(i - 1, j + 1))
				{
					min = tmpSumE.at<double>(i - 1, j + 1);
					index = 1;
				}
				if (min > tmpSumE.at<double>(i - 1, j - 1))
				{
					min = tmpSumE.at<double>(i - 1, j - 1);
					index = -1;
				}
			}
			tmpSumE.at<double>(i, j) = tmpSumE.at<double>(i, j) + min;
			traceV.at<double>(i - 1, j) = index;
		}
	}

	double min2;
	int index2=0;
	min2 = tmpSumE.at<double>(tmpSumE.rows - 1, 0);
	for (int i = 0; i < tmpSumE.cols; i++)
	{
		if (min2 > tmpSumE.at<double>(tmpSumE.rows - 1, i))
		{
			min2 = tmpSumE.at<double>(tmpSumE.rows - 1, i);
			index2 = i;
		}
	}
	
	Mat minTrace(tmpSumE.rows, 1, CV_64F, Scalar(0));
	minTrace.at<double>(tmpSumE.rows - 1, 0) = index2;
	for (int i = tmpSumE.rows - 2; i >0; i--)
	{
		minTrace.at<double>(i, 0) = index2 + traceV.at<double>(i, index);
		index2 = minTrace.at<double>(i, 0);

	}

	return minTrace;

}

Mat drawSeam(Mat& tmp, Mat& minTrace)
{
	
	Mat seam(tmp.rows, tmp.cols, tmp.type());
	tmp.copyTo(seam);
	for (int i = 0; i < tmp.rows; i++)
	{
		int k = minTrace.at<double>(i, 0);
		for (int j = 0; j < tmp.cols; j++)
		{

			seam.at<Vec3b>(i, j)[0] = tmp.at<Vec3b>(i, j)[0];
			seam.at<Vec3b>(i, j)[1] = tmp.at<Vec3b>(i, j)[1];
			seam.at<Vec3b>(i, j)[2] = tmp.at<Vec3b>(i, j)[2];
			if (j == k)
			{
				seam.at<Vec3b>(i, k)[0] = 0;
				seam.at<Vec3b>(i, k)[1] = 0;
				seam.at<Vec3b>(i, k)[2] = 255;
			}
			
		}
		

	}
	return seam;
	
}

void deleteCol(Mat& tmp, Mat& resize, Mat& mintrace)
{
	
	for (int i = 0; i < resize.rows; i++)
	{
		int k = mintrace.at<double>(i, 0);
		for (int j = 0; j < tmp.cols-1; j++)
		{
			if (j <= k - 1)

			{
				resize.at<Vec3b>(i, j)[0] = tmp.at<Vec3b>(i, j)[0];
				resize.at<Vec3b>(i, j)[1] = tmp.at<Vec3b>(i, j)[1];
				resize.at<Vec3b>(i, j)[2] = tmp.at<Vec3b>(i, j)[2];
			}
			
			else
			{
				resize.at<Vec3b>(i, j)[0] = tmp.at<Vec3b>(i, j + 1)[0];
				resize.at<Vec3b>(i, j)[1] = tmp.at<Vec3b>(i, j + 1)[1];
				resize.at<Vec3b>(i, j)[2] = tmp.at<Vec3b>(i, j + 1)[2];
			}

		}
		
	} 

	
	
}

int main()
{
	/////输入原始图像
	Mat image = imread("16.jpg");
	namedWindow("Original");
	imshow("Original", image);
	//
	//////获得显著度图
	Mat tmpS;
	

	/////获得梯度图
	Mat tmpG;
	

	/////获取得到的显著度图和梯度图中的最大值作为能量图
	Mat tmpE;
	
	
	////////////////动态规划
	
	Mat minTrace;
	
	////将seam绘制在原图上
	Mat seamImage;
	
	///记录seam线的轨迹
	Mat trace[NUM];

	Mat tmpImage;
	image.copyTo(tmpImage);

	for (int i = 0; i < NUM; i++)
	{
		//////获得显著度图
		tmpImage.copyTo(tmpS);
		getSaliencyMap(tmpS);
		/////获得梯度图
		tmpImage.copyTo(tmpG);
		getGradientMap(tmpG);

		/////获取得到的显著度图和梯度图中的最大值作为能量图
		tmpImage.copyTo(tmpE);
		getEnergyMap(tmpE, tmpS, tmpG);
		

		Mat sumEnergy(tmpImage.rows, tmpImage.cols, CV_64F, Scalar(0));
		Mat traceV(tmpImage.rows - 1, tmpImage.cols, CV_64F, Scalar(0));
		////////////////动态规划
		minTrace = dynamicProgramming(sumEnergy, tmpE, traceV);
		minTrace.copyTo(trace[i]);//将最小轨迹保存到数组里，便于保存seam的分布状况

		////将seam绘制在原图上
		seamImage= drawSeam(tmpImage, trace[i]);
		namedWindow("Seam");
		imshow("Seam", seamImage);

		Mat resize(tmpImage.rows, tmpImage.cols - 1, tmpImage.type());
		deleteCol(tmpImage,resize, minTrace);
		namedWindow("Resize");
		imshow("Resize", resize);

		tmpImage = resize;
		waitKey(1);   

	}
	imwrite("Saliency.jpg", tmpS);
	imwrite("Gradient.jpg", tmpG);
	imwrite("Energy.jpg", tmpE);
	//imwrite("Seam.jpg", seamImage);
	imwrite("Resize.jpg", tmpImage);

	
	Mat SeamImage;
	image.copyTo(SeamImage);
	for (int i = 0; i < NUM; i++)
	{
		for (int j = 0; j < SeamImage.rows; j++)

		{
			int k = trace[i].at<double>(j, 0);
			for (int m=0;m<SeamImage.cols;m++)
			{
				SeamImage.at<Vec3b>(j, m)[0] = SeamImage.at<Vec3b>(j, m)[0];
				SeamImage.at<Vec3b>(j, m)[1] = SeamImage.at<Vec3b>(j, m)[1];
				SeamImage.at<Vec3b>(j, m)[2] = SeamImage.at<Vec3b>(j, m)[2];

				if (m == k)
				{
					SeamImage.at<Vec3b>(j, k)[0] = 0;
					SeamImage.at<Vec3b>(j, k)[1] = 0;
					SeamImage.at<Vec3b>(j, k)[2] = 255;
				}
			}
			
		}
	}
	imwrite("Seam.jpg", SeamImage);


	
}