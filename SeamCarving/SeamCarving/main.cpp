#include<opencv2/opencv.hpp>
#include<stdio.h>

#include "Saliency.h" 

using namespace cv;
using namespace std;

#define NUM 20

void getGradientMap(Mat& image)
{
	Mat image_gray(image.rows, image.cols, CV_8U, Scalar(0));
	cvtColor(image, image_gray, CV_BGR2GRAY); //��ɫͼ��ת��Ϊ�Ҷ�ͼ��

	Mat gradiant_H(image.rows, image.cols, CV_64F, Scalar(0));//ˮƽ�ݶȾ���
	Mat gradiant_V(image.rows, image.cols, CV_64F, Scalar(0));//��ֱ�ݶȾ���

	Mat kernel_H = (Mat_<double>(3, 3) << 1, 2, 1,
		0, 0, 0,
		-1, -2, -1); //��ˮƽ�ݶ���ʹ�õľ���ˣ�����ʼֵ��

	Mat kernel_V = (Mat_<double>(3, 3) << 1, 0, -1,
		2, 0, -2,
		1, 0, -1); //��ֱ�ݶ���ʹ�õľ���ˣ�����ʼֵ��

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
	for (int i = 0; i < tmp.rows - 1; i++)//���ϵ���,��ȡ��ֱ�����ϵ�seam
	{
		if(i==0)////////���һ���е���С����
	  { 
		for (int j = 0; j < tmp.cols; j++)////������һ��
		{
			if (min > E.at<double>(i, j))
			{
				min = E.at<double>(i, j);////�ҳ���һ���е���Сֵ
				col = j;////��¼������һ��
				index = col;
			}
			
			
		}
		tmp.at<Vec3b>(i, index)[0] = 0;////����С���ر�ɺ�ɫ
		tmp.at<Vec3b>(i, index)[1] = 0;
		tmp.at<Vec3b>(i, index)[2] = 255;
	  }
		else      
		{
			 index = col;
			min = E.at<double>(i, col);////1��tmp.rows-1��
			if (col == 0)///����ǵ�һ��
			{
				///ֻ��͵ڶ������رȽϣ��ҳ���Сֵ
				if (min > E.at<double>(i, col + 1))
				{
					index = col + 1;
				}
			}
			else if (col == tmp.cols)////��������һ��
			{
				///ֻ��͵��������رȽϣ��ҳ���Сֵ
				if (min > E.at<double>(i, col - 1))
				{
					index = col- 1;
				}
			}
			else////������м���
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
	/////����ԭʼͼ��
	Mat image = imread("16.jpg");
	namedWindow("Original");
	imshow("Original", image);
	//
	//////���������ͼ
	Mat tmpS;
	

	/////����ݶ�ͼ
	Mat tmpG;
	

	/////��ȡ�õ���������ͼ���ݶ�ͼ�е����ֵ��Ϊ����ͼ
	Mat tmpE;
	
	
	////////////////��̬�滮
	
	Mat minTrace;
	
	////��seam������ԭͼ��
	Mat seamImage;
	
	///��¼seam�ߵĹ켣
	Mat trace[NUM];

	Mat tmpImage;
	image.copyTo(tmpImage);

	for (int i = 0; i < NUM; i++)
	{
		//////���������ͼ
		tmpImage.copyTo(tmpS);
		getSaliencyMap(tmpS);
		/////����ݶ�ͼ
		tmpImage.copyTo(tmpG);
		getGradientMap(tmpG);

		/////��ȡ�õ���������ͼ���ݶ�ͼ�е����ֵ��Ϊ����ͼ
		tmpImage.copyTo(tmpE);
		getEnergyMap(tmpE, tmpS, tmpG);
		

		Mat sumEnergy(tmpImage.rows, tmpImage.cols, CV_64F, Scalar(0));
		Mat traceV(tmpImage.rows - 1, tmpImage.cols, CV_64F, Scalar(0));
		////////////////��̬�滮
		minTrace = dynamicProgramming(sumEnergy, tmpE, traceV);
		minTrace.copyTo(trace[i]);//����С�켣���浽��������ڱ���seam�ķֲ�״��

		////��seam������ԭͼ��
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