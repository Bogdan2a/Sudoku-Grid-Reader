// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include "fstream"
#include <stack>
#include <opencv2/core/utils/logger.hpp>
#include <random>



bool isInside(Mat img, int i, int j) {
	if ((i < img.rows) && (j < img.cols) && (i >= 0) && (j >= 0))
		return true;
	else
		return false;
}

Mat_<uchar> Lab7dilation(Mat_<uchar> img, Mat_<uchar> strel)
{
	Mat_<uchar> dst(img.rows, img.cols, 255);
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			if (img(i, j) == 0)
			{
				for (int u = 0; u < strel.rows; u++)
					for (int v = 0; v < strel.cols; v++)
					{
						if (strel(u, v) == 0)
						{
							int i2 = i + u - strel.rows / 2;
							int j2 = j + v - strel.cols / 2;
							if (isInside(dst, i2, j2))
								dst(i2, j2) = 0;
						}
					}
			}

	return dst;
}

Mat_<uchar> Lab7erosion(Mat_<uchar> img, Mat_<uchar> strel)
{
	Mat_<uchar> dst(img.rows, img.cols, 255);
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			if (img(i, j) == 0)
			{
				int allblack = 1;
				for (int u = 0; u < strel.rows; u++)
					for (int v = 0; v < strel.cols; v++)
					{
						if (strel(u, v) == 0)
						{
							int i2 = i + u - strel.rows / 2;
							int j2 = j + v - strel.cols / 2;
							if (isInside(img, i2, j2) && img(i2, j2) == 255)
								allblack = 0;
						}
					}
				if (allblack == 1)
					dst(i, j) = 0;
			}

	return dst;
}

Mat_<uchar> Lab7setDifference(Mat_<uchar> imgA, Mat_<uchar> imgB)
{
	Mat_<uchar> negA(imgA.rows, imgA.cols, 255);
	Mat_<uchar> dst(imgA.rows, imgA.cols, 255);
	for (int i = 0; i < imgA.rows; i++)
	{
		for (int j = 0; j < imgA.cols; j++)
		{
			uchar val = imgA(i, j);
			uchar neg = 255 - val;
			negA(i, j) = neg;
		}
	}
	cv::bitwise_and(negA, imgB, dst);

	return dst;
}

Mat_<float> convolution(Mat_<uchar> src, Mat_<float> H)
{
	Mat_<float> dst(src.rows, src.cols, 255);
	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
		{
			float sum = 0;
			for (int u = 0; u < H.rows; u++)
				for (int v = 0; v < H.cols; v++)
				{
					//i2=i+u-rows/2 (rows si cols de la nucleu)
					//j2=j+v-cols/2
					int i2 = i + u - H.rows / 2;
					int j2 = j + v - H.cols / 2;
					if (isInside(src, i2, j2))
						sum = sum + H(u, v) * src(i2, j2);
				}
			dst(i, j) = sum;
		}
	return dst;
}

Mat_<uchar> norm(Mat_<float> img, Mat_<float> H)
{
	Mat_<uchar> dst(img.rows, img.cols, 255);
	float a = 0;
	float b = 0;
	float c = 0;
	float d = 255;
	for (int i = 0; i < H.rows; i++)
		for (int j = 0; j < H.cols; j++)
			if (H(i, j) < 0)
				a = a + H(i, j);
			else
				b = b + H(i, j);
	a = a * 255;
	b = b * 255;
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
		{
			dst(i, j) = (img(i, j) - a) * (d - c) / (b - a) + c;
		}
	return dst;
}

Mat LabelingDFSProject(Mat src) {

	Mat labels = Mat(src.rows, src.cols, CV_32SC1, Scalar(0));
	Mat colored(src.rows, src.cols, CV_8UC3, Scalar(255, 255, 255));

	uchar label = 0;
	Vec3b color(rand() % 256, rand() % 256, rand() % 256);

	int di4[4] = { -1,0,1,0 };
	int dj4[4] = { 0,-1,0,1 };
	uchar neighbors4[4];
	int di8[8] = { -1,0,1,0,-1,-1,1,1 };
	int dj8[8] = { 0,-1,0,1,-1,1,-1,1 };
	uchar neighbors8[8];


	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			if (src.at<uchar>(i, j) == 0 && labels.at<int>(i, j) == 0)
			{
				label++;
				std::stack<Point> Q;
				labels.at<int>(i, j) = label;
				Q.push(Point(i, j));
				while (!Q.empty())
				{
					Point q = Q.top();
					Q.pop();
					for (int k = 0; k < 8; k++)
					{
						int ni = q.x + di8[k];
						int nj = q.y + dj8[k];
						if (isInside(src, ni, nj) && src.at<uchar>(ni, nj) == 0 && labels.at<int>(ni, nj) == 0)
						{
							labels.at<int>(ni, nj) = label;
							Q.push(Point(ni, nj));
						}
					}
				}
			}
		}
	}
	int a[300] = { 0 };
	for (int k = 1; k <= label; k++) {
		Vec3b color = Vec3b(rand() % 256, rand() % 256, rand() % 256);
		for (int i = 1; i < src.rows - 1; i++) {
			for (int j = 1; j < src.cols - 1; j++) {
				if (labels.at<int>(i, j) == k)
				{
					colored.at<Vec3b>(i, j) = color;
					a[k]++;
				}

			}
		}
	}
	imshow("colored labeling", colored);
	int max = 0;
	for (int i = 1; i <= label; i++)
	{
		if (a[i] > max)
			max = a[i];
	}
	auto dst = src.clone();
	dst.setTo(255);
	for (int i = 1; i <= label; i++)
	{
		if (a[i] == max)
		{
			for (int j = 1; j < src.rows - 1; j++)
			{
				for (int k = 1; k < src.cols - 1; k++)
				{
					if (labels.at<int>(j, k) == i)
					{
						dst.at<uchar>(j, k) = 0;
					}
				}
			}
		}
	}
	return dst;
}

inline double Det(double a, double b, double c, double d)
{
	return a * d - b * c;
}

bool LineLineIntersect(int x1, int y1, //Line 1 start
	int x2, int y2, //Line 1 end
	int x3, int y3, //Line 2 start
	int x4, int y4, //Line 2 end
	int& xOut, int& yOut) //Output 
{
	double detL1 = Det(x1, y1, x2, y2);
	double detL2 = Det(x3, y3, x4, y4);
	double x1mx2 = x1 - x2;
	double x3mx4 = x3 - x4;
	double y1my2 = y1 - y2;
	double y3my4 = y3 - y4;

	double xnom = Det(detL1, x1mx2, detL2, x3mx4);
	double ynom = Det(detL1, y1my2, detL2, y3my4);
	double denom = Det(x1mx2, y1my2, x3mx4, y3my4);

	(int)xOut = xnom / denom;
	(int)yOut = ynom / denom;

	return true; //All OK
}

int main()
{
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);

	Mat src = imread("Images/test.bmp", 0); //load image
	imshow("original image", src);

	Mat_<float> H5(5, 5, 1.f); //kernel for convolution
	auto dst1 = convolution(src, H5); //apply convolution
	auto dstn1 = norm(dst1, H5); //apply normalization
	imshow("filtered image", dstn1);

	//thresholding
	Mat_<uchar> dst2 = dstn1.clone();
	dst2.setTo(0);
	int t = 117;
	for (int i = 0; i < dstn1.rows; i++)
		for (int j = 0; j < dstn1.cols; j++)
			if (dstn1(i, j) > t)
				dst2(i, j) = 255;
			else
				dst2(i, j) = 0;

	imshow("thresholded image", dst2);

	//erosion
	Mat_<uchar> strel(3, 3);
	strel.setTo(0);
	strel(0, 1) = 1;
	strel(1, 0) = 1;
	strel(1, 1) = 1;
	strel(1, 2) = 1;
	strel(2, 1) = 1;
	auto dst3 = Lab7erosion(dst2, strel);
	auto dst4 = Lab7dilation(dst3, strel);
	auto dst5 = Lab7erosion(dst4, strel);
	auto dst6 = Lab7erosion(dst5, strel);
	auto dst7 = Lab7setDifference(dst4, dst6);
	auto dst8 = Lab7erosion(dst7, strel);
	auto dst9 = Lab7dilation(dst8, strel);
	imshow("eroded image", dst9);

	//invert image
	Mat_<uchar> dst10 = dst9.clone();
	dst10.setTo(0);
	for (int i = 0; i < dst9.rows; i++)
		for (int j = 0; j < dst9.cols; j++)
			dst10(i, j) = 255 - dst9(i, j);
	imshow("inverted image", dst10);


	//labeling
	auto outsideBorder = LabelingDFSProject(dst10);    //return an image with the largest connected component
	imshow("labeled image", outsideBorder);

	auto dst11 = outsideBorder.clone();
	Mat cdst = outsideBorder.clone();
	Canny(outsideBorder, dst11, 50, 200, 3);

	// Standard Hough Line Transform
	std::vector<Vec2f> lines; // will hold the results of the detection
	HoughLines(dst11, lines, 1, CV_PI / 180, 150, 0, 0); // runs the actual detection

	std::stack<Point> Q;

	// Draw the lines
	for (size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0], theta = lines[i][1];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a * rho, y0 = b * rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		line(cdst, pt1, pt2, 0, 1, LINE_AA);
	}
	imshow("Hough line transform", cdst);


	Mat_<uchar> binarizedcdst = cdst.clone();
	binarizedcdst.setTo(0);
	for (int i = 0; i < cdst.rows; i++)
		for (int j = 0; j < cdst.cols; j++)
			if (cdst.at<uchar>(i, j) > 240)
				binarizedcdst(i, j) = 255;
			else
				binarizedcdst(i, j) = 0;
	cdst = binarizedcdst;

	//scan the first line of the image and save the first and the last pixels
	Point firstPixelUp;
	Point lastPixelUp;
	for (int i = 0; i < cdst.cols; i++)
	{
		if (cdst.at<uchar>(0, i) == 0)
		{
			firstPixelUp = Point(i, 0);
			break;
		}
	}
	for (int i = cdst.cols - 1; i >= 0; i--)
	{
		if (cdst.at<uchar>(0, i) == 0)
		{
			lastPixelUp = Point(i, 0);
			break;
		}
	}
	//same for the down line
	Point firstPixelDown;
	Point lastPixelDown;
	for (int i = 0; i < cdst.cols; i++)
	{
		if (cdst.at<uchar>(cdst.rows - 1, i) == 0)
		{
			firstPixelDown = Point(i, cdst.rows - 1);
			break;
		}
	}
	for (int i = cdst.cols - 1; i >= 0; i--)
	{
		if (cdst.at<uchar>(cdst.rows - 1, i) == 0)
		{
			lastPixelDown = Point(i, cdst.rows - 1);
			break;
		}
	}
	//scan the first column of the image and save the first and the last pixels
	Point firstPixelLeft;
	Point lastPixelLeft;
	for (int i = 0; i < cdst.rows; i++)
	{
		if (cdst.at<uchar>(i, 0) == 0)
		{
			firstPixelLeft = Point(0, i);
			break;
		}
	}
	for (int i = cdst.rows - 1; i >= 0; i--)
	{
		if (cdst.at<uchar>(i, 0) == 0)
		{
			lastPixelLeft = Point(0, i);
			break;
		}
	}
	//same for the right column
	Point firstPixelRight;
	Point lastPixelRight;
	for (int i = 0; i < cdst.rows; i++)
	{
		if (cdst.at<uchar>(i, cdst.cols - 1) == 0)
		{
			firstPixelRight = Point(cdst.cols - 1, i);
			break;
		}
	}
	for (int i = cdst.rows - 1; i >= 0; i--)
	{
		if (cdst.at<uchar>(i, cdst.cols - 1) == 0)
		{
			lastPixelRight = Point(cdst.cols - 1, i);
			break;
		}
	}

	//print the points
	std::cout << "firstPixelUp: " << firstPixelUp << std::endl;
	std::cout << "lastPixelUp: " << lastPixelUp << std::endl;
	std::cout << "firstPixelDown: " << firstPixelDown << std::endl;
	std::cout << "lastPixelDown: " << lastPixelDown << std::endl;
	std::cout << "firstPixelLeft: " << firstPixelLeft << std::endl;
	std::cout << "lastPixelLeft: " << lastPixelLeft << std::endl;
	std::cout << "firstPixelRight: " << firstPixelRight << std::endl;
	std::cout << "lastPixelRight: " << lastPixelRight << std::endl;
	//draw lines between them
	auto testampunctele = cdst.clone();
	testampunctele.setTo(255);
	line(testampunctele, firstPixelUp, firstPixelDown, 0, 1);
	line(testampunctele, lastPixelUp, lastPixelDown, 0, 1);
	line(testampunctele, firstPixelLeft, firstPixelRight, 0, 1);
	line(testampunctele, lastPixelLeft, lastPixelRight, 0, 1);

	Point upLeftCorner, upRightCorner, downLeftCorner, downRightCorner;

	LineLineIntersect(firstPixelUp.x, firstPixelUp.y,
		firstPixelDown.x, firstPixelDown.y,
		firstPixelLeft.x, firstPixelLeft.y,
		firstPixelRight.x, firstPixelRight.y,
		upLeftCorner.x, upLeftCorner.y);

	LineLineIntersect(lastPixelUp.x, lastPixelUp.y,
		lastPixelDown.x, lastPixelDown.y,
		firstPixelLeft.x, firstPixelLeft.y,
		firstPixelRight.x, firstPixelRight.y,
		upRightCorner.x, upRightCorner.y);

	LineLineIntersect(firstPixelUp.x, firstPixelUp.y,
		firstPixelDown.x, firstPixelDown.y,
		lastPixelLeft.x, lastPixelLeft.y,
		lastPixelRight.x, lastPixelRight.y,
		downLeftCorner.x, downLeftCorner.y);

	LineLineIntersect(lastPixelUp.x, lastPixelUp.y,
		lastPixelDown.x, lastPixelDown.y,
		lastPixelLeft.x, lastPixelLeft.y,
		lastPixelRight.x, lastPixelRight.y,
		downRightCorner.x, downRightCorner.y);

	//draw circles in those 4 corners
	circle(testampunctele, upLeftCorner, 2, 0, 2);
	circle(testampunctele, downLeftCorner, 2, 0, 2);
	circle(testampunctele, upRightCorner, 2, 0, 2);
	circle(testampunctele, downRightCorner, 2, 0, 2);

	imshow("Corners", testampunctele);


	Point2f corners[4] = { upLeftCorner,upRightCorner,downLeftCorner,downRightCorner };
	Point2f dstVertices[4] = {
		Point2f(0, 0),
		Point2f(500, 0),
		Point2f(0, 500),
		Point2f(500, 500)
	};
	Mat_<uchar> alignedSquare;
	// Calculate the perspective transform matrix
	Mat perspectiveTransformMatrix = getPerspectiveTransform(corners, dstVertices);
	// Apply the perspective transform to the original image

	warpPerspective(src, alignedSquare, perspectiveTransformMatrix, Size(500, 500));
	imshow("alignedSquare", alignedSquare);

	auto aligneddst1 = convolution(alignedSquare, H5); //apply convolution
	auto aligneddstn1 = norm(aligneddst1, H5); //apply normalization

	//thresholding
	Mat_<uchar> aligneddst2 = aligneddstn1.clone();
	aligneddst2.setTo(0);
	t = 120;
	for (int i = 0; i < aligneddstn1.rows; i++)
		for (int j = 0; j < aligneddstn1.cols; j++)
			if (aligneddstn1(i, j) > t)
				aligneddst2(i, j) = 255;
			else
				aligneddst2(i, j) = 0;
	//erode the vertical lines in the image with a special kernel
	Mat_<uchar> strel1(5, 5);
	strel1.setTo(0);
	strel1(2, 0) = 1;
	strel1(2, 1) = 1;
	strel1(2, 3) = 1;
	strel1(2, 4) = 1;


	auto aligneddst3 = Lab7erosion(aligneddst2, strel1);  //because the black and white are inverted apply dilation to achieve erosion on white
	auto alligneddst4 = Lab7dilation(aligneddst3, strel);
	//imshow("thresholded aligned image", aligneddst2);
	//imshow("thresholded aligned image eroded", aligneddst3);
	//split the aligneddst3 into 81 equal squares
	std::vector<Mat_<uchar>> squares;
	int squareSize = 500 / 9;
	for (int i = 0; i < 9; i++)
		for (int j = 0; j < 9; j++)
		{
			Mat_<uchar> square = alligneddst4(Range(i * squareSize, (i + 1) * squareSize), Range(j * squareSize, (j + 1) * squareSize));
			squares.push_back(square);
		}
	// save the squares in a folder
	for (int i = 0; i < squares.size(); i++)
	{
		std::string path = "E:\\FACULTATE\\AN 3\\Sem2\\IP\\Project\\Squares\\square" + std::to_string(i) + ".bmp";
		imwrite(path, squares[i]);
	}

	waitKey();
	return 0;
}