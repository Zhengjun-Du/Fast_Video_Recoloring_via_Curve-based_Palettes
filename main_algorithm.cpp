#include "main_algorithm.h" 
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
using namespace std;

Data::Data(){
	currentFrame = 0;
	isPaletteCalc = false;
	isVideoOpen = false;
}

void Data::close() {
	if (isVideoOpen) {
		for (size_t i = 0; i < videoNum; i++) {
			delete[] oriVideo_R[i];
			delete[] oriVideo_G[i];
			delete[] oriVideo_B[i];
			delete[] changedVideo_R[i];
			delete[] changedVideo_G[i];
			delete[] changedVideo_B[i];
		}
		delete[] oriVideo_R;
		delete[] oriVideo_G;
		delete[] oriVideo_B;
		delete[] changedVideo_R;
		delete[] changedVideo_G;
		delete[] changedVideo_B;

		isVideoOpen = false;
	}

	if (isPaletteCalc) {
		for (size_t i = 0; i < this->paletteNum; i++) {
			delete[] oriPalette_R[i];
			delete[] oriPalette_G[i];
			delete[] oriPalette_B[i];
			delete[] changedPalette_R[i];
			delete[] changedPalette_G[i];
			delete[] changedPalette_B[i];
		}
		delete[] oriPalette_R;
		delete[] oriPalette_G;
		delete[] oriPalette_B;
		delete[] changedPalette_R;
		delete[] changedPalette_G;
		delete[] changedPalette_B;

		selectedFrame.clear();
		selectedId.clear();
		for (size_t i = 0; i < paletteNum; i++) {
			selectedColor[i].clear();
		}
		selectedColor.clear();
		for (size_t i = 0; i < paletteNum * videoNum; i++) {
			isSelected[i] = false;
		}

		for (size_t i = 0; i < videoNum; i++) {
			delete[] weights[i];
		}
		delete[] weights;

		isPaletteCalc = false;
	}

}

void Data::OpenVideo(QString path) {
	this->close();

	cv::VideoCapture capture(path.toStdString());
	if (!capture.isOpened()) return;


	QFileInfo fileinfo(path);
	this->videoname = (fileinfo.baseName()).toStdString();

	videoNum = capture.get(cv::CAP_PROP_FRAME_COUNT);
	fps = capture.get(cv::CAP_PROP_FPS);
	video_rows = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
	video_cols = capture.get(cv::CAP_PROP_FRAME_WIDTH);
	videoSize = video_rows * video_cols;

	//因为内存原因设置最大能够读取的帧数
	cout << "ori video frames: " << videoNum << endl;
	int maxVideoNum = (921600.0 / videoSize) * 500;
	if (videoNum > maxVideoNum) videoNum = maxVideoNum;
	cout << "load video frames: " << videoNum << endl;

	//根据视频长度设置控制点数
	this->controlNum = videoNum / 30;
	controlNum = controlNum < 6 ? controlNum : 6;

	string videoFolder = "./" + videoname;
	string picFolder;
	if (_access(videoFolder.c_str(), 0) != 0) {
		if (_mkdir(videoFolder.c_str()) != 0) {
			cout << "folder set up failed : " << path.toStdString() << endl;
		}

		picFolder = videoFolder + "/pics";
		if (_mkdir(picFolder.c_str()) != 0) {
			cout << "folder set up failed : " << path.toStdString() << endl;
		}

		picFolder += "/";

		for (int i = 0; i < videoNum; i++) {
			cv::Mat frame;
			capture >> frame;
			if (frame.empty()) {
				videoNum = i;
				break;
			}
			string picPath = picFolder + to_string(i) + ".jpg";
			cv::imwrite(picPath, frame);
		}
	}
	else {
		picFolder = videoFolder + "/pics/";
	}

	capture.release();

	oriVideo_R = new sint * [videoNum];
	oriVideo_G = new sint * [videoNum];
	oriVideo_B = new sint * [videoNum];
	changedVideo_R = new sint * [videoNum];
	changedVideo_G = new sint * [videoNum];
	changedVideo_B = new sint * [videoNum];
	for (size_t i = 0; i < videoNum; i++) {
		oriVideo_R[i] = new sint[videoSize];
		oriVideo_G[i] = new sint[videoSize];
		oriVideo_B[i] = new sint[videoSize];
		changedVideo_R[i] = new sint[videoSize];
		changedVideo_G[i] = new sint[videoSize];
		changedVideo_B[i] = new sint[videoSize];
	}
	isVideoOpen = true;

#pragma omp parallel for
	for (int i = 0; i < videoNum; i++) {
		cv::Mat img = cv::imread(picFolder + to_string(i) + ".jpg");
		if (img.empty()) {
			this->videoNum = i;
			break;
		}
		for (size_t row = 0; row < video_rows; row++) {
			uchar* uc_pixel = img.data + row * img.step;
			int index = row * video_cols;
			for (size_t col = 0; col < video_cols; col++) {
				oriVideo_R[i][index] = (int)uc_pixel[2];
				oriVideo_G[i][index] = (int)uc_pixel[1];
				oriVideo_B[i][index] = (int)uc_pixel[0];

				uc_pixel += 3;
				index++;
			}
		}
		memcpy(changedVideo_R[i], oriVideo_R[i], sizeof(sint) * videoSize);
		memcpy(changedVideo_G[i], oriVideo_G[i], sizeof(sint) * videoSize);
		memcpy(changedVideo_B[i], oriVideo_B[i], sizeof(sint) * videoSize);
	}

	emit updated();
}

//提取视频调色板（包括每帧图像调色板、Bezier曲线拟合、权重计算）
void Data::calcPalette() {
	// 视频打开失败则终止
	if (!isVideoOpen) return;
	// 点击计算调色板按钮后释放之前分配的动态内存
	if (isPaletteCalc) {
		for (size_t i = 0; i < this->paletteNum; i++) {
			delete[] oriPalette_R[i];
			delete[] oriPalette_G[i];
			delete[] oriPalette_B[i];
			delete[] changedPalette_R[i];
			delete[] changedPalette_G[i];
			delete[] changedPalette_B[i];
		}
		delete[] oriPalette_R;
		delete[] oriPalette_G;
		delete[] oriPalette_B;
		delete[] changedPalette_R;
		delete[] changedPalette_G;
		delete[] changedPalette_B;

		for (size_t i = 0; i < videoNum; i++) {
			delete[] weights[i];
		}
		delete[] weights;

		isPaletteCalc = false;
	}
	isPaletteCalc = true;

	//ori image palettes extraction
	//---------------------------------------------------------------------------------------------------
	clock_t start, end;
	start = clock();

	// 建立二维数组分别存储视频帧的Lab分量
	vector<vector<double>> seedsL(videoNum);
	vector<vector<double>> seedsA(videoNum);
	vector<vector<double>> seedsB(videoNum);
	// 对视频帧的大小进行一定的缩放，K值存储需要分割的超像素数量
	int K = (videoSize / 921600.0) * 500;

	// 记录每个超像素包含的像素数量
	vector<vector<int>> clustersize(videoNum);

	// 超像素分割(输入原图像像素(RGB)和K以及图像大小提取出超像素(Lab) )
#pragma omp parallel for
	for (int i = 0; i < videoNum; i++) {
		generateSuperpixels(oriVideo_R[i], oriVideo_G[i], oriVideo_B[i], seedsL[i], seedsA[i], seedsB[i], K, video_cols, video_rows, clustersize[i]);
	}

	//贝塞尔曲线数量的获取(使用SLIC利用超像素)
	clock_t start1, end1;
	start1 = clock();

	// 记录提取关键帧的频数
	int sample_step = 5;
	// 每sample_step帧提取一次关键帧
	int sampleNum = videoNum / sample_step;

	int estimateNum = 0;

	// 记录每一个关键帧估计曲线的数量
	double paletteNums = 0;
	// 权重衰减的阈值
	int threshold = 4;

#pragma omp parallel for reduction(+:paletteNums)
	for (int i = 0; i < sampleNum; i++) {
		paletteNums += calcPaletteNum(seedsL[i * sample_step], seedsA[i * sample_step], seedsB[i * sample_step], clustersize[i * sample_step], threshold);
		//cout << kk << endl;
	}

	// 将所有关键帧的调色板颜色数量的平均值作为视频调色板的颜色数量
	paletteNums /= sampleNum;

	end1 = clock();

	// 输出视频调色板的颜色数量（四舍五入取整）和提取调色板所用的时间
	cout << "palette number : " << (int)(paletteNums + 0.5) << endl;
	cout << "palette number extraction time : " << end1 - start1 << " ms" << endl;

	estimateNum = (int)(paletteNums + 0.5);

	// 重新分配动态内存
	this->paletteNum = estimateNum;
	oriPalette_R = new double* [paletteNum];
	oriPalette_G = new double* [paletteNum];
	oriPalette_B = new double* [paletteNum];
	changedPalette_R = new double* [paletteNum];
	changedPalette_G = new double* [paletteNum];
	changedPalette_B = new double* [paletteNum];
	for (size_t i = 0; i < paletteNum; i++) {
		oriPalette_R[i] = new double[videoNum];
		oriPalette_G[i] = new double[videoNum];
		oriPalette_B[i] = new double[videoNum];
		changedPalette_R[i] = new double[videoNum];
		changedPalette_G[i] = new double[videoNum];
		changedPalette_B[i] = new double[videoNum];
	}


	//k-Means聚类(以超像素进行改进的K-Means聚类提取出图像调色板)
	extractFirstPalette(0, seedsL[0], seedsA[0], seedsB[0]);
	for (size_t i = 1; i < videoNum; i++) {
		extractPalette(i, seedsL[i], seedsA[i], seedsB[i], 1);
	}
	end = clock();
	cout << "ori palette extraction time : " << end - start << " ms" << endl;

	// 清理和释放内存
	for (size_t i = 0; i < videoNum; i++) {
		seedsL[i].clear();
		seedsA[i].clear();
		seedsB[i].clear();
		seedsL[i].shrink_to_fit();
		seedsA[i].shrink_to_fit();
		seedsB[i].shrink_to_fit();
	}
	seedsL.clear(); seedsL.shrink_to_fit();
	seedsA.clear(); seedsA.shrink_to_fit();
	seedsB.clear(); seedsB.shrink_to_fit();

	//exportImagePalette();

//Bezeri fitting
//---------------------------------------------------------------------------------------------------
	this->controlNum = controlNum;

	start = clock();
	this->A = vector<double>(videoNum * controlNum);
	for (int i = 0; i < videoNum; i++) {
		for (int j = 0; j < controlNum; j++) {
			A[i * controlNum + j] = BernsteinNum(j, controlNum - 1, (double)i / (videoNum - 1));
		}
	}

	bezeirControl_L = vector<vector<double>>(paletteNum, vector<double>(controlNum));
	bezeirControl_A = vector<vector<double>>(paletteNum, vector<double>(controlNum));
	bezeirControl_B = vector<vector<double>>(paletteNum, vector<double>(controlNum));

	double* pointR;
	double* pointG;
	double* pointB;
	data3 data = { videoNum, controlNum, pointR, pointG, pointB, this->A };

	double minf;
	nlopt::opt opt(nlopt::LD_LBFGS, 3 * controlNum);
	opt.set_lower_bounds(-500);
	opt.set_upper_bounds(500);
	opt.set_xtol_rel(1e-8);
	
	for (size_t i = 0; i < paletteNum; i++) {
		data.pointR = this->oriPalette_R[i];
		data.pointG = this->oriPalette_G[i];
		data.pointB = this->oriPalette_B[i];
		vector<double> x = vector<double>(3 * controlNum);
		opt.set_min_objective(bezeirCost, &data);
		nlopt::result result = opt.optimize(x, minf);
		for (int j = 0; j < controlNum; j++)
		{
			bezeirControl_L[i][j] = x[j];
			bezeirControl_A[i][j] = x[controlNum + j];
			bezeirControl_B[i][j] = x[2 * controlNum + j];
		}
	}
	end = clock();
	cout << "Bezeir fitting time: " << end - start << " ms" << endl;

	//calc weights
	//---------------------------------------------------------------------------------------------------
	start = clock();
	weights = new float* [videoNum];
#pragma omp parallel for
	for (int i = 0; i < videoNum; i++) {
		weights[i] = new float[paletteNum * videoSize];
		calcWeights(i);
	}
	end = clock();
	cout << "Weights calc time: " << end - start << " ms" << endl;

#pragma omp parallel for
	for (int i = 0; i < paletteNum; i++) {
		for (size_t j = 0; j < videoNum; j++) {
			cv::Vec3d color;
			for (size_t k = 0; k < controlNum; k++) {
				color[0] += A[j * controlNum + k] * bezeirControl_L[i][k];
				color[1] += A[j * controlNum + k] * bezeirControl_A[i][k];
				color[2] += A[j * controlNum + k] * bezeirControl_B[i][k];
			}
			LAB2RGB(color);
			oriPalette_R[i][j] = color[0];
			oriPalette_G[i][j] = color[1];
			oriPalette_B[i][j] = color[2];
		}
	}

	for (size_t i = 0; i < paletteNum; i++) {
		memmove(changedPalette_R[i], oriPalette_R[i], sizeof(double) * videoNum);
		memmove(changedPalette_G[i], oriPalette_G[i], sizeof(double) * videoNum);
		memmove(changedPalette_B[i], oriPalette_B[i], sizeof(double) * videoNum);
	}

	isSelected = vector<bool>(paletteNum * videoNum, false);
	selectedColor = vector<vector<int>>(paletteNum);

	coefficient1 = vector<double>((controlNum - 1) * videoNum);
#pragma omp parallel for
	for (int i = 0; i < videoNum; i++) {
		for (size_t j = 0; j < controlNum - 1; j++) {
			coefficient1[i * (controlNum - 1) + j] = (controlNum - 1) * BernsteinNum(j, controlNum - 2, (double)i / (videoNum - 1));
		}
	}

	coefficient2 = vector<double>(controlNum * videoNum);

	for (int k = 0; k < controlNum; k++) {
		for (size_t i = 0; i < videoNum; i++) {
			coefficient2[k * videoNum + i] = 2 * (controlNum - 1) * (BernsteinNum(k - 1, controlNum - 2, (double)i / (videoNum - 1)) - BernsteinNum(k, controlNum - 2, (double)i / (videoNum - 1)));
		}
	}

	oriDiri_L = vector<vector<double>>(paletteNum, vector<double>(videoNum));
	oriDiri_A = vector<vector<double>>(paletteNum, vector<double>(videoNum));
	oriDiri_B = vector<vector<double>>(paletteNum, vector<double>(videoNum));

#pragma omp parallel for
	for (int k = 0; k < paletteNum; k++) {
		for (size_t i = 0; i < videoNum; i++) {
			for (size_t j = 0; j < controlNum - 1; j++) {
				oriDiri_L[k][i] += coefficient1[i * (controlNum - 1) + j] * (bezeirControl_L[k][j + 1] - bezeirControl_L[k][j]);
				oriDiri_A[k][i] += coefficient1[i * (controlNum - 1) + j] * (bezeirControl_A[k][j + 1] - bezeirControl_A[k][j]);
				oriDiri_B[k][i] += coefficient1[i * (controlNum - 1) + j] * (bezeirControl_B[k][j + 1] - bezeirControl_B[k][j]);
			}
		}
	}

}

void Data::extractFirstPalette(int index, const vector<double>& seedsL, const vector<double>& seedsA, const vector<double>& seedsB) {

	int seedsNum = seedsL.size();
	vector<double> weights = vector<double>(seedsNum, 1.0);
	vector<cv::Vec3d> grid_mean;
	grid_mean.resize(seedsNum);
	// 将第0帧画面的LAB值赋值给grid_mean数组
	for (size_t i = 0; i < seedsNum; i++) {
		grid_mean[i][0] = seedsL[i];
		grid_mean[i][1] = seedsA[i];
		grid_mean[i][2] = seedsB[i];
	}

	vector<cv::Vec3d> centroids = vector<cv::Vec3d>(paletteNum);// 存储调色板颜色对应的LAB值

	double max = 0;
	int max_index = 0;
	cv::Vec3d white{ 100, 0, 0 };
	double min_d = 999;
	/*for (size_t i = 0; i < weights.size(); i++) {
		double d = cv::norm(grid_mean[i], white);
		if (d < min_d) {
			max = i;
			min_d = d;
		}
	}
	centroids[0] = grid_mean[max_index];*/
	centroids[0] = grid_mean[0];// 选取第一个LAB颜色作为第一个调色板颜色
	//根据每次选取与当前颜色差距最大的颜色作为新的调色板颜色
	for (size_t i = 1; i < centroids.size(); i++) {
		cv::Vec3d last_center = centroids[i - 1];

		//decrease weights
		//select current max
		max = 0;
		max_index = 0;
		for (size_t j = 0; j < weights.size(); j++) {
			weights[j] *= (1 - exp(
				-pow(norm(grid_mean[j], last_center) / 80.0, 2)
			));

			if (weights[j] > max) {
				max = weights[j];
				max_index = j;
			}
		}
		centroids[i] = grid_mean[max_index];
	}
	// 通过遍历所有颜色，将颜色的LAB值累加到最接近的调色板颜色对应的temp_centroids上，然后取平均值来更新调色板的颜色
	// 但迭代过程中调色板颜色差距变化较小时结束迭代
	int label;
	vector<cv::Vec3d> temp_centroids = vector<cv::Vec3d>(paletteNum);
	vector<double> centroids_num = vector<double>(paletteNum);
	for (size_t it = 0; it < 500; it++) {
		fill(temp_centroids.begin(), temp_centroids.end(), 0);
		fill(centroids_num.begin(), centroids_num.end(), 0);

		//label cloest centroid
		for (size_t i = 0; i < grid_mean.size(); i++) {
			label = 0;
			min_d = 9999;
			for (size_t j = 0; j < centroids.size(); j++) {
				double dis = norm(grid_mean[i], centroids[j]);
				if (dis < min_d) {
					label = j;
					min_d = dis;
				}
			}
			temp_centroids[label] += grid_mean[i];
			centroids_num[label]++;
		}

		//new centorid
		for (size_t i = 0; i < temp_centroids.size(); i++) {
			if (centroids_num[i] == 0) {
				cout << "empty centroid :" << i << endl;
			}

			temp_centroids[i] = temp_centroids[i] / centroids_num[i];
		}

		if (norm(temp_centroids, centroids) < 1e-8) {
			centroids.assign(temp_centroids.begin(), temp_centroids.end());
			break;
		}

		centroids.assign(temp_centroids.begin(), temp_centroids.end());

	}

	exportPoint(0, grid_mean, centroids);

	for (size_t i = 0; i < paletteNum; i++) {
		oriPalette_R[i][index] = centroids[i][0];
		oriPalette_G[i][index] = centroids[i][1];
		oriPalette_B[i][index] = centroids[i][2];
	}

}

void Data::extractPalette(int index, const vector<double>& seedsL, const vector<double>& seedsA, const vector<double>& seedsB, int max_it) {

	int seedsNum = seedsL.size();
	vector<cv::Vec3d> grid_mean;
	grid_mean.resize(seedsNum);
	// 将第index帧画面的LAB值赋值给grid_mean数组
	for (size_t i = 0; i < seedsNum; i++) {
		grid_mean[i][0] = seedsL[i];
		grid_mean[i][1] = seedsA[i];
		grid_mean[i][2] = seedsB[i];
	}

	vector<cv::Vec3d> centroids = vector<cv::Vec3d>(paletteNum);// 存储调色板颜色对应的LAB值
	// 将上一帧的调色板颜色作为这帧的初始调色板
	for (size_t i = 0; i < paletteNum; i++) {
		centroids[i][0] = oriPalette_R[i][index - 1];
		centroids[i][1] = oriPalette_G[i][index - 1];
		centroids[i][2] = oriPalette_B[i][index - 1];
	}

	int label;
	double min_d;
	vector<cv::Vec3d> temp_centroids = vector<cv::Vec3d>(paletteNum);
	vector<double> centroids_num = vector<double>(paletteNum);
	// 通过遍历所有颜色，将颜色的LAB值累加到最接近的调色板颜色对应的temp_centroids上，然后取平均值来更新调色板的颜色
	for (size_t it = 0; it < max_it; it++) {
		fill(temp_centroids.begin(), temp_centroids.end(), 0);
		fill(centroids_num.begin(), centroids_num.end(), 0);

		//label cloest centroid
		for (size_t i = 0; i < grid_mean.size(); i++) {
			label = 0;
			min_d = 9999;
			for (size_t j = 0; j < centroids.size(); j++) {
				double dis = squareDistance(grid_mean[i], centroids[j]);
				if (dis < min_d) {
					label = j;
					min_d = dis;
				}
			}
			temp_centroids[label] += grid_mean[i];
			centroids_num[label]++;
		}

		//new centorid
		for (size_t i = 0; i < temp_centroids.size(); i++) {
			if (centroids_num[i] == 0) {
				cout << "empty centroid: frame " << index << " /palette " << i << endl;
				temp_centroids[i] = centroids[i];
				centroids_num[i] = 1;
			}

			temp_centroids[i] = temp_centroids[i] / centroids_num[i];
		}

		if (norm(temp_centroids, centroids) < 1e-8) {
			centroids.assign(temp_centroids.begin(), temp_centroids.end());
			break;
		}

		centroids.assign(temp_centroids.begin(), temp_centroids.end());
	}

	exportPoint(index, grid_mean, centroids);

	for (size_t i = 0; i < paletteNum; i++) {
		oriPalette_R[i][index] = temp_centroids[i][0];
		oriPalette_G[i][index] = temp_centroids[i][1];
		oriPalette_B[i][index] = temp_centroids[i][2];
	}

}

double Data::calc_param(int index, vector<double>& lamda) {
	lamda = vector<double>(paletteNum * paletteNum);

	// 累加所有调色板颜色间的LAB距离
	double sigma = 0;
	for (size_t i = 0; i < paletteNum; i++) {
		for (size_t j = i + 1; j < paletteNum; j++) {
			sigma += lab_distance(oriPalette_R[i][index], oriPalette_G[i][index], oriPalette_B[i][index],
				oriPalette_R[j][index], oriPalette_G[j][index], oriPalette_B[j][index]);
		}
	}

	sigma /= (paletteNum - 1) * paletteNum / 2;// 求平均距离
	double param = 5.0 / (sigma * sigma);

	Eigen::MatrixXd D(paletteNum, paletteNum);// 创建paletteNum行paletteNum列的动态矩阵

	for (size_t i = 0; i < paletteNum; i++) {
		for (size_t j = 0; j < paletteNum; j++) {
			D(i, j) = phiFunction(oriPalette_R[i][index], oriPalette_G[i][index], oriPalette_B[i][index],
				oriPalette_R[j][index], oriPalette_G[j][index], oriPalette_B[j][index], param);
		}
	}

	D = D.inverse(); // 矩阵逆运算
	for (size_t i = 0; i < paletteNum; i++) {
		for (size_t j = 0; j < paletteNum; j++) {
			lamda[i * paletteNum + j] = D(i, j);
		}
	}

	return param;
}

void Data::calc_singlePoint_weights(int findex, const cv::Vec3d& point, double param, const vector<double>& lamda, vector<double>& singleWeight) {
	singleWeight = vector<double>(this->paletteNum);

	// 求加权后每个调色板颜色与point颜色的差距
	for (size_t i = 0; i < paletteNum; i++) {
		for (size_t j = 0; j < paletteNum; j++) {
			singleWeight[i] += lamda[i * paletteNum + j] * phiFunction(oriPalette_R[j][findex], oriPalette_G[j][findex], oriPalette_B[j][findex],
				point[0], point[1], point[2], param);
		}
	}

	double sum = 0;
	for (size_t i = 0; i < paletteNum; i++) {
		if (singleWeight[i] > 0) {
			sum += singleWeight[i];
		}
		else {
			singleWeight[i] = 0;
		}
	}

	for (size_t i = 0; i < paletteNum; i++) {
		singleWeight[i] /= sum;
	}

}

void Data::calcWeights(int findex) {
	int gridn = 16;

	double step = 255.0 / gridn;

	vector<double> lamda;
	double param = calc_param(findex, lamda);
	//calc grid verts weights
	vector<vector<double>> grid_weights = vector<vector<double>>(paletteNum, vector<double>(pow(gridn + 1, 3)));

	vector<double> single_point_weights;
	cv::Vec3d gridv;
	int index = 0;
	// 求调色板颜色对应像素的权重
	for (size_t r = 0; r < gridn + 1; r++) {
		for (size_t g = 0; g < gridn + 1; g++) {
			for (size_t b = 0; b < gridn + 1; b++) {
				gridv = { r * step, g * step, b * step };
				RGB2LAB(gridv);
				calc_singlePoint_weights(findex, gridv, param, lamda, single_point_weights);
				index = r * (gridn + 1) * (gridn + 1) + g * (gridn + 1) + b;
				for (size_t i = 0; i < paletteNum; i++) {
					grid_weights[i][index] = single_point_weights[i];
				}
			}
		}
	}

	double r_frac, g_frac, b_frac;
	int r, g, b;

	for (size_t i = 0; i < videoSize; i++) {
		r = (int)(oriVideo_R[findex][i] / step);
		g = (int)(oriVideo_G[findex][i] / step);
		b = (int)(oriVideo_B[findex][i] / step);
		r_frac = oriVideo_R[findex][i] / step - r;
		g_frac = oriVideo_G[findex][i] / step - g;
		b_frac = oriVideo_B[findex][i] / step - b;
		if (abs(r - gridn) < 1e-4) {
			r = 15;
			r_frac = 1.0;
		}
		if (abs(g - gridn) < 1e-4) {
			g = 15;
			g_frac = 1.0;
		}
		if (abs(b - gridn) < 1e-4) {
			b = 15;
			b_frac = 1.0;
		}
		// 将对应块的权重赋值给findex帧第j个颜色第i个像素
		for (size_t j = 0; j < paletteNum; j++) {
			weights[findex][j * videoSize + i] = grid_weights[j][r * (gridn + 1) * (gridn + 1) + g * (gridn + 1) + b] * (1 - r_frac) * (1 - g_frac) * (1 - b_frac)
				+ grid_weights[j][r * (gridn + 1) * (gridn + 1) + g * (gridn + 1) + b + 1] * (1 - r_frac) * (1 - g_frac) * b_frac
				+ grid_weights[j][r * (gridn + 1) * (gridn + 1) + (g + 1) * (gridn + 1) + b] * (1 - r_frac) * g_frac * (1 - b_frac)
				+ grid_weights[j][r * (gridn + 1) * (gridn + 1) + (g + 1) * (gridn + 1) + b + 1] * (1 - r_frac) * g_frac * b_frac
				+ grid_weights[j][(r + 1) * (gridn + 1) * (gridn + 1) + g * (gridn + 1) + b] * r_frac * (1 - g_frac) * (1 - b_frac)
				+ grid_weights[j][(r + 1) * (gridn + 1) * (gridn + 1) + g * (gridn + 1) + b + 1] * r_frac * (1 - g_frac) * b_frac
				+ grid_weights[j][(r + 1) * (gridn + 1) * (gridn + 1) + (g + 1) * (gridn + 1) + b] * r_frac * g_frac * (1 - b_frac)
				+ grid_weights[j][(r + 1) * (gridn + 1) * (gridn + 1) + (g + 1) * (gridn + 1) + b + 1] * r_frac * g_frac * b_frac;
		}

	}
}

void Data::videoRecolor() {
	if (!isPaletteCalc) return;

	time_t start, end;
	start = clock();

#pragma omp parallel for
	for (int fi = 0; fi < videoNum; fi++) {
		memcpy(changedVideo_R[fi], oriVideo_R[fi], sizeof(sint) * videoSize);
		memcpy(changedVideo_G[fi], oriVideo_G[fi], sizeof(sint) * videoSize);
		memcpy(changedVideo_B[fi], oriVideo_B[fi], sizeof(sint) * videoSize);
		for (int i = 0; i < videoSize; i++) {
			for (size_t j = 0; j < paletteNum; j++) {
				changedVideo_R[fi][i] += weights[fi][j * videoSize + i] * (changedPalette_R[j][fi] - oriPalette_R[j][fi]);
				changedVideo_G[fi][i] += weights[fi][j * videoSize + i] * (changedPalette_G[j][fi] - oriPalette_G[j][fi]);
				changedVideo_B[fi][i] += weights[fi][j * videoSize + i] * (changedPalette_B[j][fi] - oriPalette_B[j][fi]);
			}
		}

		for (size_t i = 0; i < videoSize; i++) {
			if (changedVideo_R[fi][i] < 0) changedVideo_R[fi][i] = 0;
			else if (changedVideo_R[fi][i] > 255) changedVideo_R[fi][i] = 255;

			if (changedVideo_G[fi][i] < 0) changedVideo_G[fi][i] = 0;
			else if (changedVideo_G[fi][i] > 255) changedVideo_G[fi][i] = 255;

			if (changedVideo_B[fi][i] < 0) changedVideo_B[fi][i] = 0;
			else if (changedVideo_B[fi][i] > 255) changedVideo_B[fi][i] = 255;
		}
	}

	end = clock();
	cout << end - start << " ms" << endl;
}

void Data::imageRecolor(int findex) {
	if (!isPaletteCalc) return;

#pragma omp parallel for
	for (int i = 0; i < videoSize; i++) {
		changedVideo_R[findex][i] = oriVideo_R[findex][i];
		changedVideo_G[findex][i] = oriVideo_G[findex][i];
		changedVideo_B[findex][i] = oriVideo_B[findex][i];
		for (size_t j = 0; j < paletteNum; j++) {
			changedVideo_R[findex][i] += weights[findex][j * videoSize + i] * (changedPalette_R[j][findex] - oriPalette_R[j][findex]);
			changedVideo_G[findex][i] += weights[findex][j * videoSize + i] * (changedPalette_G[j][findex] - oriPalette_G[j][findex]);
			changedVideo_B[findex][i] += weights[findex][j * videoSize + i] * (changedPalette_B[j][findex] - oriPalette_B[j][findex]);
		}
	}

	for (size_t i = 0; i < videoSize; i++) {
		if (changedVideo_R[findex][i] < 0) changedVideo_R[findex][i] = 0;
		else if (changedVideo_R[findex][i] > 255) changedVideo_R[findex][i] = 255;

		if (changedVideo_G[findex][i] < 0) changedVideo_G[findex][i] = 0;
		else if (changedVideo_G[findex][i] > 255) changedVideo_G[findex][i] = 255;

		if (changedVideo_B[findex][i] < 0) changedVideo_B[findex][i] = 0;
		else if (changedVideo_B[findex][i] > 255) changedVideo_B[findex][i] = 255;
	}

	emit updated();
}

void Data::changePosition(int frameNum) {
	this->currentFrame = frameNum;

	imageRecolor(frameNum);

	emit updated();
}

void Data::curveDeformation() {
	if (!isPaletteCalc) return;

	double minf;
	nlopt::opt opt(nlopt::LD_LBFGS, controlNum);

	opt.set_lower_bounds(-500);
	opt.set_upper_bounds(500);
	opt.set_xtol_rel(1e-8);

	time_t start, end;
	start = clock_t();

	for (int pindex = 0; pindex < paletteNum; pindex++) {
		cout << "第"<< pindex+1<<"个调色板颜色：" << endl;
		int mnum = selectedColor[pindex].size();
		if (mnum < 1) continue;

		vector<double> changedL(mnum);
		vector<double> changedA(mnum);
		vector<double> changedB(mnum);

		for (size_t i = 0; i < mnum; i++) {
			int findex = selectedColor[pindex][i];
			cv::Vec3d color = { changedPalette_R[pindex][findex],changedPalette_G[pindex][findex],changedPalette_B[pindex][findex] };
			RGB2LAB(color);
			changedL[i] = color[0];
			changedA[i] = color[1];
			changedB[i] = color[2];
		}
		double lamda = 0.001;
		data2 dt = { controlNum, videoNum, A, oriDiri_L[pindex], selectedColor[pindex], changedL, coefficient1, coefficient2, lamda};
		opt.set_min_objective(moveCost, &dt);
		cout << "minf:" << minf << endl;

		vector<double>x(bezeirControl_L[pindex]);
		nlopt::result result = opt.optimize(x, minf);

		for (size_t i = 0; i < videoNum; i++) {
			changedPalette_R[pindex][i] = 0;
			changedPalette_G[pindex][i] = 0;
			changedPalette_B[pindex][i] = 0;
		}

		for (size_t i = 0; i < videoNum; i++) {
			for (size_t j = 0; j < controlNum; j++) {
				changedPalette_R[pindex][i] += A[i * controlNum + j] * x[j];
			}
		}

		dt.oriDiri = oriDiri_A[pindex];
		dt.changedPoint = changedA;
		opt.set_min_objective(moveCost, &dt);

		x = bezeirControl_A[pindex];
		result = opt.optimize(x, minf);
		cout << "minf:" << minf << endl;

		for (size_t i = 0; i < videoNum; i++) {
			for (size_t j = 0; j < controlNum; j++) {
				changedPalette_G[pindex][i] += A[i * controlNum + j] * x[j];
			}
		}

		dt.oriDiri = oriDiri_B[pindex];
		dt.changedPoint = changedB;
		opt.set_min_objective(moveCost, &dt);

		x = bezeirControl_B[pindex];
		result = opt.optimize(x, minf);
		cout << "minf:" << minf << endl;

		for (size_t i = 0; i < videoNum; i++) {
			for (size_t j = 0; j < controlNum; j++) {
				changedPalette_B[pindex][i] += A[i * controlNum + j] * x[j];
			}
		}

		for (size_t i = 0; i < videoNum; i++) {
			cv::Vec3d color = { changedPalette_R[pindex][i], changedPalette_G[pindex][i] , changedPalette_B[pindex][i] };
			LAB2RGB(color);
			changedPalette_R[pindex][i] = color[0];
			changedPalette_G[pindex][i] = color[1];
			changedPalette_B[pindex][i] = color[2];
		}
	}

	end = clock_t();
	cout << "palette reshape time: " << end - start << " ms" << endl;

	emit updated();
}

void Data::exportVideo(QString filename) {

	exportOriPalette(filename);
	exportChangedPalette(filename);

	videoRecolor();

	string Path = filename.toStdString() + "/" + videoname + "_recolor.mp4";
	cv::VideoWriter videoWriter(Path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(video_cols, video_rows), true);
	for (size_t i = 0; i < videoNum; i++) {
		cv::Mat frame(video_rows, video_cols, CV_8UC3);
		for (size_t row = 0; row < video_rows; row++) {
			uchar* uc_pixel = frame.data + row * frame.step;
			int index = row * video_cols;
			for (size_t col = 0; col < video_cols; col++) {
				uc_pixel[0] = changedVideo_B[i][index];
				uc_pixel[1] = changedVideo_G[i][index];
				uc_pixel[2] = changedVideo_R[i][index];

				uc_pixel += 3;
				index++;
			}
		}
		videoWriter << frame;
	}
	videoWriter.release();

	Path = filename.toStdString() + "/" + videoname + "_ori.mp4";
	videoWriter = cv::VideoWriter(Path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(video_cols, video_rows), true);
	for (size_t i = 0; i < videoNum; i++) {
		cv::Mat frame(video_rows, video_cols, CV_8UC3);
		for (size_t row = 0; row < video_rows; row++) {
			uchar* uc_pixel = frame.data + row * frame.step;
			int index = row * video_cols;
			for (size_t col = 0; col < video_cols; col++) {
				uc_pixel[0] = (uchar)oriVideo_B[i][index];
				uc_pixel[1] = (uchar)oriVideo_G[i][index];
				uc_pixel[2] = (uchar)oriVideo_R[i][index];

				uc_pixel += 3;
				index++;
			}
		}
		videoWriter << frame;
	}
	videoWriter.release();

}

void Data::exportOriPalette(QString filename) {
	string Path = filename.toStdString() + "/" + videoname + "_oriPalette.mp4";

#pragma omp parallel for
	for (int i = 0; i < videoNum; i++) {
		for (size_t j = 0; j < paletteNum; j++) {
			if (oriPalette_R[j][i] < 0) oriPalette_R[j][i] = 0;
			else if (oriPalette_R[j][i] > 255) oriPalette_R[j][i] = 255;

			if (oriPalette_G[j][i] < 0) oriPalette_G[j][i] = 0;
			else if (oriPalette_G[j][i] > 255) oriPalette_G[j][i] = 255;

			if (oriPalette_B[j][i] < 0) oriPalette_B[j][i] = 0;
			else if (oriPalette_B[j][i] > 255) oriPalette_B[j][i] = 255;
		}
	}

	cv::VideoWriter videoWriter(Path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(50, 50 * paletteNum), true);

	for (size_t i = 0; i < videoNum; i++) {
		cv::Mat frame(50 * paletteNum, 50, CV_8UC3);
		for (size_t row = 0; row < 50 * paletteNum; row++) {
			uchar* uc_pixel = frame.data + row * frame.step;
			int index = row / 50;
			for (size_t col = 0; col < 50; col++) {
				uc_pixel[0] = (uchar)oriPalette_B[index][i];
				uc_pixel[1] = (uchar)oriPalette_G[index][i];
				uc_pixel[2] = (uchar)oriPalette_R[index][i];

				uc_pixel += 3;
			}
		}
		videoWriter << frame;
	}
}

void Data::exportImagePalette() {
	string Path = "./imagePalette/";

#pragma omp parallel for
	for (int i = 0; i < videoNum; i++) {
		cv::Vec3d color;
		cv::Mat frame(50 * paletteNum, 50, CV_8UC3);
		for (size_t row = 0; row < 50 * paletteNum; row++) {
			uchar* uc_pixel = frame.data + row * frame.step;
			int index = row / 50;
			color = { oriPalette_R[index][i],oriPalette_G[index][i],oriPalette_B[index][i] };
			LAB2RGB(color);
			for (size_t col = 0; col < 50; col++) {
				uc_pixel[0] = (uchar)color[2];
				uc_pixel[1] = (uchar)color[1];
				uc_pixel[2] = (uchar)color[0];

				uc_pixel += 3;
			}
		}
		cv::imwrite(Path + to_string(i) + ".jpg", frame);
	}

}

void Data::exportChangedPalette(QString filename) {
	string Path = filename.toStdString() + "/" + videoname + "_changedPalette.mp4";

	cv::VideoWriter videoWriter(Path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(50, 50 * paletteNum), true);

#pragma omp parallel for
	for (int i = 0; i < videoNum; i++) {
		for (size_t j = 0; j < paletteNum; j++) {
			if (changedPalette_R[j][i] < 0) changedPalette_R[j][i] = 0;
			else if (changedPalette_R[j][i] > 255) changedPalette_R[j][i] = 255;

			if (changedPalette_G[j][i] < 0) changedPalette_G[j][i] = 0;
			else if (changedPalette_G[j][i] > 255) changedPalette_G[j][i] = 255;

			if (changedPalette_B[j][i] < 0) changedPalette_B[j][i] = 0;
			else if (changedPalette_B[j][i] > 255) changedPalette_B[j][i] = 255;
		}
	}

	for (size_t i = 0; i < videoNum; i++) {
		cv::Mat frame(50 * paletteNum, 50, CV_8UC3);
		for (size_t row = 0; row < 50 * paletteNum; row++) {
			uchar* uc_pixel = frame.data + row * frame.step;
			int index = row / 50;
			for (size_t col = 0; col < 50; col++) {
				uc_pixel[0] = changedPalette_B[index][i];
				uc_pixel[1] = changedPalette_G[index][i];
				uc_pixel[2] = changedPalette_R[index][i];

				uc_pixel += 3;
			}
		}
		videoWriter << frame;
	}
}


void Data::setPaletteColor(int id, QColor c) {
	changedPalette_R[id][currentFrame] = qRed(c.rgb());
	changedPalette_G[id][currentFrame] = qGreen(c.rgb());
	changedPalette_B[id][currentFrame] = qBlue(c.rgb());

	int index = currentFrame * paletteNum + id;
	if (!isSelected[index]) {
		isSelected[index] = true;
		//selectedFrame.push_back(currentFrame);
		//selectedId.push_back(id);
		selectedColor[id].push_back(currentFrame);
	}

	for (size_t i = 0; i < paletteNum; i++) {
		for (size_t j = 0; j < selectedColor[i].size(); j++) {
			cout << "(" << i << ", " << selectedColor[i][j] << ") ";
		}
	}
	cout << endl;
}

void Data::readPaletteColor() {
	int palette_id, palette_frame, palette_r, palette_g, palette_b;
	ifstream file("C:\\Users\\jiany\\Desktop\\vr\\file.txt");
	string line;

	if (!file.is_open())
	{
		cout << "Failed to open file.\n";
		return;
	}

	while (getline(file, line)) {
		if (line.find("Palette ID:") != string::npos) {
			stringstream ss;
			string temp;
			char ch;

			ss.str(line);
			ss >> temp >> temp >> palette_id >> temp >> temp >> palette_frame;

			if (getline(file, line)) {
				ss.clear();
				ss.str(line);
				ss >> temp;
				ss >> palette_r;
				cout << palette_r << ' ';
				ss >> temp;
				ss >> palette_g;
				cout << palette_g << ' ';
				ss >> temp;
				ss >> palette_b;
				cout << palette_b << endl;

				changedPalette_R[palette_id][palette_frame] = palette_r;
				changedPalette_G[palette_id][palette_frame] = palette_g;
				changedPalette_B[palette_id][palette_frame] = palette_b;

				int index = palette_frame * paletteNum + palette_id;
				if (!isSelected[index]) {
					isSelected[index] = true;
					//selectedFrame.push_back(paletteNum);
					//selectedId.push_back(palette_id);
					selectedColor[palette_id].push_back(palette_frame);
				}

				imageRecolor(palette_frame);
			}
		}
	}

	file.close();

	for (size_t i = 0; i < paletteNum; i++) {
		for (size_t j = 0; j < selectedColor[i].size(); j++) {
			cout << "(" << i << ", " << selectedColor[i][j] << ") ";
		}
	}
	cout << endl;

	return;
}

void Data::exportPaletteColor() {
	std::ofstream file("C:\\Users\\jiany\\Desktop\\vr\\file.txt", std::ios::app);
	if (file.is_open()) {
		for (size_t i = 0; i < paletteNum; i++)
		{
			size_t ed = selectedColor[i].size();
			for (size_t j = 0; j < ed; j++)
			{
				file << "Palette ID: " << i << ", Frame: " << selectedColor[i][j] << "\n";
				file << "Colors:R= " << int(changedPalette_R[i][selectedColor[i][j]]);
				file << " ,G= " << int(changedPalette_G[i][selectedColor[i][j]]);
				file << " ,B= " << int(changedPalette_B[i][selectedColor[i][j]]) << "\n";
			}
		}
		file.close(); // 关闭文件
	}
	else {
		std::cerr << "Failed to open file for writing.\n";
	}

	return;
}


void Data::resetPaletteColor(int id) {
	changedPalette_R[id][currentFrame] = oriPalette_R[id][currentFrame];
	changedPalette_G[id][currentFrame] = oriPalette_G[id][currentFrame];
	changedPalette_B[id][currentFrame] = oriPalette_B[id][currentFrame];
}

void Data::resetFramePalettes() {
	for (size_t i = 0; i < paletteNum; i++) {
		changedPalette_R[i][currentFrame] = oriPalette_R[i][currentFrame];
		changedPalette_G[i][currentFrame] = oriPalette_G[i][currentFrame];
		changedPalette_B[i][currentFrame] = oriPalette_B[i][currentFrame];
	}
}

void Data::resetAllPaletteColors() {
	if (!isVideoOpen || !isPaletteCalc) return;
	for (size_t i = 0; i < paletteNum; i++) {
		memmove(changedPalette_R[i], oriPalette_R[i], sizeof(double) * videoNum);
		memmove(changedPalette_G[i], oriPalette_G[i], sizeof(double) * videoNum);
		memmove(changedPalette_B[i], oriPalette_B[i], sizeof(double) * videoNum);
	}

	selectedFrame.clear();
	selectedId.clear();

	for (size_t i = 0; i < paletteNum; i++) {
		selectedColor[i].clear();
	}
	for (size_t i = 0; i < paletteNum * videoNum; i++) {
		isSelected[i] = false;
	}
	emit updated();
}

void Data::removeSelection(int id) {
	int index = -1;
	for (size_t i = 0; i < selectedColor[id].size(); i++) {
		if (selectedColor[id][i] == currentFrame) index = i;
	}
	if (index == -1) return;

	for (vector<int>::iterator it = selectedColor[id].begin(); it != selectedColor[id].end();) {
		if (*it == currentFrame) {
			selectedColor[id].erase(it);
			break;
		}
		else it++;
	}

	isSelected[currentFrame * paletteNum + id] = false;

	for (size_t i = 0; i < paletteNum; i++) {
		for (size_t j = 0; j < selectedColor[i].size(); j++) {
			cout << "(" << i << ", " << selectedColor[i][j] << ") ";
		}
	}
	cout << endl;
}

// 计算调色板曲线的数量
double Data::calcPaletteNum(const vector<double>& seedsL, const vector<double>& seedsA, const vector<double>& seedsB, const vector<int>& clustersize, int threshold) {

	// 记录曲线的数量
	double paletteNum = 0;

	// 当前关键帧的超像素数
	int frameNum = seedsL.size();

	//cout << frameNum << endl;

	// 记录超像素的颜色
	cv::Vec3d color;

	// 存储代表颜色
	vector<cv::Vec3d> sample;

	// 存储颜色的权重
	vector<double> weights;

	for (size_t i = 0; i < frameNum; i++) {
		// 获取超像素的Lab色彩值
		color = { (double)seedsL[i], (double)seedsA[i], (double)seedsB[i] };
		
		// 存入代表颜色
		sample.push_back(color);

		// 把超像素范围内的像素数量作为对应方格的权重
		weights.push_back(clustersize[i]);
	}
	// 存储最高权重所在的索引
	int max_index = 0;

	// 存储最高权重的值
	double max_value = 0;

	// 存储调色板的代表颜色
	vector<cv::Vec3d> palette;

	// 根据代表颜色的数量进行对应次数的权重衰减
	for (size_t i = 0; i < sample.size(); i++) {
		max_value = 0;
		// 找出当前权重最大的代表颜色，并更新索引和最大值
		for (size_t j = 0; j < sample.size(); j++) {
			if (weights[j] > max_value) {
				max_index = j;
				max_value = weights[j];
			}
		}

		// 当权重衰减到一定范围后，停止衰减
		if (max_value < threshold) break;

		// 将当前权重最大的代表颜色录入调色板
		palette.push_back(sample[max_index]);

		// 衰减各代表颜色的权重
		for (size_t j = 0; j < sample.size(); j++) {
			weights[j] *= (1 - exp(
				-pow(norm(sample[max_index], sample[j]) / 80, 4)
			));
		}

	}
	// 视频调色板的颜色数量的累加
	paletteNum += palette.size();
	
	return paletteNum;
}

void Data::exportCurrentFrame(QString filename) {
	cv::Mat frame(video_rows, video_cols, CV_8UC3);

	string Path = filename.toStdString() + "/" + videoname + "_" + to_string(currentFrame) + ".jpg";

	for (size_t row = 0; row < video_rows; row++) {
		uchar* uc_pixel = frame.data + row * frame.step;
		int index = row * video_cols;
		for (size_t col = 0; col < video_cols; col++) {
			int index = row * video_cols + col;
			uc_pixel[0] = changedVideo_B[currentFrame][index];
			uc_pixel[1] = changedVideo_G[currentFrame][index];
			uc_pixel[2] = changedVideo_R[currentFrame][index];

			uc_pixel += 3;
			index++;
		}
	}
	cv::imwrite(Path, frame);
}