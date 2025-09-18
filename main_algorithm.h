#ifndef DATA_H
#define DATA_H


#include<algorithm>
#include<QString>
#include<QObject>
#include<vector>
#include <QThread>
#include "utility.h"
#include <string>
#include <array>
#include <opencv2/opencv.hpp>
#include <QFile>
#include <QFileinfo.h>
#include <QDebug>
#include <cmath>
#include <QProgressDialog>
#include <QMessagebox>
#include <QTime>
#include <omp.h>
#include <map>
#include "my_util.h"
#include <fstream>
#include <cstring>
#include <io.h>
#include <direct.h>
#include <nlopt.hpp>
#include <Eigen/Dense>
#include "SLIC.h"
using namespace std;

typedef short int sint;

class Data : public QObject
{
	Q_OBJECT

public:
	int currentFrame;
	bool isPaletteCalc;
	bool isVideoOpen;
	vector<bool> isSelected;

	void OpenVideo(QString path);
	void calcPalette();
	double calcPaletteNum(const vector<double>& seedsL, const vector<double>& seedsA, const vector<double>& seedsB, const vector<int>& clustersize, int threshold );
	void extractFirstPalette(int index, const vector<double>& seedsL, const vector<double>& seedsA, const vector<double>& seedsB);
	void extractPalette(int index, const vector<double>& seedsL, const vector<double>& seedsA, const vector<double>& seedsB, int max_it = 1);
	double calc_param(int index, vector<double>& lamda);
	void calc_singlePoint_weights(int findex, const cv::Vec3d& point, double param, const vector<double>& lamda, vector<double>& singleWeight);
	void calcWeights(int findex);
	void imageRecolor(int findex);
	void videoRecolor();
	void curveDeformation();

	int getWidth() const { return video_cols; }
	int getHeight() const { return video_rows; }
	int getVideoNum() const { return videoNum; }
	int getPaletteNum() const { return paletteNum; }
	double getFps() const { return fps; }

	sint* getCurrentImage_R(bool isAfter) const { return isAfter ? changedVideo_R[currentFrame] : oriVideo_R[currentFrame]; }
	sint* getCurrentImage_G(bool isAfter) const { return isAfter ? changedVideo_G[currentFrame] : oriVideo_G[currentFrame]; }
	sint* getCurrentImage_B(bool isAfter) const { return isAfter ? changedVideo_B[currentFrame] : oriVideo_B[currentFrame]; }
	void changePosition(int frameNum);
	void exportVideo(QString filename);
	void exportOriPalette(QString filename);
	void exportImagePalette();
	void exportChangedPalette(QString filename);
	void exportCurrentFrame(QString filename);
	
	Data();
	void close();
	//void Reset();
	
	double** getChangedPalette_R() { return changedPalette_R; }
	double** getChangedPalette_G() { return changedPalette_G; }
	double** getChangedPalette_B() { return changedPalette_B; }
	double** getOriginalPalette_R() { return oriPalette_R; }
	double** getOriginalPalette_G() { return oriPalette_G; }
	double** getOriginalPalette_B() { return oriPalette_B; }

	double* getCurrentChangedPalette_R() { return changedPalette_R[currentFrame]; }
	double* getCurrentChangedPalette_G() { return changedPalette_G[currentFrame]; }
	double* getCurrentChangedPalette_B() { return changedPalette_B[currentFrame]; }
	double* getCurrentOriPaletet_R() { return oriPalette_R[currentFrame]; }
	double* getCurrentOriPaletet_G() { return oriPalette_G[currentFrame]; }
	double* getCurrentOriPaletet_B() { return oriPalette_B[currentFrame]; }

	void setPaletteColor(int id, QColor c);
	void readPaletteColor();
	void exportPaletteColor();
	void resetPaletteColor(int id);
	void resetFramePalettes();
	void resetAllPaletteColors();
	void removeSelection(int id);

	vector<double> A;

public slots:
signals:
	void updated();

private:
	string videoname;
	sint** oriVideo_R;
	sint** oriVideo_G;
	sint** oriVideo_B;
	sint** changedVideo_R;
	sint** changedVideo_G;
	sint** changedVideo_B;

	double** oriPalette_R;
	double** oriPalette_G;
	double** oriPalette_B;
	double** changedPalette_R;
	double** changedPalette_G;
	double** changedPalette_B;

	float** weights;

	vector<vector<double>> bezeirControl_L;
	vector<vector<double>> bezeirControl_A;
	vector<vector<double>> bezeirControl_B;

	vector<vector<int>> selectedColor;
	vector<int> selectedFrame;
	vector<int> selectedId;

	vector<double> coefficient1;
	vector<double> coefficient2;
	
	vector<vector<double>> oriDiri_L;
	vector<vector<double>> oriDiri_A;
	vector<vector<double>> oriDiri_B;

	int paletteNum;
	int controlNum;

	int videoNum;
	int video_rows;
	int video_cols;
	int videoSize;
	int video_depth = 3;
	int fps;
};

#endif // DATA_H