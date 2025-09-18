#include "SLIC.h"

// ��sRGB��ɫֵת��ΪCIE 1931 XYZɫ�ʿռ�
void RGB2XYZ(
	const int& sR,
	const int& sG,
	const int& sB,
	double& X,
	double& Y,
	double& Z)
{
	double R = sR / 255.0;
	double G = sG / 255.0;
	double B = sB / 255.0;

	double r, g, b;

	if (R <= 0.04045)	r = R / 12.92;
	else				r = pow((R + 0.055) / 1.055, 2.4);
	if (G <= 0.04045)	g = G / 12.92;
	else				g = pow((G + 0.055) / 1.055, 2.4);
	if (B <= 0.04045)	b = B / 12.92;
	else				b = pow((B + 0.055) / 1.055, 2.4);

	X = r * 0.4124564 + g * 0.3575761 + b * 0.1804375;
	Y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750;
	Z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041;
}

// ��sRGB��ɫֵת��ΪLabɫ�ʿռ�ֵ
void RGB2LAB(const int& sR, const int& sG, const int& sB, double& lval, double& aval, double& bval)
{
	//------------------------
	// sRGB to XYZ conversion
	//------------------------
	double X, Y, Z;
	RGB2XYZ(sR, sG, sB, X, Y, Z);

	//------------------------
	// XYZ to LAB conversion
	//------------------------
	double epsilon = 0.008856;	//actual CIE standard
	double kappa = 903.3;		//actual CIE standard

	double Xr = 0.950456;	//reference white
	double Yr = 1.0;		//reference white
	double Zr = 1.088754;	//reference white

	double xr = X / Xr;
	double yr = Y / Yr;
	double zr = Z / Zr;

	double fx, fy, fz;
	if (xr > epsilon)	fx = pow(xr, 1.0 / 3.0);
	else				fx = (kappa * xr + 16.0) / 116.0;
	if (yr > epsilon)	fy = pow(yr, 1.0 / 3.0);
	else				fy = (kappa * yr + 16.0) / 116.0;
	if (zr > epsilon)	fz = pow(zr, 1.0 / 3.0);
	else				fz = (kappa * zr + 16.0) / 116.0;

	lval = 116.0 * fy - 16.0;
	aval = 500.0 * (fx - fy);
	bval = 200.0 * (fy - fz);
}

// ��һ��RGB��ɫֵת��Ϊ��Ӧ��Labɫ�ʿռ�ֵ
void DoRGBtoLABConversion(
	const sint* imageR,
	const sint* imageG,
	const sint* imageB,
	double* imgL,
	double* imgA,
	double* imgB,
	const int& size
) {

	for (int i = 0; i < size; i++) {
		RGB2LAB(imageR[i], imageG[i], imageB[i], imgL[i], imgA[i], imgB[i]);
	}

}

// ��Labɫ�ʿռ��е�ͼ����б�Ե���
void DetectLabEdges(
	const double* lvec,
	const double* avec,
	const double* bvec,
	const int& width,
	const int& height,
	vector<double>& edges) {
	// ����ͼ����������
	int sz = width * height;

	// ���� edges �����Ĵ�С��ʹ����ͼ��������һ�£���������Ԫ�س�ʼ��Ϊ 0
	edges.resize(sz, 0);

	// ����ͼ��Ĵ�ֱ����
	for (int j = 1; j < height - 1; j++)
	{
		// ����ͼ���ˮƽ����
		for (int k = 1; k < width - 1; k++)
		{	
			// �������ǰ���ص����ڵ�����
			int i = j * width + k;

			// ���㵱ǰ���ص�ˮƽ������ݶȣ�Lab�ռ�������ľ��룩��ƽ��
			double dx = (lvec[i - 1] - lvec[i + 1]) * (lvec[i - 1] - lvec[i + 1]) +
				(avec[i - 1] - avec[i + 1]) * (avec[i - 1] - avec[i + 1]) +
				(bvec[i - 1] - bvec[i + 1]) * (bvec[i - 1] - bvec[i + 1]);

			// ���㵱ǰ���ص㴹ֱ������ݶȣ�Lab�ռ�������ľ��룩��ƽ��
			double dy = (lvec[i - width] - lvec[i + width]) * (lvec[i - width] - lvec[i + width]) +
				(avec[i - width] - avec[i + width]) * (avec[i - width] - avec[i + width]) +
				(bvec[i - width] - bvec[i + width]) * (bvec[i - width] - bvec[i + width]);

			//edges[i] = (sqrt(dx) + sqrt(dy));
			// ��ˮƽ����ʹ�ֱ�����ݶȵ�ƽ������Ϊ�ݶȵķ�ֵ
			edges[i] = (dx + dy);
		}
	}
}

void PerturbSeeds(
	const double* imgL,
	const double* imgA,
	const double* imgB,
	vector<double>& kseedsl,
	vector<double>& kseedsa,
	vector<double>& kseedsb,
	vector<double>& kseedsx,
	vector<double>& kseedsy,
	const int& m_width,
	const int& m_height,
	const vector<double>& edges)
{
	// �涨8�������ƫ����
	const int dx8[8] = { -1, -1,  0,  1, 1, 1, 0, -1 };
	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1 };

	// ���������ĵ�����
	int numseeds = kseedsl.size();

	// �������г���������
	for (int n = 0; n < numseeds; n++)
	{
		// ��ȡ��ǰ���������ĵ�ԭʼλ�ú�����
		int ox = kseedsx[n];//original x
		int oy = kseedsy[n];//original y
		int oind = oy * m_width + ox;

		// �洢��ǰ����
		int storeind = oind;

		// �������������ĵİ˸�����
		for (int i = 0; i < 8; i++)
		{
			// �Գ���������ʩ��ƫ�ƣ���¼�µ�λ��
			int nx = ox + dx8[i];//new x
			int ny = oy + dy8[i];//new y

			// ��ֹ�����߽�
			if (nx >= 0 && nx < m_width && ny >= 0 && ny < m_height)
			{
				// ���㵱ǰ������
				int nind = ny * m_width + nx;

				// ������������ĵ���λ�õı�Եǿ�ȵ��ڵ�ǰ�洢��λ��������洢����
				if (edges[nind] < edges[storeind])
				{
					storeind = nind;
				}
			}
		}
		// ������������ĵ��������ģ���Ҫͬʱ���ĳ��������ĵ�λ�ú�Labɫ�ʿռ��ֵ
		if (storeind != oind)
		{
			kseedsx[n] = storeind % m_width;
			kseedsy[n] = storeind / m_width;
			kseedsl[n] = imgL[storeind];
			kseedsa[n] = imgA[storeind];
			kseedsb[n] = imgB[storeind];
		}
	}
}

void GetLABXYSeeds_ForGivenK(
	const double* imgL,
	const double* imgA,
	const double* imgB,
	vector<double>& kseedsl,
	vector<double>& kseedsa,
	vector<double>& kseedsb,
	vector<double>& kseedsx,
	vector<double>& kseedsy,
	const int& K,
	const int& m_width,
	const int& m_height,
	const vector<double>& edgemag)
{
	// ����ͼ�����������
	int sz = m_width * m_height;
	// ����SILC�㷨���涨������֮��ļ����������
	double step = sqrt(double(sz) / double(K));
	// ����ȡ��
	int T = step;

	// ���� x �� y �����ƫ�ƣ����ڴӲ��������Ŀ�ʼ���񻯡�
	int xoff = step / 2;
	int yoff = step / 2;

	// �洢�����ص�����
	int n(0); 
	// �洢��ǰ������
	int r(0);
	// ����ͼ��Ĵ�ֱ����
	for (int y = 0; y < m_height; y++)
	{
		// �������ڴ�ֱ�����ϵ�����λ��
		int Y = y * step + yoff;
		// �������ڴ�ֱ�����ϵ�����λ�ó����߽�˵����������ȫ���ָ����
		if (Y > m_height - 1) break;

		// ����ͼ���ˮƽ����
		for (int x = 0; x < m_width; x++)
		{
			//int X = x*step + xoff;//square grid
			// ��������ˮƽ�����ϵ�����λ�ã����ݵ�ǰ��������żƫ����������ͬ��Ŀ����Ϊ��ʹ�����طֲ����Ӿ���
			int X = x * step + (xoff << (r & 0x1));//hex grid

			// ��������ˮƽ�����ϵ�����λ�ó����߽�˵�����еĳ�������ȫ���ָ����
			if (X > m_width - 1) break;

			// �������ǰ������������ͼ�������ڵ�����
			int i = Y * m_width + X;

			// �洢���������ĵ�Labɫ�ʿռ��ֵ
			kseedsl.push_back(imgL[i]);
			kseedsa.push_back(imgA[i]);
			kseedsb.push_back(imgB[i]);
			// �洢���������ĵ�λ����Ϣ
			kseedsx.push_back(X);
			kseedsy.push_back(Y);
			// ������������1
			n++;
		}
		// ��Ӧ������1
		r++;
	}

	PerturbSeeds(imgL, imgA, imgB, kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, m_width, m_height, edgemag);
}

void PerformSuperpixelSegmentation_VariableSandM(
	const double* imgL,
	const double* imgA,
	const double* imgB,
	vector<double>& kseedsl,
	vector<double>& kseedsa,
	vector<double>& kseedsb,
	vector<double>& kseedsx,
	vector<double>& kseedsy,
	const int& m_width,
	const int& m_height,
	vector<int>& klabels,
	const int& STEP,
	const int& NUMITR,
	vector<int>& clustersize)
{
	// �������ص�����
	int sz = m_width * m_height;
	// ��ȡ�����ؾ������ĵ�����
	const int numk = kseedsl.size();

	//double cumerr(99999.9);
	// �洢��������
	int numitr(0);

	//----------------
	// ���������صĴ�С
	int offset = STEP;
	if (STEP < 10) offset = STEP * 1.5;
	//----------------

	vector<double> sigmal(numk, 0);
	vector<double> sigmaa(numk, 0);
	vector<double> sigmab(numk, 0);
	vector<double> sigmax(numk, 0);
	vector<double> sigmay(numk, 0);
	vector<double> inv(numk, 0);//to store 1/clustersize[k] values
	vector<double> distxy(sz, DBL_MAX);
	vector<double> distlab(sz, DBL_MAX);
	vector<double> distvec(sz, DBL_MAX);
	vector<double> maxlab(numk, 10 * 10);//THIS IS THE VARIABLE VALUE OF M, just start with 10
	vector<double> maxxy(numk, STEP * STEP);//THIS IS THE VARIABLE VALUE OF M, just start with 10

	double invxywt = 1.0 / (STEP * STEP);//NOTE: this is different from how usual SLIC/LKM works

	// �涨��������������10��
	while (numitr < NUMITR)
	{
		//------
		//cumerr = 0;
		numitr++;
		//------
		// ��ÿ�����ص��볬���ؾ������ĵ��ۺ����ƶ����¸�ֵΪ˫���ȸ����������ֵ
		distvec.assign(sz, DBL_MAX);

		// ����ÿ�������ؾ�������
		for (int n = 0; n < numk; n++)
		{
			// ȷ�������ص������Ե�ǰ�����صľ�������Ϊ���ģ�y1 ��ʾ�ϱ߽磬y2 ��ʾ�±߽磬x1 ��ʾ��߽磬x2 ��ʾ�ұ߽�
			int y1 = std::max(0, (int)(kseedsy[n] - offset));
			int y2 = std::min(m_height, (int)(kseedsy[n] + offset));
			int x1 = std::max(0, (int)(kseedsx[n] - offset));
			int x2 = std::min(m_width, (int)(kseedsx[n] + offset));

			// �Դ�ֱ�����������������
			for (int y = y1; y < y2; y++)
			{
				// ��ˮƽ�����������������
				for (int x = x1; x < x2; x++)
				{
					// ���㵱ǰ���ص�����
					int i = y * m_width + x;
					// �ж��Ƿ�Խ�磬Խ������ֹ����
					_ASSERT(y < m_height&& x < m_width&& y >= 0 && x >= 0);

					// ��¼��ǰ���ص�Labɫ�ʿռ��ֵ
					double l = imgL[i];
					double a = imgA[i];
					double b = imgB[i];

					// ���㵱ǰ�����볬�������ĵ���ɫ���ƶ�
					distlab[i] = (l - kseedsl[n]) * (l - kseedsl[n]) +
						(a - kseedsa[n]) * (a - kseedsa[n]) +
						(b - kseedsb[n]) * (b - kseedsb[n]);

					// ���㵱ǰ�����볬�������ĵĿռ����ƶ�
					distxy[i] = (x - kseedsx[n]) * (x - kseedsx[n]) +
						(y - kseedsy[n]) * (y - kseedsy[n]);

					//------------------------------------------------------------------------
					// �������ǰ�����볬�������ĵ��ۺ����ƶ�
					double dist = distlab[i] / maxlab[n] + distxy[i] * invxywt;//only varying m, prettier superpixels
					//double dist = distlab[i]/maxlab[n] + distxy[i]/maxxy[n];//varying both m and S
					//------------------------------------------------------------------------

					// �����ǰ�����뵱ǰ�����ؾ������ĵ��ۺ����ƶȵ��ڼ�¼����С�ۺ����ƶ�
					if (dist < distvec[i])
					{
						// ���µ�ǰ������������ĵ���С�ۺ����ƶ�
						distvec[i] = dist;
						// �������ر��Ϊ���ڵ�ǰ�ĳ�����
						klabels[i] = n;
					}
				}
			}
		}
		//-----------------------------------------------------------------
		// Assign the max color distance for a cluster
		//-----------------------------------------------------------------
		// ���ε���ʱ�������߸�ֵΪ0
		if (0 == numitr)
		{
			maxlab.assign(numk, 1);
			maxxy.assign(numk, 1);
		}
		// �������Ŀռ����ƶȺ���ɫ���ƶ�
		{for (int i = 0; i < sz; i++)
		{
			if (maxlab[klabels[i]] < distlab[i]) maxlab[klabels[i]] = distlab[i];
			if (maxxy[klabels[i]] < distxy[i]) maxxy[klabels[i]] = distxy[i];
		}}
		//-----------------------------------------------------------------
		// Recalculate the centroid and store in the seed values
		//-----------------------------------------------------------------
		// ��ÿ�������ص�lab������xy�������ۻ�ֵ���Լ������������������¸�ֵΪ0
		sigmal.assign(numk, 0);
		sigmaa.assign(numk, 0);
		sigmab.assign(numk, 0);
		sigmax.assign(numk, 0);
		sigmay.assign(numk, 0);
		clustersize.assign(numk, 0);

		// ����ÿ�����أ�ͳ�Ƹ��������ĵ���������
		for (int j = 0; j < sz; j++)
		{
			// ��õ�ǰ���������ľ������ĵ�����
			int temp = klabels[j];
			// �жϸ������Ƿ����0��������ֹ����
			_ASSERT(klabels[j] >= 0);

			// �������ص�lab������xy�����ۼӽ������صĶ�Ӧ����
			sigmal[klabels[j]] += imgL[j];
			sigmaa[klabels[j]] += imgA[j];
			sigmab[klabels[j]] += imgB[j];
			sigmax[klabels[j]] += (j % m_width);
			sigmay[klabels[j]] += (j / m_width);

			// ��Ӧ�����ص���������������һ
			clustersize[klabels[j]]++;
		}

		// ȷ��ÿ���������ĵ�����������Ϊ0�����Ҽ��㳬����ͳ�������������ĵ���
		{for (int k = 0; k < numk; k++)
		{
			//_ASSERT(clustersize[k] > 0);
			if (clustersize[k] <= 0) clustersize[k] = 1;
			inv[k] = 1.0 / double(clustersize[k]);//computing inverse now to multiply, than divide later
		}}


	}
}

void generateSuperpixels(
	const sint* imageR,
	const sint* imageG,
	const sint* imageB,
	vector<double>& kseedsl,
	vector<double>& kseedsa,
	vector<double>& kseedsb,
	const int& K,
	const int& cols,
	const int& rows,
	vector<int>& clustersize) {

	// ����ͼ�����������
	int size = rows * cols;

	// �洢���������ĵ�λ��
	vector<double> kseedsx(0);
	vector<double> kseedsy(0);

	// ��¼ÿ�����������ĳ�����
	vector<int> labels(size, -1);

	// �洢���ص�Labɫ�ʿռ��ֵ
	double* imgL, * imgA, * imgB;
	imgL = new double[size];
	imgA = new double[size];
	imgB = new double[size];

	// ������� RGB ͼ��ת��Ϊ Lab ɫ�ʿռ�,����洢�� imgL��imgA��imgB ������
	DoRGBtoLABConversion(imageR, imageG, imageB, imgL, imgA, imgB, size);

	// �洢ͼ����ÿ�����ص�ı�Եǿ��
	vector<double> edgemag(0);
	// ��Labɫ�ʿռ��е�ͼ����б�Ե��⣬����Ӧ���صı�Եǿ�ȴ洢��edgemag������
	DetectLabEdges(imgL, imgA, imgB, cols, rows, edgemag);

	// ȷ�����������ĵ�λ��
	GetLABXYSeeds_ForGivenK(imgL, imgA, imgB, kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, K, cols, rows, edgemag);

	// �涨�����صĲ���
	int STEP = sqrt(double(size) / double(K)) + 2.0;

	// �ָ����
	PerformSuperpixelSegmentation_VariableSandM(imgL, imgA, imgB, kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, cols, rows, labels, STEP, 10, clustersize);

	delete[] imgL;
	delete[] imgA;
	delete[] imgB;
}