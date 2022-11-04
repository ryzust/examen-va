/**
TODO:
Borrar este comentario
*/

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include <map>
#define pi 3.1416
#define e 2.72


using namespace cv;
using namespace std;

/*
	Dado un tama�o de kernel y sigma calcula el kernel gaussiano
*/
vector<vector<float>> gaussianKernel(int kSize, int sigma) {
	int amountSlide = (kSize - 1) / 2;
	vector<vector<float>> v(kSize, vector<float>(kSize, 0));
	// si el centro es (0,0)
	for (int i = -amountSlide; i <= amountSlide; i++)
	{
		for (int j = -amountSlide; j <= amountSlide; j++)
		{
			float resultado = (1 / (2 * pi * sigma * sigma)) * pow(e, -((i * i + j * j) / (2 * sigma * sigma)));
			v[i + amountSlide][j + amountSlide] = resultado;
		}
	}
	return v;
}


// Operacion de convolucion para un pixel en una matriz con coordenadas x,y utilizando un kernel
float applyFilterToPix(Mat original, vector<vector<float>> kernel, int kSize, int x, int y) {
	int rows = original.rows;
	int cols = original.cols;
	// Denominamos amountSlide como la cantidad de casillas entre el centro de una matriz y uno de sus bordes
	// Tambien puede verse como el excedente al aplicar un kernel sobre una matriz en una de sus esquinas
	int amountSlide = (kSize - 1) / 2;
	float sumFilter = 0;
	float sumKernel = 0;
	// Recorremos de esta manera para asegurar que el pixel central tendr� coordenadas (0,0)
	for (int i = -amountSlide; i <= amountSlide; i++)
	{
		for (int j = -amountSlide; j <= amountSlide; j++)
		{
			float kTmp = kernel[i + amountSlide][j + amountSlide];
			int tmpX = x + i;
			int tmpY = y + j;
			float tmp = 0;
			// si el pixel con coordenadas (tmpX,tmpY) existe...
			if (!(tmpX < 0 || tmpX >= cols || tmpY < 0 || tmpY >= rows)) {
				tmp = original.at<uchar>(Point(tmpX, tmpY));
			}

			sumFilter += (kTmp * tmp);
			sumKernel += kTmp;
		}
	}
	if (sumKernel == 0) { sumKernel = 1; }
	return sumFilter / sumKernel;
}

// Operacion de convolucion para una matriz con un kernel
Mat applyFilterToMat(Mat original, vector<vector<float>> kernel) {
	// Considerando un kernel simetrico
	int kSize = kernel[0].size();
	Mat filteredImg(original.rows, original.cols, CV_8UC1);
	for (int i = 0; i < original.rows; i++)
	{
		for (int j = 0; j < original.cols; j++) {
			int pixVal = applyFilterToPix(original, kernel, kSize, i, j);
			if (pixVal < 0) { pixVal = 0; }
			if (pixVal > 255) { pixVal = 255; }
			filteredImg.at<uchar>(Point(i, j)) = uchar(pixVal);
		}
	}
	return filteredImg;
}


Mat generateBorder(Mat original, int borderSize) {
	// Nuevas dimensiones despues de aplicar borde
	int extRows = original.rows + borderSize;
	int extCols = original.cols + borderSize;

	// Creacion de imagen redimensionada
	Mat newImg(extRows, extCols, CV_8UC1);
	for (int i = 0; i < extRows; i++) {
		for (int j = 0; j < extCols; j++) {
			// relleno de bordes
			if (i <= borderSize || j <= borderSize || i > original.rows || j > original.cols) {
				newImg.at<uchar>(Point(i, j)) = uchar(0);
			}
			else {
				//Relleno de imagen original en el centro de la nueva imagen
				newImg.at<uchar>(Point(i, j)) = uchar(original.at<uchar>(Point(i - borderSize, j - borderSize)));
			}
		}
	}
	return newImg;
}

void printMatSize(Mat mat, String name) {
	cout << name << ": [" << mat.rows << "," << mat.cols << "]" << endl;
}

// Convierte una imagen a su equivalente en escala de grises utilizando el estandar NTSC
Mat convertToGrayscale(Mat img) {
	Mat grayscale(img.rows, img.cols, CV_8UC1);
	// Pasamos a escala de grises usando NTSC
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{

			double R = img.at<Vec3b>(Point(j, i)).val[2];
			double G = img.at<Vec3b>(Point(j, i)).val[1];
			double B = img.at<Vec3b>(Point(j, i)).val[0];

			grayscale.at<uchar>(Point(j, i)) = uchar(0.299 * R + 0.587 * G + 0.114 * B);
		}
	}
	return grayscale;
}

// Toma dos matrices y obtiene su magnitud
Mat getMagnitude(Mat m1, Mat m2) {
	// asumiendo que m1 y m2 tienen dimensiones iguales
	Mat res(m1.rows, m1.cols, CV_8UC1);
	int x, y = 0;
	for (int i = 0; i < m1.rows; i++) {
		for (int j = 0; j < m1.cols; j++) {

			x = m1.at<uchar>(Point(j, i));
			y = m2.at<uchar>(Point(j, i));

			res.at<uchar>(Point(j, i)) = uchar(sqrt(pow(x, 2) + pow(y, 2)));
		}
	}
	return res;
}

// Dadas los gradientes en x y y, devuelve la direccion de los bordes para cada pixel
vector<vector<float>> getEdgeAngles(Mat gx, Mat gy) {
	// asumiendo que m1 y m2 tienen dimensiones iguales
	vector<vector<float>> angles(gx.rows, vector<float>(gx.cols, 0));
	int x, y = 0;
	for (int i = 0; i < gx.rows; i++) {
		for (int j = 0; j < gx.cols; j++) {

			x = gx.at<uchar>(Point(j, i));
			y = gy.at<uchar>(Point(j, i));
			// obtenemos el angulo en el que se encuentra el borde en grados
			angles[i][j] = (atan2(y, x) * 180) / pi;
		}
	}
	return angles;
}

// Frecuencias acumuladas para cada valor de pixel
map<int, int> getCDFMap(Mat img) {
	map<int, int> cdf;

	int min = 256;
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++) {
			int tmp = img.at<uchar>(Point(j, i));
			cdf[tmp] += 1;
		}
	}

	int acc = 0;
	map<int, int>::iterator it = cdf.begin();
	min = (*it).second;
	for (it; it != cdf.end(); it++)
	{
		int tmp = (*it).second;
		int key = (*it).first;

		cdf[key] += acc;
		acc = cdf[key];

	}

	cdf[-1] = min;
	return cdf;
}

// Funcion histograma
map<int, int> getHvMap(map<int, int> cdf, int sideLength) {
	map<int, int> Hv;
	int min = cdf[-1];
	cdf.erase(-1);
	int mn = sideLength * sideLength;
	for (map<int, int>::iterator i = cdf.begin(); i != cdf.end(); i++)
	{
		int tmp_cdf = (*i).second;
		int key = (*i).first;
		float val = (static_cast<float>(tmp_cdf-min) / static_cast<float>(mn-min)) * 255.0;
		Hv[key] = round(val);

	}
	return Hv;
}

Mat equalize(Mat m1, map<int, int> Hv) {
	// asumiendo que m1 y m2 tienen dimensiones iguales
	Mat res(m1.rows, m1.cols, CV_8UC1);
	int x = 0;
	for (int i = 0; i < m1.rows; i++) {
		for (int j = 0; j < m1.cols; j++) {

			x = static_cast<int>(m1.at<uchar>(Point(j, i)));
			// reemplazamos por el valor obtenido en la funcion de histograma
			res.at<uchar>(Point(j, i)) = uchar(Hv[x]);
		}
	}
	return res;
}

// Operacion de convolucion para un pixel en una matriz con coordenadas x,y utilizando un kernel
float nonMaxPix(Mat original, vector<vector<float>> angles, int x, int y) {
	int rows = original.rows;
	int cols = original.cols;
	// Denominamos amountSlide como la cantidad de casillas entre el centro de una matriz y uno de sus bordes
	// Tambien puede verse como el excedente al aplicar un kernel sobre una matriz en una de sus esquinas
	int amountSlide = 1;
	// Recorremos de esta manera para asegurar que el pixel central tendr� coordenadas (0,0)
	float pix = original.at<uchar>(Point(x, y));
	for (int i = -amountSlide; i <= amountSlide; i++)
	{
		for (int j = -amountSlide; j <= amountSlide; j++)
		{
			int tmpX = x + i;
			int tmpY = y + j;
			float tmp = 0;
			float angle1 = angles[x][y];
			// si el pixel con coordenadas (tmpX,tmpY) existe...
			if (!(tmpX < 0 || tmpX >= cols || tmpY < 0 || tmpY >= rows)) {
				tmp = original.at<uchar>(Point(tmpX, tmpY));
				// si el pixel actual no es el mas fuerte en su direccion
				float angle2 = angles[tmpX][tmpY];
				if (tmp > pix && angle1 == angle2) {
					return 0;
				}
			}

		}
	}

	return pix;
}

// Operacion de convolucion para una matriz con un kernel
Mat nonMaxMat(Mat original, vector<vector<float>> angles) {
	// Considerando un kernel simetrico
	Mat filteredImg(original.rows, original.cols, CV_8UC1);
	for (int i = 0; i < original.rows; i++)
	{
		for (int j = 0; j < original.cols; j++) {
			int pixVal = nonMaxPix(original, angles, i, j);
			if (pixVal < 0) { pixVal = 0; }
			if (pixVal > 255) { pixVal = 255; }
			filteredImg.at<uchar>(Point(i, j)) = uchar(pixVal);
		}
	}
	return filteredImg;
}

int getMaxIntensity(Mat img) {
	int filas = img.rows;
	int columnas = img.cols;

	int max = 0;

	for (int i = 0; i < filas; i++) {
		for (int j = 0; j < columnas; j++) {
			if (img.at<uchar>(Point(i, j)) > max) {
				max = img.at<uchar>(Point(i, j));
			}
		}
	}

	return max;
}

bool is8Connected(Mat m1, int x, int y) {
	int rows = m1.rows;
	int cols = m1.cols;
	// Denominamos amountSlide como la cantidad de casillas entre el centro de una matriz y uno de sus bordes
	// Tambien puede verse como el excedente al aplicar un kernel sobre una matriz en una de sus esquinas
	int amountSlide = 1;
	// Recorremos de esta manera para asegurar que el pixel central tendr� coordenadas (0,0)
	float pix = m1.at<uchar>(Point(x, y));
	for (int i = -amountSlide; i <= amountSlide; i++)
	{
		for (int j = -amountSlide; j <= amountSlide; j++)
		{
			int tmpX = x + i;
			int tmpY = y + j;
			float tmp = 0;
			// si el pixel con coordenadas (tmpX,tmpY) existe...
			if (!(tmpX < 0 || tmpX >= cols || tmpY < 0 || tmpY >= rows)) {
				tmp = m1.at<uchar>(Point(tmpX, tmpY));
				// si hay un vecino 8 conectado 
				if (tmp == 255) {
					return true;
				}
			}

		}
	}
	return false;
}

Mat hysteresis(Mat imgNonMax, float sup, float inf) {
	Mat imgHysteresis(imgNonMax.rows, imgNonMax.cols, CV_8UC1);

	// Obtenemos el equivalente 
	sup = getMaxIntensity(imgNonMax) * sup;
	inf = sup * inf;

	int weak = inf;
	int strong = 255;
	int irrelevant = 0;

	Mat imgHysteresis2(imgNonMax.rows, imgNonMax.cols, CV_8UC1);

	// marcar valores como fuertes, irrelevantes o debiles
	for (int i = 0; i < imgNonMax.rows; i++) {
		for (int j = 0; j < imgNonMax.cols; j++) {

			if (imgNonMax.at<uchar>(Point(i, j)) >= sup) {
				imgHysteresis.at<uchar>(Point(i, j)) = strong;
			}
			else if (inf < imgNonMax.at<uchar>(Point(i, j)) < sup) {
				imgHysteresis.at<uchar>(Point(i, j)) = weak;
			}
			else {
				imgHysteresis.at<uchar>(Point(i, j)) = irrelevant;
			}
		}
	}
	
	for (int i = 0; i < imgNonMax.rows; i++) {
		for (int j = 0; j < imgNonMax.cols; j++) {
			// volvemos fuertes aquellos pixeles debiles 8-conectados
			if (imgHysteresis.at<uchar>(Point(i, j)) == weak) {
				if (is8Connected(imgHysteresis, i, j)) {
					imgHysteresis2.at<uchar>(Point(i, j)) = strong;
				}
				else {
					imgHysteresis2.at<uchar>(Point(i, j)) = irrelevant;
				}
			}else {
				imgHysteresis2.at<uchar>(Point(i, j)) = imgHysteresis.at<uchar>(Point(i, j));
			}
			
		}
	}

	return imgHysteresis2;
}

int main()
{
	int sigma = 1;
	int kSize = 3;
	cout << "Ingresa el tamano del kernel" << endl;
	cin >> kSize;

	if (kSize % 2 == 0 || kSize <= 0) {
		cout << "Valor de kernel invalido" << endl;
		exit(0);
	}

	cout << "Ingresa sigma" << endl;
	cin >> sigma;

	if (sigma <= 0) {
		cout << "Valor de sigma invalido" << endl;
		exit(0);
	}

	char NombreImagen[] = "C:/Users/Lance/Desktop/lena.jpg";
	Mat imagen;

	// operadores sobel
	vector<vector<float>> gy
	{
		{1, 2, 1},
		{0, 0, 0},
		{-1, -2, -1}
	};
	vector<vector<float>> gx
	{
		{-1, 0, 1},
		{-2, 0, 2},
		{-1, 0, 1}
	};

	imagen = imread(NombreImagen);

	if (!imagen.data)
	{
		cout << "Error al cargar la imagen: " << NombreImagen << endl;
		exit(0);
	}


	// Procesamiento
	int fila_original = imagen.rows;
	int columna_original = imagen.cols;

	Mat paso2 = convertToGrayscale(imagen);

	vector<vector<float>> gKernel = gaussianKernel(kSize, sigma);
	Mat paso3 = applyFilterToMat(paso2, gKernel);

	map<int, int> cum_freq = getCDFMap(paso3);
	map<int, int> hist_values = getHvMap(cum_freq, fila_original);
	Mat paso4 = equalize(paso3, hist_values);
	Mat paso51 = applyFilterToMat(paso4, gx);
	Mat paso52 = applyFilterToMat(paso4, gy);
	Mat paso5 = getMagnitude(paso51, paso52);


	vector<vector<float>> angulos = getEdgeAngles(paso51, paso52);
	Mat nonmax2 = nonMaxMat(paso5, angulos);
	Mat hys2 = hysteresis(nonmax2, 0.9, 0.35);

	namedWindow("original", WINDOW_AUTOSIZE);
	imshow("original", imagen);

	namedWindow("gris2", WINDOW_AUTOSIZE);
	imshow("gris2", paso2);

	namedWindow("filtro2", WINDOW_AUTOSIZE);
	imshow("filtro2", paso3);

	namedWindow("gx2", WINDOW_AUTOSIZE);
	imshow("gx2", paso51);

	namedWindow("gy2", WINDOW_AUTOSIZE);
	imshow("gy2", paso52);

	namedWindow("g2", WINDOW_AUTOSIZE);
	imshow("g2", paso5);

	namedWindow("eq2", WINDOW_AUTOSIZE);
	imshow("eq2", paso4);

	namedWindow("hy2", WINDOW_AUTOSIZE);
	imshow("hy2", hys2);
	waitKey(0);
	return 0;
}