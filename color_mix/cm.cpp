#include <bits/stdc++.h>
#include <iostream>
#include "Eigen/Dense"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "nanoflann.hpp"
#include "KDTreeVectorOfVectorsAdaptor.h"
#include <ctime>
#include <cstdlib>
using namespace nanoflann;
using namespace std;
using namespace cv;
using namespace Eigen;

const int dim = 5;

typedef vector<vector<double>> my_vector_of_vectors_t;
typedef vector<set<int, greater<int>>> my_vector_of_set_t;


void generateFVectorCM(my_vector_of_vectors_t &samples, Mat &img)
{
	// CV_Assert(img.depth() == CV_8U);

	int channels = img.channels();
	int nRows = img.rows;
	int nCols = img.cols;

	samples.resize(nRows*nCols);
	
	int i,j,k;	
	for( i = 0; i < nRows; ++i)
		for ( j = 0; j < nCols; ++j){
			samples[i*nCols+j].resize(dim);		
			samples[i*nCols+j][0] = img.at<cv::Vec3b>(i,j)[0];
			samples[i*nCols+j][1] = img.at<cv::Vec3b>(i,j)[1];
			samples[i*nCols+j][2] = img.at<cv::Vec3b>(i,j)[2];
			samples[i*nCols+j][3] = double(i);
			samples[i*nCols+j][4] = double(j);
		}

	cout << "feature vectors done"<<endl;
}


void kdtree_CM(Mat &img, my_vector_of_vectors_t& indm, my_vector_of_vectors_t& samples)
{
	
	// Generate feature vectors for intra U:
	generateFVectorCM(samples, img);	

	// Query point: same as samples from which KD tree is generated

	// construct a kd-tree index:
	// Dimensionality set at run-time (default: L2)
	// ------------------------------------------------------------
	typedef KDTreeVectorOfVectorsAdaptor< my_vector_of_vectors_t, double >  my_kd_tree_t;
	my_kd_tree_t mat_index(dim /*dim*/, samples, 10 /* max leaf */ );
	mat_index.index->buildIndex();

	// do a knn search with cm = 20
	const size_t num_results = 20; 

	int i,j;
	int N = img.rows*img.cols;

	vector<size_t> ret_indexes(num_results);
	vector<double> out_dists_sqr(num_results);
	nanoflann::KNNResultSet<double> resultSet(num_results);

	indm.resize(N);
	for(i = 0; i < N; i++){
		resultSet.init(&ret_indexes[0], &out_dists_sqr[0] );
		mat_index.index->findNeighbors(resultSet, &samples[i][0], nanoflann::SearchParams(10));	

		// cout << "knnSearch(nn="<<num_results<<"): \n";
		indm[i].resize(num_results);
		for (j = 0; j < num_results; j++){
			// cout << "ret_index["<<j<<"]=" << ret_indexes[j] << " out_dist_sqr=" << out_dists_sqr[j] << endl;
			indm[i][j] = ret_indexes[j];
		}
	}
}



void lle(my_vector_of_vectors_t& indm, my_vector_of_vectors_t& samples, float eps){
	
	int k = indm[0].size(); //number of neighbours that we are considering 
	int n = indm.size(); //number of pixels
	my_vector_of_vectors_t wcm;
	wcm.resize(n);

	MatrixXf C = MatrixXf::Zero(k, k);
	VectorXf rhs = VectorXf::Ones(k);

	MatrixXf Z(dim, k);
	VectorXf weights;
	for(int i = 0; i < n; i++){
		// filling values in Z
		int index_nbr;
		for(int j = 0; j < k; j++){
			index_nbr = indm[i][j];
			for(int p = 0; p < dim; p++){
				Z(p,j) = samples[index_nbr][p] - samples[i][p];
			}
		}
		cout<<"done"<<endl;
		// adding some constant to ensure invertible matrices
		C = Z.transpose()*Z;
		C.diagonal().array() += eps;
		weights = C.ldlt().solve(rhs);
		weights /= weights.sum();
		wcm[i].resize(k);
		cout<<weights<<endl;
		for(int j = 0; j < k; j++)
			wcm[i][j] = weights[j];
	}


}


int main()
{
	Mat image,tmap;
	my_vector_of_vectors_t samples, indm, Euu;
	string img_path = "../../data/input_lowres/plasticbag.png";
	image = imread(img_path, CV_LOAD_IMAGE_COLOR);   // Read the file
	kdtree_CM(image, indm, samples);
	cout<<"KD Tree done"<<endl;
	float eps = 0.0001;
	lle(indm, samples, eps);
}
