//
//  main.cpp
//  biometrics-04
//
//  Created by Martin Kiesel on 18/04/16.
//  Copyright © 2016 Martin Kiesel. All rights reserved.
//

#include <iostream>
#include <ctime>


#include "mahalanobis_distance.hpp"
#include "euclidean_distance.hpp"
#include "svm.hpp"
#include "pca.hpp"

#include <opencv2/core.hpp>

const biometrics_4::types::uintf TRAIN_SIZE = 10;
const biometrics_4::types::uintf TEST_SIZE = 2;

int main(int argc, const char * argv[])
{
    using namespace biometrics_4;
    using namespace classifier;
    std::vector<double> euclids, svms;


//    cv::Mat tst(4, 3, CV_32FC1);
//    cv::Mat t;
//    tst.row(0).col(0) = 5;
//    tst.row(0).col(1) = 2;
//    tst.row(0).col(2) = 8;
//
//    tst.row(1).col(0) = 8;
//    tst.row(1).col(1) = 6;
//    tst.row(1).col(2) = 3;
//
//    tst.row(2).col(0) = 11;
//    tst.row(2).col(1) = 2;
//    tst.row(2).col(2) = 7;
//
//    tst.row(3).col(0) = 2;
//    tst.row(3).col(1) = 1;
//    tst.row(3).col(2) = 9;
//
//    std::cout << tst << std::endl;
//    cv::reduce(tst, t, 0, CV_REDUCE_AVG);
//    std::cout << t << std::endl;

    biometrics_4::pca::Pca pca(TRAIN_SIZE, TEST_SIZE);
    const types::PcaData &pca_data = pca.getPcaData();
    
    euclid::EuclidDistance euclid(&pca_data);
    svm::Svm svm(&pca_data);
    mahalanobis::Mahalanobis mahalanobis(&pca_data);
    //size_t start = 0, end = 0;

//    for (types::uintf i = 1; i <= 60; i++)
    for (const auto &i : {1, 10, 15, 20, 25, 30, 45, 50, 55, 60})
    {
        //start = clock();
        pca.setComponents(i);
        //end = clock();
        //std::cout << (end - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;
        svms.push_back(mahalanobis.classify());
    }

    for (const auto &e : svms)
    {
        std::cout << e << std::endl;
    }

    return 0;
}