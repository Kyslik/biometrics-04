//
//  main.cpp
//  biometrics-04
//
//  Created by Martin Kiesel on 18/04/16.
//  Copyright © 2016 Martin Kiesel. All rights reserved.
//

#include <iostream>
#include <ctime>

#include "pca.hpp"
#include "euclidean_distance.hpp"
#include "svm.hpp"

const biometrics_4::types::uintf TRAIN_SIZE = 10;
const biometrics_4::types::uintf TEST_SIZE = 2;

int main(int argc, const char * argv[])
{
    using namespace biometrics_4;
    using namespace classifier;
    std::vector<double> euclids, svms;

    biometrics_4::pca::Pca pca(TRAIN_SIZE, TEST_SIZE);
    const types::PcaData &pca_data = pca.getPcaData();

    euclid::EuclidDistance euclid(&pca_data);
    svm::Svm svm(&pca_data);

    //size_t start = 0, end = 0;

//    for (types::uintf i = 1; i <= 60; i++)
    for (const auto &i : {1, 5, 10, 15, 20, 25, 30, 45, 50, 55, 60})
    {
        //start = clock();
        pca.setComponents(i);
        //end = clock();
        //std::cout << (end - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;
        //euclids.push_back(euclid.classify());
        svms.push_back(svm.classify(i));
    }

    for (const auto &e : svms)
    {
        std::cout << e << std::endl;
    }

    return 0;
}