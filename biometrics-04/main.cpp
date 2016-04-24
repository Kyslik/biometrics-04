//
//  main.cpp
//  biometrics-04
//
//  Created by Martin Kiesel on 18/04/16.
//  Copyright © 2016 Martin Kiesel. All rights reserved.
//

#include <iostream>

#include "mahalanobis_distance.hpp"
#include "euclidean_distance.hpp"
#include "svm.hpp"
#include "pca.hpp"

const biometrics_4::types::uintf TRAIN_SIZE = 10;
const biometrics_4::types::uintf TEST_SIZE = 2;
const biometrics_4::types::uintf CROSS_VALIDATION = 15;
const biometrics_4::types::uintf MAX_PCA_COMPONENTS = 60;


biometrics_4::types::DoubleDimension calculateMean(const biometrics_4::types::DoubleMatrix &matrix);

int main(int argc, const char * argv[])
{
    using namespace biometrics_4;
    using namespace classifier;
    using namespace types;

    DoubleMatrix euclids(CROSS_VALIDATION, DoubleDimension(MAX_PCA_COMPONENTS, 0.0));
    DoubleMatrix svms(CROSS_VALIDATION, DoubleDimension(MAX_PCA_COMPONENTS, 0.0));
    DoubleMatrix mahals(CROSS_VALIDATION, DoubleDimension(MAX_PCA_COMPONENTS, 0.0));

    biometrics_4::pca::Pca pca(TRAIN_SIZE, TEST_SIZE);
    const PcaData &pca_data = pca.getPcaData();
    
    euclid::EuclidDistance euclid(&pca_data);
    svm::Svm svm(&pca_data);
    mahalanobis::Mahalanobis mahalanobis(&pca_data);

    for (uintf i = 0; i < CROSS_VALIDATION; i++)
    {
        for (uintf j = 1; j <= MAX_PCA_COMPONENTS; j++)
        {
            pca.setComponents(j);

            euclids[i][j - 1]  = euclid.classify();
            svms[i][j - 1]     = svm.classify();
            mahals[i][j - 1]   = mahalanobis.classify();
        }
        pca.randomize();
    }

    DoubleDimension euclidean_classifier    = calculateMean(euclids);
    DoubleDimension svm_classifier          = calculateMean(svms);
    DoubleDimension mahalanobis_classifier  = calculateMean(mahals);

    for (const auto &item : euclidean_classifier)
    {
        std::cout << item << std::endl;
    }

    std::cout << std::endl;

    for (const auto &item : svm_classifier)
    {
        std::cout << item << std::endl;
    }

    std::cout << std::endl;

    for (const auto &item : mahalanobis_classifier)
    {
        std::cout << item << std::endl;
    }

    return 0;
}

biometrics_4::types::DoubleDimension calculateMean(const biometrics_4::types::DoubleMatrix &matrix)
{
    using biometrics_4::types::DoubleDimension;
    using biometrics_4::types::uintf;

    if (CROSS_VALIDATION == 0)
        return matrix[0];

    DoubleDimension dst(MAX_PCA_COMPONENTS, 0.0);

    for (const auto &vec : matrix)
    {
        uintf i = 0;
        for (const auto &item : vec)
        {
            dst[i] += item;
            i++;
        }
    }

    for (auto &item : dst)
        item /= CROSS_VALIDATION;

    return dst;
}