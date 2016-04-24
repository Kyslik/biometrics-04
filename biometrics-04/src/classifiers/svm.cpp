//
//  svm.cpp
//  biometrics-04
//
//  Created by Martin Kiesel on 23/04/16.
//  Copyright © 2016 Martin Kiesel. All rights reserved.
//

#include "svm.hpp"
#include <opencv2/ml.hpp>

namespace biometrics_4
{
    namespace classifier
    {
        namespace svm
        {
            using types::uintf;

            double Svm::classify(uintf components)
            {
                if (_data->train_projections.empty()) return -1;
                uintf train_size = static_cast<uintf>(_data->train_projections.size());
                cv::Mat train_data(train_size, components, CV_32FC1);
                cv::Mat label_data(train_size, 1, CV_32SC1);

                for (uintf i = 0; i < train_size; i++)
                {
                    _data->train_projections[i].row(0).copyTo(train_data.row(i));
                    label_data.at<int>(i, 0) = _data->labels[i];
                }

                cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();

                svm->setType(cv::ml::SVM::C_SVC);
                svm->setGamma(0.0001);
                svm->setC(2000);
                svm->setKernel(cv::ml::SVM::RBF);
                svm->train(train_data, cv::ml::ROW_SAMPLE, label_data);

                uintf predictions = 0;
                uintf total_comparsions = 0;

                for (uintf i = 0; i < _data->test_projections.size(); i++)
                {
                    for (const auto &projection : _data->test_projections[i])
                    {
                        total_comparsions++;
                        uintf prediction = svm->predict(projection);
                        if (prediction == i)
                            predictions++;
                    }
                }

                svm.release();
                train_data.release();
                label_data.release();

                return (static_cast<double>(predictions * 100) / total_comparsions);
            }
        }
    }
}