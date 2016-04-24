//
//  mahalanobis_distance.cpp
//  biometrics-04
//
//  Created by Martin Kiesel on 24/04/16.
//  Copyright © 2016 Martin Kiesel. All rights reserved.
//

#include "mahalanobis_distance.hpp"

#include <opencv2/core.hpp>

namespace biometrics_4
{
    namespace classifier
    {
        namespace mahalanobis
        {
            using types::uintf;

            void Mahalanobis::constructClassPredictions()
            {
                _class_projections.clear();
                _mean_projections.clear();

                uintf classes = static_cast<uintf>(_data->test_projections.size());
                uintf class_projections = static_cast<uintf>(_data->train_projections.size()) / classes;
                uintf components = _data->eigenvalues.rows;

                for (uintf i = 0; i < classes; i++)
                {
                    _class_projections.push_back(cv::Mat(class_projections, components, CV_32FC1));
                    _mean_projections.push_back(cv::Mat(1, components, CV_32FC1));
                }

                {
                    uintf class_counter = 0;
                    uintf row_counter = 0;

                    for (const auto &projection : _data->train_projections)
                    {
                        projection.row(0).copyTo(_class_projections[class_counter].row(row_counter));
                        if ((row_counter + 1) % class_projections == 0)
                        {
                            cv::reduce(_class_projections[class_counter], _mean_projections[class_counter], 0, CV_REDUCE_AVG);
                            class_counter++;
                            row_counter = 0;
                        }
                        else
                            row_counter++;
                    }
                }
            }

            double Mahalanobis::classify()
            {
                if (_data->train_projections.empty()) return -1;
                constructClassPredictions();

                uintf predictions = 0;
                uintf total_comparsions = 0;

                for (uintf i = 0; i < _data->test_projections.size(); i++)
                {
                    for (const auto &projection : _data->test_projections[i])
                    {
                        total_comparsions++;
                        if (predict(projection) == i)
                            predictions++;
                    }
                }

                return (static_cast<double>(predictions * 100) / total_comparsions);
            }

            int Mahalanobis::predict(const cv::Mat &projection) const
            {
                double min_dist = DBL_MAX;
                int label = -1;

                for(int i = 0; i < _class_projections.size(); i++)
                {
                    double dist = calculate(projection, i);
                    if(dist > min_dist) continue;

                    min_dist = dist;
                    label = i;
                }
                
                return label;
            };

            double Mahalanobis::calculate(const cv::Mat &projection, const uintf class_id) const
            {
                cv::Mat cov, mean;

                cv::calcCovarMatrix(_class_projections[class_id], cov, mean, CV_COVAR_ROWS | CV_COVAR_NORMAL);
                cov = cov / (_class_projections[class_id].rows - 1);

                cv::invert(cov, cov, cv::DECOMP_SVD);

                cov.convertTo(cov, CV_32FC1);

                return cv::Mahalanobis(projection, _mean_projections[class_id], cov);
            }
        }
    }
}