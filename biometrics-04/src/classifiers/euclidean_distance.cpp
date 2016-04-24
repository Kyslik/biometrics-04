//
//  euclidean_distance.cpp
//  biometrics-04
//
//  Created by Martin Kiesel on 23/04/16.
//  Copyright © 2016 Martin Kiesel. All rights reserved.
//

#include "euclidean_distance.hpp"

#include <opencv2/core.hpp>

namespace biometrics_4
{
    namespace classifier
    {
        namespace euclid
        {
            using types::uintf;
            
            double EuclidDistance::classify()
            {
                if (_data->train_projections.empty()) return -1;

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

            int EuclidDistance::predict(const cv::Mat &projection)
            {
                double min_dist = DBL_MAX;
                int label = -1;

                for(int i = 0; i < _data->train_projections.size(); i++)
                {
                    double dist = cv::norm(_data->train_projections[i], projection, cv::NORM_L2);
                    if(dist > min_dist) continue;

                    min_dist = dist;
                    label = _data->labels[i];
                }

                return label;
            };
        }
    }
}