//
//  euclidean_distance.hpp
//  biometrics-04
//
//  Created by Martin Kiesel on 23/04/16.
//  Copyright © 2016 Martin Kiesel. All rights reserved.
//

#ifndef euclidean_distance_hpp
#define euclidean_distance_hpp

#include "biometrics_types.hpp"

namespace biometrics_4
{
    namespace classifier
    {
        namespace euclid
        {
            class EuclidDistance
            {
                const types::PcaData *_data;

                int predict(const cv::Mat &projection);
            public:
                EuclidDistance(const types::PcaData *data) : _data(data) {}
                double classify();
            };
        }
    }
}

#endif /* euclidean_distance_hpp */
