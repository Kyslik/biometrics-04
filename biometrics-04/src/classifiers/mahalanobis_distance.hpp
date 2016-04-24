//
//  mahalanobis_distance.hpp
//  biometrics-04
//
//  Created by Martin Kiesel on 24/04/16.
//  Copyright © 2016 Martin Kiesel. All rights reserved.
//

#ifndef mahalanobis_distance_hpp
#define mahalanobis_distance_hpp

#include "biometrics_types.hpp"

namespace biometrics_4
{
    namespace classifier
    {
        namespace mahalanobis
        {
            class Mahalanobis
            {
                const types::PcaData *_data;
                int predict(const cv::Mat &projection);

            public:
                Mahalanobis(const types::PcaData *data) : _data(data) {}
                double classify();
            };
        }
    }
}

#endif /* mahalanobis_distance_hpp */