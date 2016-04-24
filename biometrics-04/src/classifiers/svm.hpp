//
//  svm.hpp
//  biometrics-04
//
//  Created by Martin Kiesel on 23/04/16.
//  Copyright © 2016 Martin Kiesel. All rights reserved.
//

#ifndef svm_hpp
#define svm_hpp

#include "biometrics_types.hpp"

namespace biometrics_4
{
    namespace classifier
    {
        namespace svm
        {
            class Svm
            {
                const types::PcaData *_data;

            public:
                Svm(const types::PcaData *data) : _data(data) {}
                double classify();
            };
        }
    }
}

#endif /* svm_hpp */