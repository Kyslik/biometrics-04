//
//  random.hpp
//  biometrics-04
//
//  Created by Martin Kiesel on 20/04/16.
//  Copyright © 2016 Martin Kiesel. All rights reserved.
//

#ifndef random_h
#define random_h

#include <random>

namespace biometrics_4
{
    namespace random
    {
        struct Random
        {
            inline bool getBool()
            {
                std::mt19937 _generate(_rd());
                std::bernoulli_distribution _distribution(0.5);
                return _distribution(_generate);
            }
        private:
            std::random_device _rd;
        };
    }
}

#endif /* random_h */
