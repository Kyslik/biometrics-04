//
//  main.cpp
//  biometrics-04
//
//  Created by Martin Kiesel on 18/04/16.
//  Copyright © 2016 Martin Kiesel. All rights reserved.
//

#include <iostream>
#include "pca.hpp"

const biometrics_4::types::uintf TRAIN_SIZE = 10;
const biometrics_4::types::uintf TEST_SIZE = 2;

int main(int argc, const char * argv[])
{
    biometrics_4::pca::Pca p(TRAIN_SIZE, TEST_SIZE);

    return 0;
}
