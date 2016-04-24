//
//  biometrics_types.hpp
//  biometrics-04
//
//  Created by Martin Kiesel on 20/04/16.
//  Copyright © 2016 Martin Kiesel. All rights reserved.
//

#ifndef biometrics_types_h
#define biometrics_types_h

#include <map>
#include <vector>
#include <string>

#include <opencv2/core/mat.hpp>

namespace biometrics_4
{
    namespace types
    {
        typedef uint_fast32_t uintf;
        typedef uint_fast16_t uintf16;

        typedef std::vector<double> DoubleDimension;
        typedef std::vector<DoubleDimension> DoubleMatrix;

        typedef std::vector<uintf> UintfDimension;
        typedef std::vector<std::string> StringDimension;

        typedef std::vector<cv::Mat> MatDimension;
        typedef std::vector<MatDimension> VecOfMatDimension;

        struct PcaData
        {
            cv::Mat             mean,
                                eigenvalues,
                                eigenvectors;
            
            MatDimension        train_projections;
            VecOfMatDimension   test_projections;
            UintfDimension      labels;

            void reset()
            {
                mean.release();
                eigenvalues.release();
                eigenvectors.release();
                train_projections.clear();
                test_projections.clear();
                labels.clear();
            }
        private:
            PcaData& operator=(const PcaData&);
        };
    }
}

#endif /* biometrics_types_h */
