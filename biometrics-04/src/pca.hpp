//
//  pca.hpp
//  biometrics-04
//
//  Created by Martin Kiesel on 20/04/16.
//  Copyright © 2016 Martin Kiesel. All rights reserved.
//

#ifndef pca_hpp
#define pca_hpp

#include "biometrics_types.hpp"
#include "random.hpp"

namespace biometrics_4
{
    namespace pca
    {
        class Pca
        {
            bool _is_valid = false;
            const types::uintf       _data_size,
                                     _test_size,
                                     _train_size;
            types::uintf             _components;
            types::MatDimension      _train_images;
            types::VecOfMatDimension _test_images;
            types::UintfDimension    _labels;
            random::Random           _random;
            types::PcaData           _pca_data;

            const std::string        _folder = "./data/";
            const types::StringDimension _faces =
            {"butler", "obama", "putin",
                "schwarzenegger", "stallone", "willis"};

            bool initialize();
            bool constructPcaData();
            cv::Mat constructMat(const types::MatDimension &images);
            cv::Mat constructMat(const cv::Mat &image);
            cv::Mat constructMat();

            Pca& operator=(const Pca&);
            Pca(const Pca&);
            
        public:
            Pca(const types::uintf train_size, const types::uintf test_size) :
                        _test_size(test_size),
                        _train_size(train_size),
                        _data_size(test_size + train_size),
                        _components(0)
            {
                _is_valid = initialize();
            };

            void randomize();
            inline bool isValid() const { return _is_valid; }
            inline const types::PcaData &getPcaData() { return _pca_data; }
            inline void setComponents(types::uintf num)
            {
                _components = num;
                _is_valid = constructPcaData();
            }
        };
    }
}
#endif /* pca_hpp */
