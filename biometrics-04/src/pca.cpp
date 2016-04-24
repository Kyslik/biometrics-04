//
//  pca.cpp
//  biometrics-04
//
//  Created by Martin Kiesel on 20/04/16.
//  Copyright © 2016 Martin Kiesel. All rights reserved.
//

#include "pca.hpp"
#include <opencv2/opencv.hpp>

namespace biometrics_4
{
    namespace pca
    {
        using cv::Mat;
        using types::uintf;
        using types::MatDimension;

        void Pca::randomize()
        {
            _train_images.clear();
            _test_images.clear();
            _is_valid = initialize();
        }

        bool Pca::initialize()
        {
            {
                uintf label = 0;

                for (const auto &name : _faces)
                {
                    uintf   train_size = _train_size,
                            test_size = _test_size;

                    MatDimension test_data;

                    for (types::uintf i = 1; i <= _data_size; i++)
                    {
                        std::string id_padded = std::to_string(i);
                        id_padded = std::string(2 - id_padded.length(), '0') + id_padded;

                        Mat image = cv::imread(_folder + name + "_" + id_padded + ".bmp", 0);

                        if ((train_size > 0 && _random.getBool()) || test_size == 0)
                        {
                            _train_images.push_back(image);
                            _labels.push_back(label);
                            train_size--;
                        }
                        else
                        {
                            test_data.push_back(image);
                            test_size--;
                        }
                    }
                    label++;
                    _test_images.push_back(test_data);
                }
            }
            
            return true;
        }

        bool Pca::constructPcaData()
        {
            Mat data = constructMat();
            uintf rows = data.rows;

            if (rows != _labels.size()) return false;

            if(_components <= 0 || _components > rows)
                _components = rows;

            cv::PCA pca(data, Mat(), cv::PCA::DATA_AS_ROW, static_cast<int>(_components));

            _pca_data.reset();
            _pca_data.mean = pca.mean.clone().reshape(1,1);
            _pca_data.eigenvalues = pca.eigenvalues.clone();
            cv::transpose(pca.eigenvectors, _pca_data.eigenvectors);

            for(uintf i = 0; i < rows; i++)
            {
                Mat projected = pca.project(data.row(i).clone());
                _pca_data.train_projections.push_back(projected);
            }

            for (uintf i = 0; i < _faces.size(); i++)
            {
                MatDimension projections;
                for (const auto &image : _test_images[i])
                {
                    Mat projected = pca.project(constructMat(image));
                    projections.push_back(projected);
                }
                _pca_data.test_projections.push_back(projections);
            }

            _pca_data.labels = _labels;

            return true;
        }

        Mat Pca::constructMat()
        {
            return constructMat(_train_images);
        }

        Mat Pca::constructMat(const MatDimension &images)
        {

            Mat dst(static_cast<uintf>(images.size()), images[0].rows * images[0].cols, CV_32FC1);

            for(uintf i = 0; i < images.size(); i++)
            {
                Mat image_row = images[i].clone().reshape(1, 1);
                Mat row_i = dst.row(i);
                image_row.convertTo(row_i, CV_32FC1, 1/255.);
            }

            return dst;
        }

        Mat Pca::constructMat(const cv::Mat &image)
        {
            return constructMat(types::MatDimension(1, image));
        }
    }
}