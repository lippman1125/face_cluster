#ifndef _FEATURE_HPP_
#define _FEATURE_HPP_

#include <caffe/caffe.hpp>
#include <memory>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace caffe;
class FeatureExtraction {
 public:
    FeatureExtraction(const std::string& prototxt, const std::string& model,
        float mean, float scale, bool norm);
    std::vector<float> Extract(const cv::Mat& img);

 private:
    void WrapInputLayer(std::vector<cv::Mat>* input_channels);
    void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);
    std::vector<float> PostProcess(void);

 private:
    std::shared_ptr<Net<float> > net_;
    cv::Size input_geometry_;
    int num_channels_;
    float mean_;
    float scale_;
    bool norm_;
};

#endif /*_FEATURE_HPP_*/
