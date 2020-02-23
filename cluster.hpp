#ifndef _CLUSTER_HPP_
#define _CLUSTER_HPP_

std::pair<unsigned long, std::vector<unsigned long>> _cluster(std :: vector < std :: vector < float >> & descriptors, float thresh);

class FaceCluster {
 public:
    FaceCluster(float thresh);
    std::pair<unsigned long, std::vector<unsigned long>> Cluster(std::vector<std::vector<float>>& descriptors);
    std::pair<unsigned long, std::vector<unsigned long>> Cluster(std::vector<std::vector<float>>& descriptors,
                                                                                                                          std::pair<unsigned long, std::vector<unsigned long>>& labels);
    std::vector<float> Metric(void);
 private:
    std::vector<unsigned long> labels_;
    std::vector<unsigned long> preds_;
    unsigned long labels_num_;
    unsigned long preds_num_;
    float thresh_;
    float ri_;
    float precision_;
    float recall_;
};

#endif /*_CLUSTER_HPP_*/
