#include <iostream>
#include <cassert>
#include <dlib/clustering.h>
#include <dlib/graph_utils/sample_pair.h>
#include "cluster.hpp"
#include "math_utils.hpp"

using namespace dlib;
using namespace std;

float cosine_similarity(std::vector<float>& a, std::vector<float>& b)
{
    assert(a.size() == b.size());
    float similarity =0.0;
    for (size_t i =0; i < a.size(); i++) {
        similarity += a[i]*b[i];
    }
    // cout<<"similarity: "<<similarity<<endl;
    return similarity;
}

std::pair<unsigned long, std::vector<unsigned long>> _cluster(std::vector<std::vector<float>>& descriptors, float thresh) {
    // In particular, one simple thing we can do is face clustering.  This next bit of code
    // creates a graph of connected faces and then uses the Chinese whispers graph clustering
    // algorithm to identify how many people there are and which faces belong to whom.
    std::vector<sample_pair> edges;
    for (size_t i = 0; i < descriptors.size(); ++i) {
        for (size_t j = i+1; j < descriptors.size(); ++j) {
            // Faces are connected in the graph if they are close enough.  Here we check if
            // the distance between two face descriptors is less than 0.6, which is the
            // decision threshold the network was trained to use.  Although you can
            // certainly use any other threshold you find useful.
            if (cosine_similarity(descriptors[i], descriptors[j]) > thresh)
                edges.push_back(sample_pair(i,j));
        }
    }
    std::vector<unsigned long> labels;
    unsigned long num_clusters = chinese_whispers(edges, labels);
    // This will correctly indicate that there are 4 people in the image.
    cout << "number of people found in the image: "<< num_clusters << endl;

    return std::pair<unsigned long, std::vector<unsigned long>>(num_clusters, labels);
}

FaceCluster::FaceCluster(float thresh)
{
    thresh_ = thresh;
    labels_num_ = 0;
    preds_num_ = 0;
    ri_ = 0.0;
    precision_ = 0.0;
    recall_ = 0.0;
}

std::pair<unsigned long, std::vector<unsigned long>> FaceCluster::Cluster(std :: vector < std :: vector < float > > & descriptors)
{
    return _cluster(descriptors, thresh_);
}
std::pair<unsigned long, std::vector<unsigned long>> FaceCluster::Cluster(std :: vector < std :: vector<float> > & descriptors,
                                                                                               std::pair<unsigned long, std :: vector <unsigned long >> & labels)
{
    labels_num_ = labels.first;
    labels_.assign(labels.second.begin(), labels.second.end());
    std::pair<unsigned long, std::vector<unsigned long>> preds = _cluster(descriptors, thresh_);
    preds_num_ = preds.first;
    preds_.assign(preds.second.begin(), preds.second.end());
    return preds;
}

std::vector<float> FaceCluster::Metric(void)
{
    size_t N = labels_.size();
    float TP = 0.0;
    float TN = 0.0;
    for (size_t i = 0; i < N; i++) {
        for (size_t j = i+1; j < N; j++) {
            if (labels_[i] ==  labels_[j] && preds_[i] == preds_[j]) {
                TP += 1;
            } else if (labels_[i] != labels_[j] && preds_[i] != preds_[j]) {
                TN += 1;
            }
        }
    }
    ri_ = (TP + TN) / (N*(N-1)/2);

    float TP_FN = 0.0;
    for (unsigned long i = 0; i < labels_num_; i++) {
        long n = std::count(labels_.begin(), labels_.end(), i);
        cout<<"recall Comb("<<n<<",2): "<<combination(n, 2)<<endl;
        TP_FN += combination(n, 2);
    }
    cout<<"TP_FN: "<<TP_FN<<endl;
    recall_ = TP / TP_FN;

    float TP_FP = 0.0;
    for (unsigned long i = 0; i < preds_num_; i++) {
        long n = std::count(preds_.begin(), preds_.end(), i);
        cout<<"precision Comb("<<n<<",2): "<<combination(n, 2)<<endl;
        TP_FP += combination(n, 2);
    }
    cout<<"TP_FP: "<<TP_FP<<endl;
    precision_ = TP / TP_FP;

    std::vector<float> results;
    results.push_back(ri_);
    results.push_back(precision_);
    results.push_back(recall_);
    return results;
}