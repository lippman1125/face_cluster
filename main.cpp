#include <fstream>
#include <iostream>
#include <algorithm>
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <ctime>
#include <cmath>
#include <dirent.h>
#include <string.h>
#include <strings.h>
#include <iomanip>
#include "feature.hpp"
#include "cluster.hpp"

using namespace std;
using namespace cv;
void split(std::string& s, std::string& delim,std::vector< std::string >& ret) {
    size_t last = 0;
    size_t index=s.find_first_of(delim,last);
    while (index!=std::string::npos) {
        //skip continous space
        if (index > last) {
            ret.push_back(s.substr(last,index-last));
        }
        last=index+1;
        index=s.find_first_of(delim,last);
    }

    if (index-last>0) {
        ret.push_back(s.substr(last,index-last));
    }
}

void search_files(string root, vector<string> &files){
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir (root.c_str())) != NULL) {
        while ((ent = readdir (dir)) != NULL) {
            if(!strcmp(ent->d_name, ".") || !strcmp(ent->d_name, ".."))
                continue;
            files.push_back(root + "/" + string(ent->d_name));

        }
        closedir (dir);
    }
}

int get_images(string &root, std::ifstream &facedir_file, vector<string>& imgs_path) {
    string line;
    string delim=" ";
    vector<string> dirs;
    imgs_path.clear();
    if (std::getline(facedir_file, line)) {
        split(line, delim, dirs);
        // cout <<"dirs size: "<<dirs.size()<<endl;
        for (vector<string>::iterator iter = dirs.begin(); iter != dirs.end(); iter++) {
            // cout<<"sub dir path: " << root + "/" + *iter << endl;
            search_files(root + "/" + *iter, imgs_path);
        }
        return true;
    } else {
        return false;
    }
}

int get_images(string &root, std::ifstream &facedir_file, pair<unsigned long, vector<unsigned long>>&labels, vector<string>& imgs_path) {
    string line;
    string delim=" ";
    vector<string> dirs;
    int prev_size = 0;
    int idx = 0;
    labels.second.clear();
    imgs_path.clear();
    if (std::getline(facedir_file, line)) {
        split(line, delim, dirs);
        // cout <<"dirs size: "<<dirs.size()<<endl;
        labels.first = dirs.size();
        for (vector<string>::iterator iter = dirs.begin(); iter != dirs.end(); iter++) {
            // cout<<"sub dir path: " << root + "/" + *iter << endl;
            search_files(root + "/" + *iter, imgs_path);
            for (size_t i = 0; i < imgs_path.size() - prev_size; i++) {
                labels.second.push_back(idx);
            }
            idx++;
            prev_size = imgs_path.size();
        }
        return true;
    } else {
        return false;
    }
}

int main(int argc, char * * argv) {
    if (argc != 3) {
        cout <<"face_cluster facedir_list.txt root_dir" << endl;
        return 0;
    }
    std::srand (unsigned(std::time(0)));
    ifstream facedir_list(argv[1]);
    if (!facedir_list) {
       cout<<"file :"<<argv[1] << " doesn't exit"<<endl;
       return 0;
    }
    string prototxt = "/home/lqy/workshop/image_utils/face_recognition/mnasnet/mnas0.5-long-softmax-retina.prototxt";
    string model = "/home/lqy/workshop/image_utils/face_recognition/mnasnet/mnas0.5-long-combine_merge_retina-153-lfw0.996233.caffemodel";
    class FeatureExtraction feature_extraction(prototxt, model, 128, 128, true);
    class FaceCluster face_cluster(0.35);
    vector<string> imgs_path;
    vector<float> descriptor;
    bool metric_flag = true;
    int lines= 0;

    #ifdef TEST
    cv::Mat img = cv::imread("01/image_0001.jpg");
    cout.width(7);
    if (img.data != nullptr) {
        descriptor = feature_extraction.Extract(img);
        for (size_t i = 0; i < descriptor.size(); i++) {
            cout<<setiosflags(ios::fixed)<<setprecision(6)<<descriptor[i]<<endl;
        }
    }
    return 0;
    #endif

     vector<vector<float>> face_descriptors;
     pair<unsigned long, vector<unsigned long>> labels;
     pair<unsigned long, vector<unsigned long>> preds;
     vector<float> metric_results;
     float RI_average = 0.0;
     float Precision_average = 0.0;
     float Recall_average = 0.0;
     string root(argv[2]);

    if (metric_flag) {
        while(get_images(root, facedir_list, labels, imgs_path)) {
            lines++;
            face_descriptors.clear();
            cout <<"@line: "<<lines<<"______"<<endl;
            cout <<"imgs num: "<<imgs_path.size()<<endl;
            cout <<"labels num: "<<labels.second.size()<<endl;
            assert(imgs_path.size() == labels.second.size());
            for (size_t i = 0; i < imgs_path.size(); i++) {
                // cout<<"img path: "<<imgs_path[i]<<endl;
                cv::Mat img = cv::imread(imgs_path[i]);
                if (img.data != nullptr) {
                    face_descriptors.emplace_back(feature_extraction.Extract(img));
#if 0
                    for (size_t i = 0; i < descriptor.size(); i++) {
                        cout<<descriptor[i]<<endl;
                    }
#endif
                }
            }
            cout<<"real cluster num: "<<labels.first<<endl;
            preds = face_cluster.Cluster(face_descriptors, labels);
            cout<<"pred cluster num: "<<preds.first<<endl;
            metric_results = face_cluster.Metric();
            cout<<"RI: "<<metric_results[0]<<endl;
            cout<<"Precision: "<<metric_results[1]<<endl;
            cout<<"Recall: "<<metric_results[2]<<endl;

            RI_average += metric_results[0];
            Precision_average += metric_results[1];
            Recall_average += metric_results[2];
            cout<<endl<<endl;
        }
        if (lines) {
            RI_average /=lines;
            Precision_average /=lines;
            Recall_average /=lines;
        }
        cout<<"Average:"<<endl;
        cout<<"RI average: "<<RI_average<<endl;
        cout<<"Precision average: "<<Precision_average<<endl;
        cout<<"Recall average: "<<Recall_average<<endl;
    }else {
        while(get_images(root, facedir_list, imgs_path)) {
            face_descriptors.clear();
            for (size_t i = 0; i < imgs_path.size(); i++) {
                cout<<"img path: "<<imgs_path[i]<<endl;
                cv::Mat img = cv::imread(imgs_path[i]);
                if (img.data != nullptr) {
                    face_descriptors.emplace_back(feature_extraction.Extract(img));
#if 0
                    for (size_t i = 0; i < descriptor.size(); i++) {
                        cout<<descriptor[i]<<endl;
                    }
#endif
                }
            }
            labels = _cluster(face_descriptors, 0.35);
            cout<<"cluster num: "<<labels.first<<endl;
            for (size_t i = 0; i < labels.second.size(); i++) {
                cout<<labels.second[i]<<" ";
                if ((i+1)%8==0) {
                    cout<<endl;
                }
            }
            #if 0
            std::random_shuffle (imgs_path.begin(), imgs_path.end());
            cout << "shuffle___"<<endl;
            for (size_t i = 0; i < imgs_path.size(); i++) {
                cout<<"img path: "<<imgs_path[i]<<endl;
            }
            #endif
            break;
        }
    }
    return 0;
}
