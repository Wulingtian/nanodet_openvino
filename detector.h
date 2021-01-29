#ifndef DETECTOR_H
#define DETECTOR_H
#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>
#include <iostream>
#include <chrono>
#include <cmath>
using namespace std;
using namespace cv;
using namespace InferenceEngine;



class Detector
{
public:
    typedef struct {
        float prob;
        std::string name;
        cv::Rect rect;
    } Object;

    struct Bbox{
        float x;
        float y;
        float w;
        float h;
        float prob;
        int classes;
    };

    Detector();
    ~Detector();
    //初始化
    bool init(string xml_path,double cof_threshold,double nms_area_threshold,int input_w, int input_h,int num_class, int r_rows, int r_cols, std::vector<int> s, std::vector<float> i_mean, std::vector<float> i_std);
    //释放资源
    bool uninit();
    //处理图像获取结果
    vector<Detector::Bbox> process_frame(Mat& inframe);

private:
    std::vector<float> prepareImage(cv::Mat &src_img);
    void GenerateReferMatrix();
    float IOUCalculate(const Detector::Bbox &det_a, const Detector::Bbox &det_b);
    void NmsDetect(std::vector<Detector::Bbox> &detections);
    std::vector<Detector::Bbox> postProcess(const cv::Mat &src_img,
                                            float *output);
    //存储初始化获得的可执行网络
    ExecutableNetwork _network;
    OutputsDataMap _outputinfo;
    string _input_name;
    //参数区
    string _xml_path;                             //OpenVINO模型xml文件路径
    double _cof_threshold;                //置信度阈值,计算方法是框置信度乘以物品种类置信度
    double _nms_area_threshold;  //nms最小重叠面积阈值
    int INPUT_W;
    int INPUT_H;
    int NUM_CLASS;
    int refer_rows;
    int refer_cols;
    cv::Mat refer_matrix;
    std::vector<int> strides;
    std::vector<float> img_mean;
    std::vector<float> img_std;
};
#endif