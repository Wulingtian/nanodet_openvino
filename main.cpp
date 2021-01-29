#include "detector.h"

cv::Mat renderBoundingBox(cv::Mat image, const std::vector<Detector::Bbox> &bboxes){
    //float scale = std::min(static_cast<float>(INPUT_W) / static_cast<float>(image.cols), static_cast<float>(INPUT_H) / static_cast<float>(image.rows));
    for(const auto &rect : bboxes)
    {
        cv::Rect rst(rect.x - rect.w / 2, rect.y - rect.h / 2, rect.w, rect.h);
        cv::rectangle(image, rst, cv::Scalar(255, 204,0), 2, cv::LINE_8, 0);
        //cv::rectangle(image, cv::Point(rect.x - rect.w / 2, rect.y - rect.h / 2), cv::Point(rect.x + rect.w / 2, rect.y + rect.h / 2), cv::Scalar(255, 204,0), 3);
        cv::putText(image, std::to_string(rect.prob), cv::Point(rect.x - rect.w / 2, rect.y - rect.h / 2), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,204,255));
    }
    return image;
}


int main(int argc, char const *argv[])
{
    int test_echo = 20;
    Detector* detector = new Detector;
    string xml_path = "../models/nanodet.xml";
    std::vector<int> strides = {8,16,32};
    std::vector<float> img_mean = {103.53, 116.28, 123.675};
    std::vector<float> img_std = { 57.375,  57.12,  58.395 };
    int num_class = 1;
    float cof_threshold = 0.5;
    float nms_area_threshold = 0.5;
    int input_w = 320;
    int input_h = 320;
    int refer_rows = (input_w/strides[0]) * (input_h/strides[0]) + (input_w/strides[1]) * (input_h/strides[1]) + (input_w/strides[2]) * (input_h/strides[2]);
    //std::cout << "refer_rows : " << refer_rows << std::endl;
    int refer_cols = 3;


    detector->init(xml_path,cof_threshold,nms_area_threshold,input_w,input_h,num_class,refer_rows,refer_cols,strides,img_mean,img_std);
    /*
    VideoCapture capture;
    capture.open(0);
    Mat src;
    while(1){
        capture >> src;
        vector<Detector::Object> detected_objects;
    detector->process_frame(src,detected_objects);
    for(int i=0;i<detected_objects.size();++i){
         int xmin = detected_objects[i].rect.x;
        int ymin = detected_objects[i].rect.y;
        int width = detected_objects[i].rect.width;
        int height = detected_objects[i].rect.height;
        Rect rect(xmin, ymin, width, height);//左上坐标（x,y）和矩形的长(x)宽(y)
        cv::rectangle(src, rect, Scalar(255, 0, 0),1, LINE_8,0);
    }
        imshow("cap",src);
        waitKey(1);
    }
    */
    Mat src = imread("../test_imgs/21.jpg");
    Mat osrc = src.clone();
    int total=0;
    vector<Detector::Bbox> bboxes;
    for (int j = 0; j < test_echo; ++j) {
        auto t_start = std::chrono::high_resolution_clock::now();
        bboxes = detector->process_frame(src);

        auto t_end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
        total += ms;
        std::cout << "[ " << j << " ] " << ms << " ms." << std::endl;
    }
    total /= test_echo;
    std::cout << "Average over " << test_echo << " runs is " << total << " ms." << std::endl;
    osrc = renderBoundingBox(osrc,bboxes);
    cv::imwrite("result.jpg", osrc);
}
