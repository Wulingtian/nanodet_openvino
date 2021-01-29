#include "detector.h"

Detector::Detector(){}

Detector::~Detector(){}

std::vector<float> Detector::prepareImage(cv::Mat &src_img){
//    std::cout<< "src_img.rows : " << src_img.rows<< std::endl;
//    std::cout<< "src_img.cols : " << src_img.cols<< std::endl;
//    std::cout<< "INPUT_W : " << INPUT_W<< std::endl;
//    std::cout<< "INPUT_H : " << INPUT_H<< std::endl;

    std::vector<float> result(INPUT_W * INPUT_H * 3);
    float *data = result.data();
    float ratio = float(INPUT_W) / float(src_img.cols) < float(INPUT_H) / float(src_img.rows) ? float(INPUT_W) / float(src_img.cols) : float(INPUT_H) / float(src_img.rows);
    cv::Mat flt_img = cv::Mat::zeros(cv::Size(INPUT_W, INPUT_H), CV_8UC3);
    cv::Mat rsz_img = cv::Mat::zeros(cv::Size(src_img.cols*ratio, src_img.rows*ratio), CV_8UC3);
    //auto pr_start = std::chrono::high_resolution_clock::now();
    cv::resize(src_img, rsz_img, cv::Size(), ratio, ratio);
    //cudaResize(src_img, rsz_img);
    //auto pr_end = std::chrono::high_resolution_clock::now();
    //auto po_ms = std::chrono::duration<float, std::milli>(pr_end - pr_start).count();
    rsz_img.copyTo(flt_img(cv::Rect(0, 0, rsz_img.cols, rsz_img.rows)));
    flt_img.convertTo(flt_img, CV_32FC3);

    //auto pr_end = std::chrono::high_resolution_clock::now();
    //std::cout << "********** " << po_ms << " ms." << "********** " << std::endl;

    //HWC TO CHW
    //auto pr_start = std::chrono::high_resolution_clock::now();
    int channelLength = INPUT_W * INPUT_H;
    std::vector<cv::Mat> split_img = {
            cv::Mat(INPUT_W, INPUT_H, CV_32FC1, data + channelLength * 2),
            cv::Mat(INPUT_W, INPUT_H, CV_32FC1, data + channelLength),
            cv::Mat(INPUT_W, INPUT_H, CV_32FC1, data)
    };
    //auto pr_end = std::chrono::high_resolution_clock::now();
    //auto po_ms = std::chrono::duration<float, std::milli>(pr_end - pr_start).count();
    //std::cout << "********** " << po_ms << " ms." << "********** " << std::endl;

    auto pr_start = std::chrono::high_resolution_clock::now();
    cv::split(flt_img, split_img);
    for (int i = 0; i < 3; i++) {
//        std::cout<< split_img[i].size << std::endl;
//        std::cout<< img_mean[i] << std::endl;
//        std::cout<< img_std[i] << std::endl;

        split_img[i] = (split_img[i] - img_mean[i]) / img_std[i];
    }

    auto pr_end = std::chrono::high_resolution_clock::now();

    auto po_ms = std::chrono::duration<float, std::milli>(pr_end - pr_start).count();
//    std::cout << "********** " << po_ms << " ms." << "********** " << std::endl;
    return result;
}
void Detector::GenerateReferMatrix() {
    int index = 0;
    refer_matrix = cv::Mat(refer_rows, refer_cols, CV_32FC1);
//    std::cout << "refer_matrix.size : " << refer_matrix.size << std::endl;
//    std::cout << "strides : " << strides[0] << " " << strides[1] << " " << strides[2] << std::endl;
//    std::cout << "INPUT_H : " << INPUT_H << std::endl;
//    std::cout << "INPUT_W : " << INPUT_W << std::endl;
    for (const int &stride : strides) {
        for (int h = 0; h < INPUT_H / stride; h++)
            for (int w = 0; w < INPUT_W / stride; w++) {
                auto *row = refer_matrix.ptr<float>(index);
                row[0] = float((2 * w + 1) * stride - 1) / 2;
                row[1] = float((2 * h + 1) * stride - 1) / 2;
                row[2] = stride;
                index += 1;
            }
    }
    //std::cout << "#####################################3" << std::endl;
}

//初始化
bool Detector::init(string xml_path,double cof_threshold,double nms_area_threshold,int input_w, int input_h, int num_class, int r_rows, int r_cols, std::vector<int> s, std::vector<float> i_mean,std::vector<float> i_std){
    _xml_path = xml_path;
    _cof_threshold = cof_threshold;
    _nms_area_threshold = nms_area_threshold;
    INPUT_W = input_w;
    INPUT_H = input_h;
    NUM_CLASS = num_class;
    refer_rows = r_rows;
    refer_cols = r_cols;
    strides = s;
    img_mean = i_mean;
    img_std = i_std;
    Core ie;
    auto cnnNetwork = ie.ReadNetwork(_xml_path); 
    //输入设置
    InputsDataMap inputInfo(cnnNetwork.getInputsInfo());
    InputInfo::Ptr& input = inputInfo.begin()->second;
    _input_name = inputInfo.begin()->first;
    input->setPrecision(Precision::FP32);
    input->getInputData()->setLayout(Layout::NCHW);
    ICNNNetwork::InputShapes inputShapes = cnnNetwork.getInputShapes();
    SizeVector& inSizeVector = inputShapes.begin()->second;
    cnnNetwork.reshape(inputShapes);
    //输出设置
    _outputinfo = OutputsDataMap(cnnNetwork.getOutputsInfo());
    for (auto &output : _outputinfo) {
        output.second->setPrecision(Precision::FP32);
    }
    //获取可执行网络
    //_network =  ie.LoadNetwork(cnnNetwork, "GPU");
    _network =  ie.LoadNetwork(cnnNetwork, "CPU");
    return true;
}

//释放资源
bool Detector::uninit(){
    return true;
}

//处理图像获取结果
vector<Detector::Bbox> Detector::process_frame(Mat& inframe){

    cv::Mat showImage = inframe.clone();
    std::vector<float> pr_img = prepareImage(inframe);
    InferRequest::Ptr infer_request = _network.CreateInferRequestPtr();
    Blob::Ptr frameBlob = infer_request->GetBlob(_input_name);
    InferenceEngine::LockedMemory<void> blobMapped = InferenceEngine::as<InferenceEngine::MemoryBlob>(frameBlob)->wmap();
    float* blob_data = blobMapped.as<float*>();

    memcpy(blob_data, pr_img.data(), 3 * INPUT_H * INPUT_W * sizeof(float));

    //执行预测
    infer_request->Infer();
    //获取各层结果
    vector<Rect> origin_rect;
    vector<float> origin_rect_cof;
    int i=0;
    vector<Bbox> bboxes;
    for (auto &output : _outputinfo) {
        auto output_name = output.first;
        Blob::Ptr blob = infer_request->GetBlob(output_name);
        LockedMemory<const void> blobMapped = as<MemoryBlob>(blob)->rmap();
        float *output_blob = blobMapped.as<float *>();
        bboxes = postProcess(showImage,output_blob);
        ++i;
    }
    return bboxes;
}

float Detector::IOUCalculate(const Detector::Bbox &det_a, const Detector::Bbox &det_b) {
    cv::Point2f center_a(det_a.x, det_a.y);
    cv::Point2f center_b(det_b.x, det_b.y);
    cv::Point2f left_up(std::min(det_a.x - det_a.w / 2, det_b.x - det_b.w / 2),
                        std::min(det_a.y - det_a.h / 2, det_b.y - det_b.h / 2));
    cv::Point2f right_down(std::max(det_a.x + det_a.w / 2, det_b.x + det_b.w / 2),
                           std::max(det_a.y + det_a.h / 2, det_b.y + det_b.h / 2));
    float distance_d = (center_a - center_b).x * (center_a - center_b).x + (center_a - center_b).y * (center_a - center_b).y;
    float distance_c = (left_up - right_down).x * (left_up - right_down).x + (left_up - right_down).y * (left_up - right_down).y;
    float inter_l = det_a.x - det_a.w / 2 > det_b.x - det_b.w / 2 ? det_a.x - det_a.w / 2 : det_b.x - det_b.w / 2;
    float inter_t = det_a.y - det_a.h / 2 > det_b.y - det_b.h / 2 ? det_a.y - det_a.h / 2 : det_b.y - det_b.h / 2;
    float inter_r = det_a.x + det_a.w / 2 < det_b.x + det_b.w / 2 ? det_a.x + det_a.w / 2 : det_b.x + det_b.w / 2;
    float inter_b = det_a.y + det_a.h / 2 < det_b.y + det_b.h / 2 ? det_a.y + det_a.h / 2 : det_b.y + det_b.h / 2;
    if (inter_b < inter_t || inter_r < inter_l)
        return 0;
    float inter_area = (inter_b - inter_t) * (inter_r - inter_l);
    float union_area = det_a.w * det_a.h + det_b.w * det_b.h - inter_area;
    if (union_area == 0)
        return 0;
    else
        return inter_area / union_area - distance_d / distance_c;
}
void Detector::NmsDetect(std::vector<Detector::Bbox> &detections) {
    sort(detections.begin(), detections.end(), [=](const Detector::Bbox &left, const Detector::Bbox &right) {
        return left.prob > right.prob;
    });

    for (int i = 0; i < (int)detections.size(); i++)
        for (int j = i + 1; j < (int)detections.size(); j++)
        {
            float iou = IOUCalculate(detections[i], detections[j]);
            if (iou > _nms_area_threshold)
                detections[j].prob = 0;
        }

    detections.erase(std::remove_if(detections.begin(), detections.end(), [](const Detector::Bbox &det)
    { return det.prob == 0; }), detections.end());
}

std::vector<Detector::Bbox> Detector::postProcess(const cv::Mat &src_img,
                              float *output) {
    GenerateReferMatrix();
    std::vector<Detector::Bbox> result;
    float *out = output;
    float ratio = std::max(float(src_img.cols) / float(INPUT_W), float(src_img.rows) / float(INPUT_H));
    cv::Mat result_matrix = cv::Mat(refer_rows, NUM_CLASS + 4, CV_32FC1, out);
    for (int row_num = 0; row_num < refer_rows; row_num++) {
        Detector::Bbox box;
        auto *row = result_matrix.ptr<float>(row_num);
        auto max_pos = std::max_element(row + 4, row + NUM_CLASS + 4);
        box.prob = row[max_pos - row];
        if (box.prob < _cof_threshold)
            continue;
        box.classes = max_pos - row - 4;
        auto *anchor = refer_matrix.ptr<float>(row_num);
        box.x = (anchor[0] - row[0] * anchor[2] + anchor[0] + row[2] * anchor[2]) / 2 * ratio;
        box.y = (anchor[1] - row[1] * anchor[2] + anchor[1] + row[3] * anchor[2]) / 2 * ratio;
        box.w = (row[2] + row[0]) * anchor[2] * ratio;
        box.h = (row[3] + row[1]) * anchor[2] * ratio;
        result.push_back(box);
    }
    //std::cout<< "#########################" << std::endl;
    NmsDetect(result);
    return result;
}



