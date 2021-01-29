cmake_minimum_required(VERSION 3.5)
project(detect_test)
set(CMAKE_CXX_STANDARD 11)
include_directories(
    #OpenVINO推理引擎的头文件
    /opt/intel/openvino/deployment_tools/inference_engine/include/
    /opt/intel/openvino/deployment_tools/ngraph/include/
)

#查找必要的依赖包
set(OpenCV_INCLUDE_DIRS /home/willer/temp/opencv-3.4.2/include)
set(OpenCV_LIBS_DIR /home/willer/temp/opencv-3.4.2/build/lib)
set(OpenCV_LIBS opencv_core opencv_imgproc opencv_imgcodecs opencv_highgui)

#set(OpenCV_INCLUDE_DIRS /home/willer/TestMyTrtEngine-trt721/include)
#set(OpenCV_LIBS_DIR /home/willer/TestMyTrtEngine-trt721/lib/opencv)
#set(OpenCV_LIBS opencv_world )


include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBS_DIR})

set(InferenceEngine_DIR "/opt/intel/openvino/deployment_tools/inference_engine/share")
find_package(InferenceEngine)
set(ngraph_DIR "/opt/intel/openvino/deployment_tools/ngraph")
find_package(ngraph REQUIRED)
set(ngraph_LIBRARIES "/opt/intel/openvino/deployment_tools/ngraph/lib/libngraph.so")
set(ngraph_INCLUDE_DIRS "/opt/intel/openvino/deployment_tools/ngraph/include/")


# 编译detector.so
add_library (detector SHARED ${CMAKE_CURRENT_SOURCE_DIR}/detector.cpp)
target_include_directories(detector
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
        PUBLIC ${OpenCV_INCLUDE_DIRS}
        PUBLIC ${InferenceEngine_INCLUDE_DIRS}
#        PUBLIC ${ngraph_INCLUDE_DIRS}
)
target_link_libraries(detector
        ${OpenCV_LIBS}
        ${InferenceEngine_LIBRARIES}
        ${ngraph_LIBRARIES}
)

add_executable(detect_test main.cpp)
target_link_libraries(detect_test
                                                                                                                                                                ${OpenCV_LIBS}
        detector
)