//
// Created by TalkUHulk on 2023/7/14.
//

#ifndef AIDB_QT_DEMO_AIDBBIN_HPP
#define AIDB_QT_DEMO_AIDBBIN_HPP

#include "AIDBData.h"
#include <vector>
#include <map>
#include <memory>

using namespace AIDB;



class AiDBBin{
public:
    AiDBBin(){
        qRegisterMetaType<std::shared_ptr<AiDBBin>>("std::shared_ptr<AiDBBin>");
    }
    std::vector<std::shared_ptr<FaceMeta>> face_meta;
    std::vector<cv::Mat> face_parsing;
    std::vector<std::vector<float>> human_keypoints;
    std::vector<std::shared_ptr<ObjectMeta>> object_meta;
    std::vector<std::shared_ptr<OcrMeta>> ocr_meta;
    std::vector<std::shared_ptr<ClsMeta>> cls_meta;
    std::vector<float> feat;
    std::vector<float> mask;
//    std::map<std::string, cv::Mat> generated;
    cv::Mat generated;
    cv::Mat cache;
    float cost_time;
};

#endif //AIDB_QT_DEMO_AIDBBIN_HPP
