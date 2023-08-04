//
// Created by TalkUHulk on 2023/7/13.
//

#include "aidbworker.hpp"
#include "QDebug"
#include <iostream>
#include "Interpreter.h"
#include "Utility.h"
#include "face_align.h"
#include "td_obj.h"
#include <QColorDialog>
#include <chrono>


QStringList AiDBWorker::_modelList{"scrfd_500m_kps", "pfpld", "3ddfa_mb05_bfm_dense","bisenet","movenet",
                                    "yolox_nano","yolov7_tiny","yolov8n",
                                    "ppocr_det", "ppocr_cls", "ppocr_ret", "mobilevit_xxs"};

QStringList AiDBWorker::_backendList{"ONNX", "MNN", "NCNN", "TNN", "PaddleLite", "OpenVINO"};


AiDBWorker::AiDBWorker(QObject * parent)
        : QObject(parent)
{
    running = true;
}
AiDBWorker::~AiDBWorker(){

}

void AiDBWorker::forward()
{

    while(running){
        cv::Mat frame;

//        std::cout << "running:" << _frame_queue.size() << ";" << _ptr_map_ins << "\n";
        if(_frame_queue.empty() || _ptr_map_ins->empty())
            continue;

//        std::cout << "@@@@" << _frame_queue.size() << std::endl;
        if(_frame_queue.size() > 1){
            int cnt = _frame_queue.size() - 1;
            while(cnt--){
                _frame_queue.pop();
            }
        }
        _frame_queue.pop(frame);
        std::shared_ptr<AiDBBin> bin = std::make_shared<AiDBBin>();

        std::chrono::time_point<std::chrono::system_clock> tic = std::chrono::system_clock::now();
        _ptr_map_ins->busy(true);

//        for(int i = 0; i < _ptr_deque_ins->size(); i++){
        for(auto& model_name: _modelList){

            auto ins = (Interpreter*)(*_ptr_map_ins)[model_name.toStdString()];
            if(ins == nullptr)
                continue;

            if(ins->which_model().find("SCRFD") != std::string::npos){
                std::vector<std::vector<float>> outputs;
                std::vector<std::vector<int>> outputs_shape;
                cv::Mat blob = *ins << frame;
                ins->forward((float*)blob.data, ins->width(),ins->height(), ins->channel(), outputs, outputs_shape);
                AIDB::Utility::scrfd_post_process(outputs, bin->face_meta, ins->width(), ins->height(), ins->scale_h());

            } else if(ins->which_model().find("PFPLD") != std::string::npos){
                if(!bin->face_meta.empty()){
                    for(auto &face_meta: bin->face_meta){
                        std::vector<std::vector<float>> outputs;
                        std::vector<std::vector<int>> outputs_shape;

                        std::shared_ptr<AIDB::FaceMeta> face_meta_roi = std::make_shared<AIDB::FaceMeta>();
                        AIDB::Utility::Common::parse_roi_from_bbox(face_meta, face_meta_roi, frame.cols, frame.rows, 1.28, 0.14);
                        cv::Mat roi(frame, cv::Rect(face_meta_roi->x1, face_meta_roi->y1, face_meta_roi->width(), face_meta_roi->height()));
                        cv::Mat blob = *ins << roi;
                        ins->forward((float*)blob.data, ins->width(),ins->height(), ins->channel(), outputs, outputs_shape);
                        AIDB::Utility::pfpld_post_process(outputs, face_meta_roi, face_meta, 98);
                    }

                }
                else{
                    qDebug() << "no face detect!" << endl;
                }
            } else if(ins->which_model().find("YOLOX") != std::string::npos){
                std::vector<std::vector<float>> outputs;
                std::vector<std::vector<int>> outputs_shape;

                cv::Mat blob = *ins << frame;

                cv::Mat src_image;
                *ins >> src_image;

                ins->forward((float*)blob.data, ins->width(), ins->height(), ins->channel(), outputs, outputs_shape);

                auto post_process = AIDB::Utility::YoloX(ins->width(), 0.25, 0.45, {8, 16, 32});

                post_process(outputs[0], outputs_shape[0], bin->object_meta, src_image.cols, src_image.rows, ins->scale_h());

            } else if(ins->which_model().find("YOLOV7") != std::string::npos){
                cv::Mat blob = *ins << frame;

                cv::Mat src_image;
                *ins >> src_image;

                std::vector<std::vector<float>> outputs;
                std::vector<std::vector<int>> outputs_shape;

                ins->forward((float*)blob.data, ins->width(), ins->height(), ins->channel(), outputs, outputs_shape);

                if(ins->which_model().find("GRID") != std::string::npos){
                    AIDB::Utility::yolov7_post_process(outputs[0], outputs_shape[0], bin->object_meta, 0.45, 0.25, ins->scale_h());
                } else{
                    AIDB::Utility::yolov7_post_process(outputs, outputs_shape, bin->object_meta, 0.45, 0.25, ins->scale_h());
                }

            } else if(ins->which_model().find("YOLOV8") != std::string::npos){
                cv::Mat blob = *ins << frame;

                cv::Mat src_image;
                *ins >> src_image;

                std::vector<std::vector<float>> outputs;
                std::vector<std::vector<int>> outputs_shape;

                ins->forward((float*)blob.data, ins->width(), ins->height(), ins->channel(), outputs, outputs_shape);

                AIDB::Utility::yolov8_post_process(outputs[0], outputs_shape[0], bin->object_meta, 0.45, 0.35, ins->scale_h());

            } else if(ins->which_model().find("MOVENET") != std::string::npos){
                cv::Mat blob = *ins << frame;

                cv::Mat src_image;
                *ins >> src_image;

                std::vector<std::vector<float>> outputs;
                std::vector<std::vector<int>> outputs_shape;

                ins->forward((float*)blob.data, ins->width(), ins->height(), ins->channel(), outputs, outputs_shape);
                cv::Mat result;
                AIDB::Utility::movenet_post_process(src_image, outputs, outputs_shape, bin->human_keypoints);

            } else if(ins->which_model().find("MOBILEVIT") != std::string::npos){
                static auto post_process = AIDB::Utility::ImageNet("extra/imagenet-1k-id.txt");

                cv::Mat blob = *ins << frame;

                cv::Mat src_image;
                *ins >> src_image;

                std::vector<std::vector<float>> outputs;
                std::vector<std::vector<int>> outputs_shape;

                ins->forward((float*)blob.data, ins->width(), ins->height(), ins->channel(), outputs, outputs_shape);
                int topK = 3;
                post_process(outputs[0], outputs_shape[0], bin->cls_meta, topK);


            } else if(ins->which_model().find("TDDFAV2") != std::string::npos){
                if(!bin->face_meta.empty()){
                    auto result = frame.clone();
                    for(auto &face_meta: bin->face_meta){
                        std::shared_ptr<AIDB::FaceMeta> face_meta_roi = std::make_shared<AIDB::FaceMeta>();
                        AIDB::Utility::Common::parse_roi_from_bbox(face_meta, face_meta_roi, frame.cols, frame.rows, 1.58, 0.14);
                        cv::Mat roi(frame, cv::Rect(face_meta_roi->x1, face_meta_roi->y1, face_meta_roi->width(), face_meta_roi->height()));

                        auto blob = *ins << roi;

                        std::vector<std::vector<float>> outputs;
                        std::vector<std::vector<int>> outputs_shape;

                        ins->forward((float*)blob.data, ins->width(), ins->height(), ins->channel(), outputs, outputs_shape);
                        std::vector<float> vertices, pose, sRt;

                        AIDB::Utility::tddfa_post_process(outputs, outputs_shape, face_meta_roi, vertices, pose, sRt, 120);

                        if(ins->which_model().find("DENSE") != std::string::npos){
                            {
//                                for(int i = 0; i < vertices.size() / 3; i+=6){
//                                    cv::circle(result, cv::Point(vertices[3*i], vertices[3*i+1]), 2, cv::Scalar(255, 255, 255), -1);
//                                }
                                AIDB::Utility::TddfaUtility::tddfa_rasterize(result, vertices, spider_man_obj, 1, true);
//                                AIDB::Utility::TddfaUtility::tddfa_rasterize(result, vertices, hulk_obj, 1, false);
                            }
                        } else{
                            for(int i = 0; i < vertices.size() / 3; i++){
                                cv::circle(result, cv::Point(vertices[3*i], vertices[3*i+1]), 2, cv::Scalar(255, 255, 255), -1);
                            }
                            AIDB::Utility::TddfaUtility::plot_pose_box(result, sRt, vertices, 68);
                        }
                    }

//                    bin->generated.insert(std::make_pair(ins->which_model(), result));
                      bin->generated = result;
                }
                else{
                    qDebug() << "no face detect!" << endl;
                }

            } else if(ins->which_model().find("BISENET") != std::string::npos){
                if(!bin->face_meta.empty()){
                    for(auto &face_meta: bin->face_meta) {
                        std::vector<std::vector<float>> outputs;
                        std::vector<std::vector<int>> outputs_shape;

                        cv::Mat align;
                        AIDB::faceAlign(frame, align, face_meta, ins->width(), "ffhq");

                        auto blob = *ins << align;

                        cv::Mat src_image;
                        *ins >> src_image;

                        ins->forward((float *) blob.data, ins->width(), ins->height(), ins->channel(), outputs,
                                     outputs_shape);

                        cv::Mat result;
                        AIDB::Utility::bisenet_post_process(src_image, result, outputs[0], outputs_shape[0]);
                        bin->face_parsing.emplace_back(result);
                    }
                }
                else{
                    qDebug() << "no face detect!" << endl;
                }

            }
//            else if(ins->which_model().find("ANIME") != std::string::npos){
//                cv::Mat blob = *ins << frame;
//
//                std::vector<std::vector<float>> outputs;
//                std::vector<std::vector<int>> outputs_shape;
//
//                ins->forward((float*)blob.data, ins->width(), ins->height(), ins->channel(), outputs, outputs_shape);
//
//                cv::Mat result;
//                AIDB::Utility::animated_gan_post_process(outputs[0], outputs_shape[0], result);
////                bin->generated.insert(std::make_pair(ins->which_model(), result));
//                bin->generated = result;
//
//            }
            else if(ins->which_model().find("PPOCR") != std::string::npos){
                static auto post_process = AIDB::Utility::PPOCR();

                if(ins->which_model().find("DBNET") != std::string::npos){

                    cv::Mat blob = *ins << frame;

                    std::vector<std::vector<float>> outputs;
                    std::vector<std::vector<int>> outputs_shape;

                    ins->forward((float*)blob.data, ins->width(), ins->height(), ins->channel(), outputs, outputs_shape);

                    post_process.dbnet_post_process(outputs[0], outputs_shape[0], bin->ocr_meta, ins->scale_h(), ins->scale_w(), frame);

                } else if(ins->which_model().find("PPOCR_CLS") != std::string::npos) {

                    if (bin->ocr_meta.empty()) {
                        qDebug() << "no characters detect!" << endl;
                    } else {
                        for (auto &ocr_result :bin->ocr_meta) {

                            cv::Mat crop_img;
                            AIDB::Utility::PPOCR::GetRotateCropImage(frame, crop_img, ocr_result);

                            std::vector<std::vector<float>> outputs;
                            std::vector<std::vector<int>> outputs_shape;

                            cv::Mat cls_blob = *ins << crop_img;
                            ins->forward((float *) cls_blob.data, ins->width(), ins->height(),
                                         ins->channel(), outputs, outputs_shape);

                            post_process.cls_post_process(outputs[0], outputs_shape[0], crop_img, crop_img, ocr_result);

                        }
                    }
                } else if(ins->which_model().find("PPOCR_CRNN") != std::string::npos){
                    if (bin->ocr_meta.empty()) {
                        qDebug() << "no characters detect!" << endl;
                    } else {
                        for (auto &ocr_result :bin->ocr_meta) {

                            cv::Mat crop_img;
                            AIDB::Utility::PPOCR::GetRotateCropImage(frame, crop_img, ocr_result);

                            std::vector<std::vector<float>> outputs;
                            std::vector<std::vector<int>> outputs_shape;

                            cv::Mat crnn_blob = *ins << crop_img;
                            ins->forward((float*)crnn_blob.data, ins->width(), ins->height(), ins->channel(), outputs, outputs_shape);
                            post_process.crnn_post_process(outputs[0], outputs_shape[0], ocr_result);

                        }
                    }
                }


            }

        }
        _ptr_map_ins->busy(false);

        std::chrono::time_point<std::chrono::system_clock> toc = std::chrono::system_clock::now();
        bin->cost_time = (float)(toc - tic).count();

        emit finish(bin);

    }

}


void AiDBWorker::stop() {
    running = false;
}

void AiDBWorker::link(AiDBMap *ptr) {
    _ptr_map_ins = ptr;
}

