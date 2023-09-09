//
// Created by TalkUHulk on 2023/7/22.
//

#ifndef AIDB_QT_DEMO_UTILS_HPP
#define AIDB_QT_DEMO_UTILS_HPP

#include <opencv2/opencv.hpp>
#include <QImage>
#include <QColor>
#include <QRectF>
#include <QPointF>
#include <deque>


typedef struct MatPlus{
    MatPlus()= default;
    explicit MatPlus(cv::Mat &mat): _mat(mat){}
    explicit MatPlus(cv::Mat &mat, std::deque<QPointF> *points): _mat(mat){
        _points_ptr = points;
        _prompt_type = 0;
    }
    explicit MatPlus(cv::Mat &mat, std::deque<QRectF> *rects): _mat(mat){
        _rects_ptr = rects;
        _prompt_type = 1;
    }
    cv::Mat _mat;
    std::deque<QPointF> *_points_ptr = nullptr;
    std::deque<QRectF>* _rects_ptr = nullptr;
    int _prompt_type; // 0: point, 1: box

} MatPlus;

typedef struct ImageRenderParam{
    float _scale;
    int _pad_x;
    int _pad_y;
    int _w;
    int _h;
    int _org_w;
    int _org_h;
} ImageRenderParam;

//typedef struct ModelConfig{
//    int _backend = 0;
//    QColor _color = {255, 255, 255};
//
//} ModelConfig;

static int Mat2Qt(const cv::Mat &src, QImage& dst) {
    if(src.type() == CV_8UC1)
    {
        dst = QImage(src.cols,src.rows,QImage::Format_Indexed8);
        dst.setColorCount(256);
        for(int i = 0; i < 256; i ++)
        {
            dst.setColor(i,qRgb(i,i,i));
        }
        uchar *pSrc = src.data;
        for(int row = 0; row < src.rows; row ++)
        {
            uchar *pDest = dst.scanLine(row);
            memcmp(pDest,pSrc,src.cols);
            pSrc += src.step;
        }
        return 0;
    }
    else if(src.type() == CV_8UC3)
    {
        const auto *pSrc = (const uchar*)src.data;
        dst = QImage(pSrc,src.cols,src.rows,src.step,QImage::Format_RGB888).rgbSwapped();
        return 0;
    }
    else if(src.type() == CV_8UC4)
    {
        const auto *pSrc = (const uchar*)src.data;
        dst = QImage(pSrc, src.cols, src.rows, src.step, QImage::Format_ARGB32);
        return 0;
    }
    else{
        return 1;
    }

}
#endif //AIDB_QT_DEMO_UTILS_HPP
