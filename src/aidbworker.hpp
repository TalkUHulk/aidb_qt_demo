//
// Created by TalkUHulk on 2023/7/13.
//

#ifndef AIDB_QT_DEMO_AIDBWORKER_HPP
#define AIDB_QT_DEMO_AIDBWORKER_HPP

#include <QThread>
#include <QObject>
#include <iostream>
#include <deque>
#include "Interpreter.h"
#include <string>
#include <opencv2/opencv.hpp>
#include "aidbbin.hpp"
#include "aidbqueue.hpp"
#include "aidbmap.hpp"
#include "utils.hpp"

class AiDBWorker: public QObject {
Q_OBJECT
public:

    AiDBWorker(QObject *parent = nullptr);
    ~AiDBWorker();
//    int push(const cv::Mat &);
//    int get(cv::Mat &);
    void stop();
    void link(AiDBMap* ptr);
public slots:
    void forward();
Q_SIGNALS:
    void finish(const std::shared_ptr<AiDBBin>);
public:
//    std::deque<cv::Mat> _input_queue;
//    AiDBQueue<cv::Mat> _frame_queue;
    AiDBQueue<MatPlus> _frame_queue;
    AiDBMap *_ptr_map_ins{};
    bool running=false;

    static QStringList _modelList;

    static QStringList _backendList;
};


#endif //AIDB_QT_DEMO_AIDBWORKER_HPP
