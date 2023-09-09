//
// Created by TalkUHulk on 2023/7/11.
//

#ifndef AIDB_QT_DEMO_MAINWINDOW_HPP
#define AIDB_QT_DEMO_MAINWINDOW_HPP

#include <QMainWindow>
#include <QWidget>
#include "opencv2/opencv.hpp"
#include <QTimer>
#include "Utility.h"
#include <vector>
#include <deque>
#include <string>
#include "aidbworker.hpp"
#include "utils.hpp"
#include <QMovie>
#include <QLabel>
#include <QRectf>
#include <QPointf>

#define TW 1282.0
#define TH 720.0

using namespace AIDB;

#define MOBILE_SAM_LIMIT 5

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow {
Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);

    ~MainWindow() override;

    //Mat型图像转换为QImage型图像
//    static QImage MatImageToQt(const cv::Mat &src);

signals:
    void inference();

public slots:
    //读取摄像头每帧数据
    void readFarme();
    void renderImage();
    void renderFrame();
    void update_log();
    void update(const std::shared_ptr<AiDBBin>& bin);
//    void draw(const std::shared_ptr<AiDBBin>);


private:
    int _cur_model_index = -1;
    int _demo_type = 0;
    //声明opencv的视频类
    cv::VideoCapture _cap;
    //声明定时器
    QTimer* _render_timer;
    QTimer* _read_timer;
    QTimer* _log_timer;
    //声明Mat类图像变量
    cv::Mat _mat;
//    QImage _show_image;
    QPixmap _image_pixmap; // 图片模式，保存原图, 插拔模型刷新显示
    QMovie *_logo;
    QLabel *_about;
    QLabel *_page;


    std::deque<QPointF> _mb_sam_points;
    std::deque<QRectF> _mb_sam_rects;
    bool _interactive = false;
    Ui::MainWindow *ui;

private:
    ImageRenderParam _render_param{};
    //Image

    void push_image();

    void push_image(const QPointF& );

    void push_image(const QRectF& );

    void video_release();

    void update_config_widget(const QString& );

    void update_backend();



    static QString log_message;
    static void myMessageOutput(QtMsgType type, const QMessageLogContext &context, const QString &msg);

private:
//    QStringList _modelList{"scrfd_500m_kps", "pfpld", "3ddfa_mb05_bfm_dense","bisenet","movenet",
//                           "yolox_nano","yolov7_tiny","yolov8n",
//                          "ppocr_det", "ppocr_cls", "ppocr_ret", "mobilevit_xxs"};
//
    QStringList _backendList{"ONNX", "MNN", "NCNN", "TNN", "PaddleLite", "OpenVINO"};

    std::vector<int> _backend;

    std::vector<void*> _radio_button;

    QColor _color = {0, 0, 255};

    AiDBMap *_map_ins{};
    std::deque<std::string> _deque_models;
    std::deque<std::string> _deque_backend;

    AiDBWorker* _worker{};
    QThread* _thread{};

    std::shared_ptr<AiDBBin> _bin;
    QString _file_name;
    AiDBQueue<cv::Mat> _frame_queue;

public:
    friend class AiDBWorker;

    bool eventFilter(QObject *watched, QEvent *event) override;
};


#endif //AIDB_QT_DEMO_MAINWINDOW_HPP
