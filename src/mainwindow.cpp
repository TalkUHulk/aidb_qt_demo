//
// Created by TalkUHulk on 2023/7/11.
//

// You may need to build the project (run Qt uic code generator) to get "ui_MainWindow.h" resolved

#include <QStandardItemModel>
#include <QTreeView>
#include <QFileDialog>
#include <QColorDialog>
#include "mainwindow.hpp"
#include "forms/ui_MainWindow.h"
#include <QThread>
#include <QTabWidget>
#include <QCheckBox>
#include <QPixmap>
#include <QPainter>
#include <QRect>
#include <QDateTime>
#include <QImageWriter>
#include <QTextStream>
#include <QDebug>

const char* coco_labels[] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
};

int line_map[20][2] = {{2,  1}, {2, 4}, {1, 3}, {4, 0}, {0, 3},
                       {4,  6}, {3, 5}, {6, 8}, {8, 10}, {5, 7}, {7, 9},
                       {6,  12}, {5, 11}, {12, 11}, {12, 14}, {11, 13},
                       {14, 16}, {13, 15}, {2, 0}, {1, 0}};

MainWindow::MainWindow(QWidget *parent) :
        QMainWindow(parent), ui(new Ui::MainWindow) {

    ui->setupUi(this);

    _demo_type = ui->tabWidget->currentIndex();

    //open qss file
    QFile file("./themes/macos.qss");
    file.open(QFile::ReadOnly);

    QString styleSheet { file.readAll() };
    this->setStyleSheet(styleSheet);
    file.close();


    qInstallMessageHandler(myMessageOutput);

    _logo = new QMovie("./resource/AiDBLogo.gif");

    ui->label_logo->setMovie(_logo);
    _logo->start();

    // status bar
    ui->statusBar->setSizeGripEnabled(false);
    _about = new QLabel(this);
    _page = new QLabel(this);

    _page->setFrameStyle(QFrame::Panel | QFrame::Raised);
    _page->setText(tr("<a href=\"https://github.com/TalkUHulk/ai.deploy.box\">AiDB</a>"));
    _page->setOpenExternalLinks(true);

    _about->setFrameStyle(QFrame::Panel | QFrame::Raised);
    _about->setText(tr("<a href=\"https://github.com/TalkUHulk\">About Me</a>"));
    _about->setOpenExternalLinks(true);

    ui->statusBar->addPermanentWidget(_page);
    ui->statusBar->addPermanentWidget(_about);

    ui->teLog->document()->setMaximumBlockCount(100);

    ui->progressBar->setRange(0, 1000);

    _radio_button.push_back(ui->radioButtonOnnx);
    _radio_button.push_back(ui->radioButtonMnn);
    _radio_button.push_back(ui->radioButtonNcnn);
    _radio_button.push_back(ui->radioButtonTnn);
    _radio_button.push_back(ui->radioButtonPaddleLite);
    _radio_button.push_back(ui->radioButtonOpenvino);

    ui->lineeditColor->setStyleSheet(QString("border: 3px solid rgb(%1, %2, %3);").arg(
            QString::number(_color.red()),
            QString::number(_color.green()),
            QString::number(_color.blue()))
    );

    // Image
    connect(ui->tabWidget, static_cast<void (QTabWidget::*)(int)>(&QTabWidget::currentChanged), [=](int index){
        _demo_type = index;
        if(_demo_type == 1){
            _cap.open(0);
            _read_timer->start(25);
            _render_timer->start(25);
            _bin.reset();
            _bin = nullptr;
            qDebug() << "Webcam mode" << endl;
        } else {
            while(!_worker->_frame_queue.empty()){
                _worker->_frame_queue.pop();
            }
            if(_cap.isOpened()){
                video_release();
            }
            qDebug() << "Image Mode" << endl;

        }

    });

    connect(ui->btnOpenImg,
            static_cast<void (QPushButton::*)(bool)>(&QPushButton::clicked),
            [=](bool click){
                _file_name = QFileDialog::getOpenFileName(this,
                                                          tr("Open File"),
                                                          "",
                                                          tr("Text files(*)"));
                renderImage();
                push_image();
                                }
            );

    connect(ui->btnSaveImg,
            static_cast<void (QPushButton::*)(bool)>(&QPushButton::clicked),
            [=](bool click){

                if(!_mat.empty()){
                    auto pixmap = ui->label_show->pixmap();
                    QString dateTime = QDateTime::currentDateTime().toString("yyyyMMdd-hhmmss");
                    QString fileName = "aidb-" + dateTime + ".png";
                    QString filePath = QFileDialog::getSaveFileName(nullptr, "Save Image", fileName, "Images (*.png *.bmp *.jpg)");
                    if (!filePath.isEmpty()) {
                        // 创建目录
                        QString directory = QFileInfo(filePath).dir().path() + "/aidb_visual";
                        QDir().mkpath(directory);

                        // 在目录中保存图像到文件
                        filePath = directory + "/" + fileName;
                        QImageWriter writer(filePath);
                        writer.setFormat("PNG"); // 指定图像格式
                        QImage cutimg = pixmap->toImage().copy(
                                QRect(_render_param._pad_x / 2,
                                      _render_param._pad_y / 2,
                                      _render_param._w,
                                      _render_param._h));

                        writer.write(cutimg.scaled( _render_param._org_w,  _render_param._org_h));

                        qDebug() << "Save image:" << filePath << endl;
                    }

                }

            }
    );


    // scrfd
    connect(ui->cbScrfd,  static_cast<void (QCheckBox::*)(bool)>(&QCheckBox::clicked), [=](int click){
        if(click){
            auto backend = _backendList[_backend[0]];
            qDebug() << "register model: scrfd_500m_kps; backend:" << backend << endl;
            _map_ins->insert("scrfd_500m_kps", backend.toStdString());
        } else{
            qDebug() << "unregister model: scrfd_500m_kps" << endl;
            _map_ins->pop("scrfd_500m_kps");
        }
        push_image();
    });

    connect(ui->btnScrfd,  static_cast<void (QPushButton::*)(bool)>(&QPushButton::clicked), [=](bool click){
        update_config_widget("scrfd_500m_kps");

    });


    // pfpld
    connect(ui->cbPfpld,  static_cast<void (QCheckBox::*)(bool)>(&QCheckBox::clicked), [=](int click){
        if(click){
            auto backend = _backendList[_backend[1]];
            qDebug() << "register model: pfpld; backend:" << backend << endl;
            if((*_map_ins)["scrfd_500m_kps"] == nullptr){
                _map_ins->insert("scrfd_500m_kps", backend.toStdString());
                ui->cbScrfd->setCheckState(Qt::CheckState::Checked);
            }
            _map_ins->insert("pfpld", backend.toStdString());
        } else{
            _map_ins->pop("pfpld");
            qDebug() << "unregister model: pfpld" << endl;
        }
        push_image();
    });

    connect(ui->btnPfpld,  static_cast<void (QPushButton::*)(bool)>(&QPushButton::clicked), [=](bool click){
        update_config_widget("pfpld");
    });

    // tddfa
    connect(ui->cbTddfa,  static_cast<void (QCheckBox::*)(bool)>(&QCheckBox::clicked), [=](int click){
        if(click){
            auto backend = _backendList[_backend[2]];
            if((*_map_ins)["scrfd_500m_kps"] == nullptr){
                _map_ins->insert("scrfd_500m_kps", backend.toStdString());
                ui->cbScrfd->setCheckState(Qt::CheckState::Checked);
                qDebug() << "register model: scrfd_500m_kps; backend:" << backend << endl;
            }
            _map_ins->insert("3ddfa_mb05_bfm_dense", backend.toStdString());
            qDebug() << "register model: 3ddfa_mb05_bfm_dense; backend:" << backend << endl;
        } else{
            _map_ins->pop("3ddfa_mb05_bfm_dense");
            qDebug() << "unregister model: 3ddfa_mb05_bfm_dense" << endl;
        }
        push_image();
    });

    connect(ui->btnTddfa,  static_cast<void (QPushButton::*)(bool)>(&QPushButton::clicked), [=](bool click){
        update_config_widget("3ddfa_mb05_bfm_dense");

    });

    // bisenet
    connect(ui->cbBisenet,  static_cast<void (QCheckBox::*)(bool)>(&QCheckBox::clicked), [=](int click){
        if(click){
            auto backend = _backendList[_backend[3]];
            if((*_map_ins)["scrfd_500m_kps"] == nullptr){
                _map_ins->insert("scrfd_500m_kps", backend.toStdString());
                ui->cbScrfd->setCheckState(Qt::CheckState::Checked);
                qDebug() << "register model: scrfd_500m_kps; backend:" << backend << endl;
            }
            _map_ins->insert("bisenet", backend.toStdString());
            qDebug() << "register model: bisenet; backend:" << backend << endl;
        } else{
            _map_ins->pop("bisenet");
            qDebug() << "unregister model: bisenet" << endl;
        }
        push_image();
    });

    connect(ui->btnBisenet,  static_cast<void (QPushButton::*)(bool)>(&QPushButton::clicked), [=](bool click){
        update_config_widget("bisenet");

    });

    // yolox
    connect(ui->cbYolox,  static_cast<void (QCheckBox::*)(bool)>(&QCheckBox::clicked), [=](int click){
        if(click){
            auto backend = _backendList[_backend[5]];
            _map_ins->insert("yolox_nano", backend.toStdString());
            qDebug() << "register model: yolox_nano; backend:" << backend << endl;
        } else{
            _map_ins->pop("yolox_nano");
            qDebug() << "unregister model: yolox_nano" << endl;
        }
        push_image();
    });

    connect(ui->btnYolox,  static_cast<void (QPushButton::*)(bool)>(&QPushButton::clicked), [=](bool click){
        update_config_widget("yolox_nano");

    });

    //yolov7
    connect(ui->cbYolov7,  static_cast<void (QCheckBox::*)(bool)>(&QCheckBox::clicked), [=](int click){

        if(click){
            auto backend = _backendList[_backend[6]];
            _map_ins->insert("yolov7_tiny", backend.toStdString());
            qDebug() << "register model: yolov7_tiny; backend:" << backend << endl;
        } else{
            _map_ins->pop("yolov7_tiny");
            qDebug() << "unregister model: yolov7_tiny" << endl;
        }
        push_image();
    });

    connect(ui->btnYolov7,  static_cast<void (QPushButton::*)(bool)>(&QPushButton::clicked), [=](bool click){
        update_config_widget("yolov7_tiny");

    });

    // yolov8
    connect(ui->cbYolov8,  static_cast<void (QCheckBox::*)(bool)>(&QCheckBox::clicked), [=](int click){
        if(click){
            auto backend = _backendList[_backend[7]];
            _map_ins->insert("yolov8n", backend.toStdString());
            qDebug() << "register model: yolov8n; backend:" << backend << endl;
        } else{
            _map_ins->pop("yolov8n");
            qDebug() << "unregister model: yolov8n" << endl;
        }
        push_image();
    });

    connect(ui->btnYolov8,  static_cast<void (QPushButton::*)(bool)>(&QPushButton::clicked), [=](bool click){

        update_config_widget("yolov8n");

    });

//    // animegan
//    connect(ui->cbAnimeGan,  static_cast<void (QCheckBox::*)(bool)>(&QCheckBox::clicked), [=](int click){
//        if(click){
//            auto backend = _backendList[_backend[8]];
//            _map_ins->insert("animeganv2_face_paint_v2", backend.toStdString());
//        } else{
//            _map_ins->pop("animeganv2_face_paint_v2");
//        }
//        push_image();
//    });


    // movenet
    connect(ui->cbMovenet,  static_cast<void (QCheckBox::*)(bool)>(&QCheckBox::clicked), [=](int click){

        if(click){
            auto backend = _backendList[_backend[4]];
            _map_ins->insert("movenet", backend.toStdString());
            qDebug() << "register model: movenet; backend:" << backend << endl;
        } else{
            _map_ins->pop("movenet");
            qDebug() << "unregister model: movenet" << endl;
        }
        push_image();
    });

    connect(ui->btnMovenet,  static_cast<void (QPushButton::*)(bool)>(&QPushButton::clicked), [=](bool click){
        update_config_widget("movenet");

    });

    // mobilevit
    connect(ui->cbMobileVit,  static_cast<void (QCheckBox::*)(bool)>(&QCheckBox::clicked), [=](int click){
        if(click){
            auto backend = _backendList[_backend[11]];
            _map_ins->insert("mobilevit_xxs", backend.toStdString());
            qDebug() << "register model: mobilevit; backend:" << backend << endl;
        } else{
            _map_ins->pop("mobilevit_xxs");
            qDebug() << "unregister model: mobilevit_xxs" << endl;
        }
        push_image();
    });

    connect(ui->btnMobileVit,  static_cast<void (QPushButton::*)(bool)>(&QPushButton::clicked), [=](bool click){
        update_config_widget("mobilevit_xxs");

    });

    // ocr
    connect(ui->cbOcr,  static_cast<void (QCheckBox::*)(bool)>(&QCheckBox::clicked), [=](int click){
        if(click){
            _map_ins->insert("ppocr_det", _backendList[_backend[8]].toStdString());
            _map_ins->insert("ppocr_cls", _backendList[_backend[9]].toStdString());
            _map_ins->insert("ppocr_ret", _backendList[_backend[10]].toStdString());
            qDebug() << "register model: ppocr_det; backend:" << _backend[8] << endl;
            qDebug() << "register model: ppocr_cls; backend:" << _backend[9] << endl;
            qDebug() << "register model: ppocr_ret; backend:" << _backend[10] << endl;
        } else{
            _map_ins->pop("ppocr_det");
            _map_ins->pop("ppocr_cls");
            _map_ins->pop("ppocr_ret");
            qDebug() << "unregister model: ppocr_det" << endl;
            qDebug() << "unregister model: ppocr_cls" << endl;
            qDebug() << "unregister model: ppocr_ret" << endl;
        }
        push_image();
    });

    connect(ui->btnOcr,  static_cast<void (QPushButton::*)(bool)>(&QPushButton::clicked), [=](bool click){
        update_config_widget("ppocr_det");

    });

    connect(ui->radioButtonOnnx, static_cast<void (QRadioButton::*)(bool)>(&QRadioButton::clicked), [=](bool click){
        _backend[_cur_model_index] = 0;
        ui->radioButtonMnn->setChecked(false);
        ui->radioButtonOpenvino->setChecked(false);
        ui->radioButtonNcnn->setChecked(false);
        ui->radioButtonTnn->setChecked(false);
        ui->radioButtonPaddleLite->setChecked(false);
        update_backend();

    });

    connect(ui->radioButtonMnn, static_cast<void (QRadioButton::*)(bool)>(&QRadioButton::clicked), [=](bool click){
        _backend[_cur_model_index] = 1;
        ui->radioButtonOnnx->setChecked(false);
        ui->radioButtonOpenvino->setChecked(false);
        ui->radioButtonNcnn->setChecked(false);
        ui->radioButtonTnn->setChecked(false);
        ui->radioButtonPaddleLite->setChecked(false);
        update_backend();

    });

    connect(ui->radioButtonNcnn, static_cast<void (QRadioButton::*)(bool)>(&QRadioButton::clicked), [=](bool click){
        _backend[_cur_model_index] = 2;
        ui->radioButtonMnn->setChecked(false);
        ui->radioButtonOnnx->setChecked(false);
        ui->radioButtonOpenvino->setChecked(false);
        ui->radioButtonTnn->setChecked(false);
        ui->radioButtonPaddleLite->setChecked(false);
        update_backend();

    });

    connect(ui->radioButtonTnn, static_cast<void (QRadioButton::*)(bool)>(&QRadioButton::clicked), [=](bool click){
        _backend[_cur_model_index] = 3;
        ui->radioButtonMnn->setChecked(false);
        ui->radioButtonOnnx->setChecked(false);
        ui->radioButtonOpenvino->setChecked(false);
        ui->radioButtonNcnn->setChecked(false);
        ui->radioButtonPaddleLite->setChecked(false);
        update_backend();

    });

    connect(ui->radioButtonPaddleLite, static_cast<void (QRadioButton::*)(bool)>(&QRadioButton::clicked), [=](bool click){
        _backend[_cur_model_index] = 4;
        ui->radioButtonMnn->setChecked(false);
        ui->radioButtonOnnx->setChecked(false);
        ui->radioButtonOpenvino->setChecked(false);
        ui->radioButtonNcnn->setChecked(false);
        ui->radioButtonTnn->setChecked(false);
        update_backend();

    });

    connect(ui->radioButtonOpenvino, static_cast<void (QRadioButton::*)(bool)>(&QRadioButton::clicked), [=](bool click){
        _backend[_cur_model_index] = 5;
        ui->radioButtonMnn->setChecked(false);
        ui->radioButtonOnnx->setChecked(false);
        ui->radioButtonNcnn->setChecked(false);
        ui->radioButtonTnn->setChecked(false);
        ui->radioButtonPaddleLite->setChecked(false);
        update_backend();

    });

    // Video
    _read_timer = new QTimer(this);
    _render_timer = new QTimer(this);
    _log_timer = new QTimer(this);

    connect(_read_timer, SIGNAL(timeout()),this, SLOT(readFarme()));
    connect(_render_timer, SIGNAL(timeout()),this, SLOT(renderFrame()));
    connect(_log_timer, SIGNAL(timeout()),this, SLOT(update_log()));

    _log_timer->start(50);


    // init AiDB
    std::vector<std::string> vec_models;

    foreach( QString str, _worker->_modelList) {
        vec_models.push_back(str.toStdString());
    }

    _map_ins = new AiDBMap(vec_models);

    _worker = new AiDBWorker();
    _thread = new QThread();
    _worker->link(_map_ins);

    connect(this, SIGNAL(inference()),_worker, SLOT(forward()));

    void (AiDBWorker::*sgn)(const std::shared_ptr<AiDBBin>)= &AiDBWorker::finish;
    connect(_worker, sgn, this, &MainWindow::update);

    // connect
//    ui->modelconfigWidget->hide();
    ui->comBoxBackend->addItems(_backendList);

    for(int i = 0; i < AiDBWorker::_modelList.size(); i++){
        _backend.push_back(0);
    }

    connect(ui->btnColor,
            static_cast<void (QPushButton::*)(bool)>(&QPushButton::clicked),
            [=](bool click){
                if(_cur_model_index != -1){
                    _color = QColorDialog::getColor(Qt::red, this,
                                                                    tr("颜色选择"),
                                                                    QColorDialog::ShowAlphaChannel);
                    auto color_str = QString::number(_color.red())+","
                                     +QString::number(_color.green())+","
                                     +QString::number(_color.blue());

                ui->lineeditColor->setStyleSheet(QString("border: 3px solid rgb(%1, %2, %3);").arg(
                        QString::number(_color.red()),
                        QString::number(_color.green()),
                        QString::number(_color.blue()))
                        );

                ui->lineeditColor->setText(color_str);


                }


            }
    );
    connect(ui->comBoxBackend, static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged),
            [=](int index){
                for(auto &b: _backend){
                    b = index;
                }
                update_backend();
                for(int i = 0; i < _radio_button.size(); i++){
                    if(i == index){
                        ((QRadioButton*)_radio_button[i])->setChecked(true);
                    } else{
                        ((QRadioButton*)_radio_button[i])->setChecked(false);
                    }
                }

            });

    // run
    _worker->moveToThread(_thread);
    _thread->start();

    emit inference();


    qInfo() << "Click the left model button to select a specific backend." << endl
            << "Check to register the model." << endl
            << "Provide Image and Webcam mode." << endl;


}

MainWindow::~MainWindow() {

    _worker->stop();
    _thread->quit();
    _thread->wait();

    _log_timer->stop();
    delete _map_ins;
    delete _logo;
    delete _log_timer;
    delete _worker;
    delete _thread;
    delete ui;
}

void MainWindow::video_release(){
    _read_timer->stop();
    _render_timer->stop();
    _cap.release();
}

void MainWindow::push_image(){
    if(!_mat.empty()){
        _worker->_frame_queue.push(_mat);
    }
}

void MainWindow::renderImage() {
    _mat = cv::imread(_file_name.toStdString());

    if(_mat.empty()){
        qCritical() << "file open failed:" << _file_name << endl;
        return;
    }

    qDebug() << "open file:" << _file_name << endl;
    int width = ui->label_show->width();
    int height = ui->label_show->height();

    int src_w = _mat.cols, src_h = _mat.rows;

    auto ratio_src = float(src_w) / float(src_h);
    auto ratio_label = float(width) / float(height);

    _render_param._org_w = src_w;
    _render_param._org_h = src_h;

    if(ratio_src >= ratio_label){
        _render_param._w = width;
        _render_param._h = float(_render_param._w) / src_w * src_h;
        _render_param._pad_y = height - _render_param._h;
        _render_param._scale = float(_render_param._w) / src_w;
    } else{
        _render_param._h = height;
        _render_param._w = float(_render_param._h) / src_h * src_w;
        _render_param._pad_x = width - _render_param._w;
        _render_param._scale = float(_render_param._h) / src_h;
    }

    cv::Mat show;
    cv::resize(_mat, show, cv::Size(_render_param._w, _render_param._h));
    cv::copyMakeBorder(show, show,
                       int(_render_param._pad_y / 2.0f), int(_render_param._pad_y / 2.0f),
                       int((_render_param._pad_x + 1) / 2.0f), int((_render_param._pad_x + 1) / 2.0f),
                       cv::BORDER_CONSTANT, cv::Scalar::all(224));

    QImage qimage;
    Mat2Qt(show, qimage);
    _image_pixmap = QPixmap::fromImage(qimage);
    ui->label_show->setPixmap(_image_pixmap);

}

void MainWindow::update(const std::shared_ptr<AiDBBin>& bin) {

    if(_demo_type == 1){
        _bin = bin;
        ui->progressBar->setValue(ui->progressBar->value()== 1000 ? 0: ui->progressBar->value() + 1);

    } else {
        if(ui->label_show->pixmap() && bin){
            QPixmap pixmap;
            if(!bin->generated.empty()){
                cv::Mat show = bin->generated;

                    cv::resize(show, show, cv::Size(_render_param._w, _render_param._h));
                    cv::copyMakeBorder(show, show,
                                       int(_render_param._pad_y / 2.0f), int(_render_param._pad_y / 2.0f),
                                       int((_render_param._pad_x + 1) / 2.0f), int((_render_param._pad_x + 1) / 2.0f),
                                       cv::BORDER_CONSTANT, cv::Scalar::all(224));


                QImage tmp;
                Mat2Qt(show, tmp);
                pixmap = QPixmap::fromImage(tmp);
            } else {
                pixmap = _demo_type == 0 ? _image_pixmap : *ui->label_video->pixmap();
            }


            QPainter painter(&pixmap);
            QPen paintpen(_color);
            paintpen.setWidth(3);
            painter.setPen(paintpen);

            if(!bin->face_meta.empty()){
                bool has_parsing = !bin->face_parsing.empty();
                for(int index = 0; index < bin->face_meta.size(); index++){
                    // scrfd
                    painter.drawRect(QRectF(bin->face_meta[index]->x1 * _render_param._scale + _render_param._pad_x / 2.0f, bin->face_meta[index]->y1 * _render_param._scale + _render_param._pad_y / 2.0f,
                                            (bin->face_meta[index]->x2 - bin->face_meta[index]->x1) * _render_param._scale, (bin->face_meta[index]->y2 - bin->face_meta[index]->y1) * _render_param._scale));


                    for(int n = 0; n < bin->face_meta[index]->kps.size() / 2; n++){
                        painter.drawEllipse(QPointF(bin->face_meta[index]->kps[2 * n] * _render_param._scale + _render_param._pad_x / 2.0f,
                                                    bin->face_meta[index]->kps[2 * n + 1] * _render_param._scale + _render_param._pad_y / 2.0f),
                                            1, 1);
                    }

                    if(has_parsing){
                        cv::Mat parsing;
                        int len = fmin(64.0f, bin->face_meta[index]->width());
                        cv::resize(bin->face_parsing[index], parsing, cv::Size(len, len));
                        QImage _image;
                        Mat2Qt(parsing, _image);
                        painter.drawPixmap(bin->face_meta[index]->x1 * _render_param._scale + _render_param._pad_x / 2.0f,
                                           (bin->face_meta[index]->y1 - len) * _render_param._scale + _render_param._pad_y / 2.0f, QPixmap::fromImage(_image));
                    }
                }

            }

            if(!bin->object_meta.empty()){

                auto font = painter.font();
                font.setPointSize(font.pointSize());
                painter.setFont(font);

                for(const auto &meta: bin->object_meta){
                    painter.drawRect(QRectF(meta->x1 * _render_param._scale + _render_param._pad_x / 2.0f, meta->y1 * _render_param._scale + _render_param._pad_y / 2.0f,
                                            (meta->x2 - meta->x1) * _render_param._scale, (meta->y2 - meta->y1) * _render_param._scale));
                    auto label = QString(coco_labels[meta->label]);
                    painter.drawText(QPoint(meta->x1 * _render_param._scale + _render_param._pad_x / 2.0f, meta->y1 * _render_param._scale + _render_param._pad_y / 2.0f),
                                     label);
                }

            }

            if(!bin->ocr_meta.empty()){

                auto font = painter.font();
                font.setPointSize(font.pointSize());
                painter.setFont(font);
                for(const auto &meta: bin->ocr_meta){
                    QPolygonF polygon;
                    float min_x = _mat.cols, min_y = _mat.rows;
                    for(const auto &p: meta->box){

                        polygon << QPointF(p._x * _render_param._scale + _render_param._pad_x / 2.0f, p._y * _render_param._scale + _render_param._pad_y / 2.0f);
                        min_x = fmin(min_x, p._x);
                        min_y = fmin(min_y, p._y);
                    }
                    // ppocr dbnet
                    painter.drawPolygon(polygon, Qt::FillRule::OddEvenFill);

                    // ppocr crnn
                    auto label = QString(meta->label.c_str());
                    painter.drawText(QPointF(min_x * _render_param._scale + _render_param._pad_x / 2.0f, min_y * _render_param._scale + _render_param._pad_y / 2.0f),
                                     label);

                }

            }

            if(!bin->human_keypoints.empty()){
                // movenet
                for(auto kps: bin->human_keypoints){
                    painter.drawEllipse(QPointF(kps[0] * _render_param._scale + _render_param._pad_x / 2.0f, kps[1] * _render_param._scale + _render_param._pad_y / 2.0f),
                                        3, 3);
                }
                for(auto & index : line_map) {
                    painter.drawLine(QPointF(bin->human_keypoints[index[0]][0] * _render_param._scale + _render_param._pad_x / 2.0f, bin->human_keypoints[index[0]][1] * _render_param._scale + _render_param._pad_y / 2.0f),
                                     QPointF(bin->human_keypoints[index[1]][0] * _render_param._scale + _render_param._pad_x / 2.0f, bin->human_keypoints[index[1]][1] * _render_param._scale + _render_param._pad_y / 2.0f));
                }
            }

            if(!bin->cls_meta.empty()){
                QString _log("Classification:");
                for(int tk = 0; tk < bin->cls_meta.size(); tk++){
                    _log += QString(" Top:%1: [label:%2, conf:%3]").arg(
                            QString::number(tk),
                            bin->cls_meta[tk]->label_str.c_str(),
                            QString::number(bin->cls_meta[tk]->conf));

                }
                qInfo() << _log << endl;

            }

            ui->label_show->setPixmap(pixmap);
        }
    }

}

void MainWindow::readFarme() {
    //读取一帧图像
    if(_cap.isOpened()){
        cv::Mat frame;
        _cap.read(frame) ;
        _worker->_frame_queue.push(frame);
        _frame_queue.push(frame);
    }

}


void MainWindow::renderFrame() {

    //获取ui->label的尺寸
    int width = ui->label_video->width();
    int height = ui->label_video->height();

    //将opencv的mat型resize到label的尺寸
    cv::Mat show;
    _frame_queue.pop(show);

    int src_w = show.cols, src_h = show.rows;

    float ratio_src = float(src_w) / float(src_h);
    float ratio_label = float(width) / float(height);
    int nw, nh, pad_x = 0, pad_y = 0;

    float scale = 1.f;
    if(ratio_src >= ratio_label){
        nw = width;
        nh = float(nw) / src_w * src_h;
        pad_y = height - nh;
        scale = float(nw) / src_w;
    } else{
        nh = height;
        nw = float(nh) / src_h * src_w;
        pad_x = width - nw;
        scale = float(nh) / src_h;
    }

    if(_bin){
        if(!_bin->generated.empty())
            show = _bin->generated;

        cv::resize(show, show, cv::Size(nw, nh));
        cv::copyMakeBorder(show, show,
                           int(pad_y / 2.0f), int((pad_y + 1) / 2.0f),
                           int(pad_x / 2.0f), int((pad_x + 1) / 2.0f),
                           cv::BORDER_CONSTANT, cv::Scalar::all(224));

        //转换图像数据类型
        QImage imag;
        Mat2Qt(show, imag);
        QPixmap pixmap = QPixmap::fromImage(imag);

        QPainter painter(&pixmap);
        QPen paintpen(_color);
        paintpen.setWidth(3);

        painter.setPen(paintpen);
        if(!_bin->face_meta.empty()){
            bool has_parsing = !_bin->face_parsing.empty();
            for(int index = 0; index < _bin->face_meta.size(); index++){
                painter.drawRect(QRectF( _bin->face_meta[index]->x1 * scale + pad_x / 2.0f,  _bin->face_meta[index]->y1 * scale + pad_y / 2.0f,
                                        ( _bin->face_meta[index]->x2 -  _bin->face_meta[index]->x1) * scale, ( _bin->face_meta[index]->y2 -  _bin->face_meta[index]->y1) * scale));

                for(int n = 0; n <  _bin->face_meta[index]->kps.size() / 2; n++){
                    painter.drawEllipse(QPointF( _bin->face_meta[index]->kps[2 * n] * scale + pad_x / 2.0f,
                                                 _bin->face_meta[index]->kps[2 * n + 1] * scale + pad_y / 2.0f),
                                        1, 1);

                    if(has_parsing){
                        cv::Mat parsing;
                        int len = fmin(64.0f, _bin->face_meta[index]->width());
                        cv::resize(_bin->face_parsing[index], parsing, cv::Size(len, len));
                        QImage _image;
                        Mat2Qt(parsing, _image);
                        painter.drawPixmap(_bin->face_meta[index]->x1 * _render_param._scale + _render_param._pad_x / 2.0f,
                                           (_bin->face_meta[index]->y1 - len) * _render_param._scale + _render_param._pad_y / 2.0f, QPixmap::fromImage(_image));
                    }
                }
            }

        }
        if(!_bin->object_meta.empty()){
            auto font = painter.font();
            font.setPointSize(font.pointSize());
            painter.setFont(font);

            for(const auto &meta: _bin->object_meta){
                painter.drawRect(QRectF(meta->x1 * scale + pad_x / 2.0f, meta->y1 * scale + pad_y / 2.0f,
                                        (meta->x2 - meta->x1) * scale, (meta->y2 - meta->y1) * scale));
                auto label = QString(coco_labels[meta->label]);
                painter.drawText(QPoint(meta->x1 * scale + pad_x / 2.0f, meta->y1 * scale + pad_y / 2.0f),
                                 label);
            }

        }

        if(!_bin->ocr_meta.empty()){
            auto font = painter.font();
            font.setPointSize(font.pointSize());
            painter.setFont(font);
            for(const auto &meta: _bin->ocr_meta){
                QPolygonF polygon;
                float min_x = show.cols, min_y = show.rows;
                for(const auto &p: meta->box){

                    polygon << QPointF(p._x * scale + pad_x / 2.0f, p._y * scale + pad_y / 2.0f);
                    min_x = fmin(min_x, p._x);
                    min_y = fmin(min_y, p._y);
                }
                painter.drawPolygon(polygon, Qt::FillRule::OddEvenFill);

                auto label = QString(meta->label.c_str());
                painter.drawText(QPointF(min_x * scale + pad_x / 2.0f, min_y * scale + pad_y / 2.0f),
                                 label);

            }

        }

        if(!_bin->human_keypoints.empty()){
            for(auto kps: _bin->human_keypoints){
                painter.drawEllipse(QPointF(kps[0] * scale + pad_x / 2.0f, kps[1] * scale + pad_y / 2.0f),
                                    3, 3);
            }
            for(auto & index : line_map) {
                painter.drawLine(QPointF(_bin->human_keypoints[index[0]][0] * scale + pad_x / 2.0f, _bin->human_keypoints[index[0]][1] * scale + pad_y / 2.0f),
                                 QPointF(_bin->human_keypoints[index[1]][0] * scale + pad_x / 2.0f, _bin->human_keypoints[index[1]][1] * scale + pad_y / 2.0f));
            }
        }

        if(!_bin->cls_meta.empty()){
            QString _log("Classification:");
            for(int tk = 0; tk < _bin->cls_meta.size(); tk++){
                _log += QString(" Top:%1: [label:%2, conf:%3]").arg(
                        QString::number(tk),
                        _bin->cls_meta[tk]->label_str.c_str(),
                        QString::number(_bin->cls_meta[tk]->conf));

            }
            qInfo() << _log << endl;

        }

//        if(!_bin->face_parsing.empty()){
//            cv::Mat GM;
//            std::for_each(_bin->face_parsing.begin(), _bin->face_parsing.end(), [=](cv::Mat &mat){ cv::resize(mat, mat, cv::Size(128, 128));});
//            hconcat(_bin->face_parsing, GM);
//            ui->labelparsing->setPixmap(QPixmap::fromImage(MatImageToQt(GM)));
//        }
//        if(!_bin->generated.empty()){
//            std::vector<cv::Mat> G;
//            for(const auto &it: _bin->generated){
//                cv::Mat tmp;
//                float scale = 128.0f / it.second.cols;
//                cv::resize(it.second, tmp, cv::Size(0, 0), scale, scale);
//                G.emplace_back(tmp);
//            }
//            cv::Mat GM;
//            hconcat(G, GM);
//            ui->labelgenerated->setPixmap(QPixmap::fromImage(MatImageToQt(GM)));
//        }

        ui->label_video->setPixmap(pixmap);
    } else{
        cv::resize(show, show, cv::Size(nw, nh));
        cv::copyMakeBorder(show, show,
                           int(pad_x / 2.0f), int(pad_y / 2.0f),
                           int((pad_x + 1) / 2.0f), int((pad_y + 1) / 2.0f),
                           cv::BORDER_CONSTANT, cv::Scalar::all(0));

        //转换图像数据类型
        QImage imag;
        Mat2Qt(show, imag);
        QPixmap pixmap = QPixmap::fromImage(imag);
        ui->label_video->setPixmap(pixmap);
    }


}

void MainWindow::update_config_widget(const QString& model_name) {
    ui->label_model_name->setText(model_name);
    _cur_model_index = AiDBWorker::_modelList.indexOf(model_name);


    switch (_cur_model_index) {
        case 8:{
            ui->radioButtonNcnn->setEnabled(false);
            ui->radioButtonTnn->setEnabled(false);
            ui->radioButtonPaddleLite->setEnabled(false);
            break;
        }
        case 9:{
            ui->radioButtonNcnn->setEnabled(false);
            ui->radioButtonTnn->setEnabled(false);
            break;
        }
        case 10:{
            ui->radioButtonNcnn->setEnabled(false);
            ui->radioButtonTnn->setEnabled(false);
            break;
        }
        case 11:{
            ui->radioButtonNcnn->setEnabled(false);
            ui->radioButtonTnn->setEnabled(false);
            break;
        }
        case 12:{
            ui->radioButtonNcnn->setEnabled(false);
            ui->radioButtonTnn->setEnabled(false);
            ui->radioButtonPaddleLite->setEnabled(false);
            break;
        }
        default:{
            ui->radioButtonNcnn->setEnabled(true);
            ui->radioButtonTnn->setEnabled(true);
            ui->radioButtonPaddleLite->setEnabled(true);
        }



    }

    //"ONNX", "MNN", "NCNN", "TNN", "PaddleLite", "OpenVINO"
    for(int i = 0; i < _radio_button.size(); i++){
        if(i == _backend[_cur_model_index]){
            ((QRadioButton*)_radio_button[i])->setChecked(true);
        } else{
            ((QRadioButton*)_radio_button[i])->setChecked(false);
        }
    }
}

void MainWindow::myMessageOutput(QtMsgType type, const QMessageLogContext &context, const QString &msg) {

    QString log_level;
    log_message.clear();
    QTextStream stream(&log_message);
    switch (type) {
        case QtDebugMsg: {
            log_level = "[D] ";
            stream << "<span style='color:green;font-size:8pt'>";
            break;
        }
        case QtInfoMsg: {
            log_level = "[I] ";
            stream << "<span style='color:blue;font-size:8pt'>";
            break;
        }
        case QtWarningMsg: {
            log_level = "[W] ";
            stream << "<span style='color:gold;font-size:8pt'>";
            break;
        }
        case QtCriticalMsg: {
            log_level = "[C] ";
            stream << "<span style='color:red;font-size:8pt'>";
            break;
        }
        case QtFatalMsg: {
            log_level = "[F] ";
            stream << "<span style='color:yellow;font-size:8pt'>";
            break;
        }
        default: {
            stream << "<span style='color:red;font-size:8pt'>";
            break;
        }
    }
    stream << "[" << QDateTime::currentDateTime().toString("yyyy-MM-dd hh:mm:ss") << "] "
           << log_level << msg << "</span>";


}

void MainWindow::update_log() {
    static QString last_log = "";
    if(last_log != log_message){
        last_log = log_message;
        ui->teLog->append(log_message);
    }
}

QString MainWindow::log_message = "";

void MainWindow::update_backend() {
    ui->btnScrfd->setToolTip(_backendList[_backend[0]]);
    ui->btnPfpld->setToolTip(_backendList[_backend[1]]);
    ui->btnBisenet->setToolTip(_backendList[_backend[3]]);
    ui->btnTddfa->setToolTip(_backendList[_backend[2]]);
    ui->btnYolox->setToolTip(_backendList[_backend[5]]);
    ui->btnYolov7->setToolTip(_backendList[_backend[6]]);
    ui->btnYolov8->setToolTip(_backendList[_backend[7]]);
    ui->btnMobileVit->setToolTip(_backendList[_backend[11]]);
    ui->btnMovenet->setToolTip(_backendList[_backend[4]]);
    ui->btnOcr->setToolTip(_backendList[_backend[8]]);

}

