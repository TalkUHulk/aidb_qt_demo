#include <QApplication>
#include <QPushButton>
#include <QScreen>
#include <QDesktopWidget>
#include "mainwindow.hpp"

// Qt 5.14
int main(int argc, char *argv[]) {


//    int width;
//    int height;
//    {
//        QApplication a (argc, argv);
//        auto const rec = QApplication::desktop()->screenGeometry();
//        height = rec.height();
//        width = rec.width();
//    }
//    if(height < 720 && width < 1280){
//        // Qt >= 5.6
//        qputenv("QT_SCALE_FACTOR", "0.5");
//        qputenv("QT_AUTO_SCREEN_SCALE_FACTOR", "1");
//
//    }

    QApplication a (argc, argv);
    if (QApplication::screens().at(0)->geometry().width() > 2000) // 2000 is just random value.
        QGuiApplication::setAttribute(Qt::AA_EnableHighDpiScaling, true);
    else {
        QGuiApplication::setAttribute(Qt::AA_EnableHighDpiScaling, false );
    }
    QGuiApplication::setAttribute(Qt::AA_EnableHighDpiScaling, true);
    QCoreApplication::setAttribute(Qt::AA_UseHighDpiPixmaps);
    MainWindow w;

    w.show ();
    return QApplication::exec ();
}

//export LD_LIBRARY_PATH=/Users/hulk/Documents/DX_Work/CodeZoo/AIDeployBox/build/source/:/Users/hulk/Documents/DX_Work/CodeZoo/AIDeployBox/libs/mac/x86_64/:$LD_LIBRARY_PATH