#include <QApplication>
#include <QPushButton>
#include "mainwindow.hpp"


int main(int argc, char *argv[]) {
//    QApplication a(argc, argv);
//    QPushButton button("Hello world!", nullptr);
//    button.resize(200, 100);
//    button.show();
//    return QApplication::exec();
    QApplication a (argc, argv);
    MainWindow w;

    w.show ();
    return QApplication::exec ();
}
