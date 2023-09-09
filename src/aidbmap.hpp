//
// Created by TalkUHulk on 2023/7/15.
//

#ifndef AIDB_QT_DEMO_AiDBMap_HPP
#define AIDB_QT_DEMO_AiDBMap_HPP

#include "Interpreter.h"
#include <deque>
#include <map>
#include <mutex>
#include <condition_variable>

class AiDBMap{
public:

    AiDBMap(){
        _map = {{"scrfd_500m_kps", nullptr},
                {"pfpld", nullptr},
                {"bisenet", nullptr},
                {"3ddfa_mb05_bfm_dense", nullptr},
                {"yolox_nano", nullptr},
                {"yolov7-tiny", nullptr},
                {"yolov8n", nullptr},
                {"ppocr_det", nullptr},
                {"ppocr_cls", nullptr},
                {"ppocr_ret", nullptr},
                {"movenet", nullptr},
                {"animeganv2_face_paint_v2", nullptr},
                {"mobilevit_xxs", nullptr},
                {"mobile_sam_encoder", nullptr},
                {"mobile_sam_point_prompt", nullptr},
                {"mobile_sam_box_prompt", nullptr}
                };
    };

    explicit AiDBMap(const std::vector<std::string> &models){
        for(const auto &m: models){
            _map.insert(std::make_pair(m, nullptr));
        }

    };

    int size(){
        return _map.size();
    }

    bool empty(){
        return _map.empty();
    }

    void insert(const std::string &model, const std::string &backend){
//        std::string strs = model + "_";
//        size_t pos = strs.find('_');
//        std::string name = strs.substr(0, pos);

        {
            std::unique_lock<std::mutex> lock(_mutex);
            this->_not_use.wait(lock, [this](){return !_busy;});
            _update = true;
            this->_map[model] = AIDB::Interpreter::createInstance(model, backend);
            _update = false;
        }
        this->_not_update.notify_one();
    }


    void pop(const std::string &model){
//        std::string strs = model + "_";
//        size_t pos = strs.find('_');
//        std::string name = strs.substr(0, pos);

        {
            std::unique_lock<std::mutex> lock(this->_mutex);
            this->_not_use.wait(lock, [this](){return !_busy;});
            _update = true;
            AIDB::Interpreter::releaseInstance((Interpreter*)_map[model]);
            _map[model] = nullptr;
            _update = false;
        }
        this->_not_update.notify_one();

    }

    void* operator[] (const std::string &model){
        return this->_map[model];
    }


    void busy(bool b){
        {
            std::unique_lock<std::mutex> lock(this->_mutex);
            this->_not_use.wait(lock, [this](){return !_update;});
            _busy = b;
        }

        this->_not_use.notify_one();
    }
private:
    std::map<std::string, void*> _map;

    std::mutex _mutex; /*!< 互斥锁 */
    std::condition_variable _not_update; /*!< 条件变量：队列未满 */
    std::condition_variable _not_use; /*!< 条件变量：队列未满 */
    int _busy = false;
    int _update = false;
};//end of class AiDBMap

#endif //AIDB_QT_DEMO_AiDBMap_HPP
