//
// Created by TalkUHulk on 2023/7/15.
//

#ifndef AIDB_QT_DEMO_AIDBDEQUE_HPP
#define AIDB_QT_DEMO_AIDBDEQUE_HPP

#include <deque>
#include <map>
#include <mutex>
#include <condition_variable>

template<typename T>
class AiDBDeque{
public:

    //! 构造函数
    /*!
     * 构造函数，初始化队列大小
     * @param capacity 目标队列大小
     */
    AiDBDeque(size_t capacity):_capacity(capacity){};

    //! 构造函数
    /*!
     * 构造函数，默认队列大小16
     */
    AiDBDeque():_capacity(16){};

    //! 当前队列大小
    /*!
     *
     * @return 队列大小
     */
    int size(){
        return _deque.size();
    }

    bool empty(){
        return _deque.empty();
    }

    //! 插入数据
    /*!
     * 向队列末尾push数据，当队列满时，阻塞等待；
     * @param item 目标数据
     */
    void push(T& item){
        {
            std::unique_lock<std::mutex> lock(_mutex);
            this->_not_full.wait(lock, [this](){ return _deque.size() < _capacity; });
            this->_deque.push_back(item);
        }
        this->_not_empty.notify_one();
    }


    //! 弹出数据
    /*!
     * 将队列头部数据赋值到形参，并从队列中弹出该数据，当队列空时，阻塞等待；
     * @param item 目标数据
     */
    void pop(T& item){
        {
            std::unique_lock<std::mutex> lock(this->_mutex);
            this->_not_empty.wait(lock, [this](){return !(this->_deque.empty());});

            this->_not_use.wait(lock, [this](){return !_busy;});

            item = std::move(this->_deque.back());
            this->_deque.pop_back();
        }
        this->_not_full.notify_one();
    }

    T& operator[] (int index){
        return this->_deque[index];
    }
    void busy(bool b){
        _busy = b;
        this->_not_use.notify_one();
    }
private:
    std::deque<T> _deque; /*!< 队列 */

    size_t _capacity; /*!< 队列体积 */
    std::mutex _mutex; /*!< 互斥锁 */
    std::condition_variable _not_empty; /*!< 条件变量：队列非空 */
    std::condition_variable _not_full; /*!< 条件变量：队列未满 */
    std::condition_variable _not_use; /*!< 条件变量：队列未满 */
    int _busy = false;
};//end of class AiDBDeque

#endif //AIDB_QT_DEMO_AIDBDEQUE_HPP
