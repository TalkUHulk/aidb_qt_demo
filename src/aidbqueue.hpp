//
// Created by TalkUHulk on 2023/7/14.
//

#ifndef AIDB_QT_DEMO_AIDBQUEUE_HPP
#define AIDB_QT_DEMO_AIDBQUEUE_HPP

#include <queue>
#include <mutex>
#include <condition_variable>

template<typename T>
class AiDBQueue{
public:
    //! 拷贝构造函数
    /*!
     * 拷贝构造函数,禁用编译器默认拷贝构造函数
     */
    AiDBQueue(const AiDBQueue &) = delete;

    //! 拷贝构造函数
    /*!
     * 拷贝构造函数,禁用编译器默认拷贝构造函数
     */
    AiDBQueue(AiDBQueue&&) = delete;

    //! =重载
    /*!
     * =重载,禁用编译器默认
     * @return AiDBQueue&
     */
    AiDBQueue& operator = (const AiDBQueue& ) = delete;

    //! =重载
    /*!
     * =重载,禁用编译器默认
     * @return AiDBQueue&
     */
    AiDBQueue& operator = (AiDBQueue&& ) = delete;

    //! 构造函数
    /*!
     * 构造函数，初始化队列大小
     * @param capacity 目标队列大小
     */
    AiDBQueue(size_t capacity):_capacity(capacity){};

    //! 构造函数
    /*!
     * 构造函数，默认队列大小16
     */
    AiDBQueue():_capacity(16){};

    //! 当前队列大小
    /*!
     *
     * @return 队列大小
     */
    int size(){
        return _queue.size();
    }

    bool empty(){
        return _queue.empty();
    }

    //! 插入数据
    /*!
     * 向队列末尾push数据，当队列满时，阻塞等待；
     * @param item 目标数据
     */
    void push(T& item){
        {
            std::unique_lock<std::mutex> lock(_mutex);
            this->_not_full.wait(lock, [this](){ return _queue.size() < _capacity; });
            this->_queue.push(item);
            //this->_queue.push_back(std::move(item));
        }
        this->_not_empty.notify_one();
    }


    //! 插入数据
    /*!
     * 向队列末尾push数据，非阻塞，当队列满时，返回false；
     * @param item 目标数据（右值引用）
     * @return 是否插入成功，true:成功，false:失败
     */
    bool try_push(T& item){
        {
            std::unique_lock<std::mutex> lock(_mutex);
            if (this->_queue.size() == this->_capacity){
                return false;
            }
            this->_queue.push_back(std::move(item));
        }
        this->_not_empty.notify_one();
        return true;
    }

    //! 弹出数据
    /*!
     * 将队列头部数据赋值到形参，并从队列中弹出该数据，当队列空时，阻塞等待；
     * @param item 目标数据
     */
    void pop(T& item){
        {
            std::unique_lock<std::mutex> lock(this->_mutex);
            this->_not_empty.wait(lock, [this](){return !(this->_queue.empty());});
            item = std::move(this->_queue.front());
            this->_queue.pop();
        }
        this->_not_full.notify_one();
    }

    void pop(){
        {
            std::unique_lock<std::mutex> lock(this->_mutex);
            this->_not_empty.wait(lock, [this](){return !(this->_queue.empty());});
            this->_queue.pop();
        }
        this->_not_full.notify_one();
    }

    //! 弹出数据
    /*!
     * 将队列头部数据赋值到形参，并从队列中弹出该数据，非阻塞，当队列空时，返回false；
     * @param item 目标数据
     * @return 是否弹出成功，true:成功，false:失败
     */
    bool try_pop(T& item){
        {
            std::unique_lock<std::mutex> lock(this->_mutex);
            if(this->_queue.empty()){
                return false;
            }
            //this->_not_empty.wait(lock, [this](){return !(this->_queue.empty());});
            item = std::move(this->_queue.front());
            this->_queue.pop();
        }
        this->_not_full.notify_one();
        return true;
    }

    void front(T& item){
        {
            std::unique_lock<std::mutex> lock(this->_mutex);
            this->_not_empty.wait(lock, [this](){return !(this->_queue.empty());});
            item = this->_queue.front();
        }
    }

private:
    std::queue<T> _queue; /*!< 队列 */
    size_t _capacity; /*!< 队列体积 */
    std::mutex _mutex; /*!< 互斥锁 */
    std::condition_variable _not_empty; /*!< 条件变量：队列非空 */
    std::condition_variable _not_full; /*!< 条件变量：队列未满 */

};//end of class AiDBQueue
#endif //AIDB_QT_DEMO_AIDBQUEUE_HPP
