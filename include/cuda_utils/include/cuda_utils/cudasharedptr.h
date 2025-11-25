#ifndef CUDASHAREDPTR_H
#define CUDASHAREDPTR_H

//========================================================================
/*!
  @file
  @class
  @brief
    This header file provide smart shared pointer for device buffer.
    Cuda device pointer will be automatically destroyed when refcount reaches to zero.

  @author  Seyed Mahdi Roostaiyan, (C) 2020
*/
//========================================================================


#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <memory>
#include <cstring>
#include <assert.h>

namespace inner_implementation {
template <typename T_ELEM>
struct CudaPtrWrapper {

public:
    CudaPtrWrapper(bool host_usage, bool buffer_reuse){
        this->host_usage = host_usage;
        this->buffer_reuse = buffer_reuse;
    }

    cudaError_t getStatus() const{
        return status;
    }

    bool create(size_t size, bool host_pinned_mem = false){
        if(n_elements == size)
            return true;
        //! decrease number of cudaFree & cudaMalloc
        if(size<=alloc_size && buffer_reuse){
            n_elements = size;
            return true;
        }
        destroy();
        if(size<=0)
            return false;
        n_elements = size;
        size_t buffer_size = n_elements*sizeof(T_ELEM);
        if(host_usage){
            pinned_mem = host_pinned_mem;
            if(pinned_mem){
                cudaError_t pinned_status = cudaMallocHost((void**)(&host_arr), buffer_size); // host pinned
                if(pinned_status!=cudaSuccess)
                    pinned_mem = false;
            }
            if(!pinned_mem){
                host_arr = static_cast<T_ELEM*>(malloc(buffer_size)); // Allocate array on host
                // host_arr = new T_ELEM[buffer_size]; // Allocate array on host
            }
        }

        status = cudaMalloc(&data, buffer_size);
        if(status!=cudaSuccess){
            std::cerr << "Error: cudaMalloc was failed " << std::endl;
            return false;
        }
        alloc_size = n_elements;
        return true;
    }

    void upload(const T_ELEM* data_arr, size_t size, cudaStream_t stream){
        if(!create(size))
            return;
        size_t buffer_size = n_elements*sizeof(T_ELEM);
        if(stream)
        {
            status = cudaMemcpyAsync(data, data_arr, buffer_size, cudaMemcpyHostToDevice, stream);
            if(status!=cudaSuccess)
                throw " Upload Error: cudaMemcpyAsync was failed ";
        }
        else {
            status = cudaMemcpy(data, data_arr, buffer_size, cudaMemcpyHostToDevice);
            if(status!=cudaSuccess)
                throw " Upload Error: cudaMemcpy was failed ";
        }
    }

    void upload(cudaStream_t stream){
        assert(host_usage && host_arr!=nullptr && n_elements>0);
        upload(host_arr, n_elements, stream);
    }

    void upload(const std::vector<T_ELEM> &data_vec, cudaStream_t stream) {
        size_t size = data_vec.size();
        if(size==0 || !create(size))
            return;

        assert(host_usage && host_arr!=nullptr && n_elements>0);
        // std::copy(data_vec.begin(), data_vec.end(), host_arr); => call destructor for malloced memory!
        host_vec = std::shared_ptr<T_ELEM[]>(new T_ELEM[n_elements]);
        std::copy(data_vec.begin(), data_vec.end(), host_vec.get());
        std::memcpy(host_arr, host_vec.get(), n_elements*sizeof(T_ELEM));
        upload(host_arr, n_elements, stream);
    }

    void download(T_ELEM* data_arr, cudaStream_t stream) const {
        if(n_elements<=0)
            return;
        size_t buffer_size = n_elements*sizeof(T_ELEM);
        if(stream)
            status = cudaMemcpyAsync(data_arr, data, buffer_size, cudaMemcpyDeviceToHost, stream);
        else
            status = cudaMemcpy(data_arr, data, buffer_size, cudaMemcpyDeviceToHost);

        if(status!=cudaSuccess){
            if(data)
                status = cudaFree(data);

            if(stream)
                throw  "Download Error: cudaMemcpyAsync was failed ";
            else
                throw  "Download Error: cudaMemcpy was failed ";

        }
    }

    void download(cudaStream_t stream) const {
        if(n_elements<=0)
            return;
        assert(host_usage && n_elements>0);
        download(host_arr, stream);
    }

    void download(std::vector<T_ELEM> &data_vec, cudaStream_t stream) const {
        if(n_elements<=0)
            return;
        assert(host_usage);
        download(host_arr, stream);
        if(stream)
            status = cudaStreamSynchronize(stream);
        else
            status = cudaDeviceSynchronize();
        if(data_vec.size()<n_elements)
            data_vec.resize(n_elements);
        std::copy(host_arr, host_arr+n_elements, data_vec.begin());
    }

    size_t size() const {
        return n_elements;
    }

    void clear(){
        if(!buffer_reuse){
            destroy();
        } else {
            n_elements = 0;
        }
    }

    T_ELEM* dev_ptr() const {
        if(n_elements>0)
            return data;
        else
            return nullptr;
    }

    T_ELEM* host_ptr() const {
        if(n_elements>0)
            return host_arr;
        else
            return nullptr;
    }

    void destroy(){
        if(data){
            status = cudaFree(data);
            if(status!=cudaSuccess){
                std::cerr <<  "Destroy Error: cudaFree was failed ";
            }
            data = nullptr;
        }
        // host_arr.reset();
        if(host_arr){
            if(pinned_mem)
                status = cudaFreeHost(host_arr);
            else
                free(host_arr); // delete [] host_arr; when `new`is used instead of `malloc`
            host_arr = nullptr;
            if(pinned_mem && status!=cudaSuccess){
                std::cerr <<  "Destroy Error: cudaFreeHost was failed ";
            }
        }
        n_elements = 0;
        alloc_size = 0;
    }

    ~CudaPtrWrapper() {
        destroy();
    }

private:
    size_t n_elements = 0;
    size_t alloc_size = 0;
    T_ELEM* data = nullptr;

    mutable T_ELEM *host_arr = nullptr;
    mutable cudaError_t status = cudaSuccess;
    bool buffer_reuse = true;
    bool host_usage = true;
    bool pinned_mem = false;

    mutable std::shared_ptr<T_ELEM[]> host_vec;

};
}

namespace fun {

enum StreamPriority {
    Default = 0,
    Low     = 1,
    High    = 2
};

}

namespace inner_implementation {
struct StreamPtrWrapper {
public:
    StreamPtrWrapper(fun::StreamPriority priority){
        if(priority==fun::Default){
            if (cudaStreamCreate(&ptr) != cudaSuccess)
                throw "Stream Create Error: cudaStreamCreate was failed";
            this->priority_num = 0;
        } else {
            int low, high;
            cudaDeviceGetStreamPriorityRange(&low, &high);
            this->priority_num = (priority==fun::Low) ? low : high;
            if (cudaStreamCreateWithPriority(&ptr, cudaStreamNonBlocking, priority_num) != cudaSuccess)
                throw "Stream Create Error: cudaStreamCreateWithPriority was failed";
        }
    }

    ~StreamPtrWrapper(){
        if(ptr && (cudaStreamDestroy(ptr)!= cudaSuccess))
            std::cerr << "failed to destroy cuda stream" << std::endl;
        ptr = nullptr;
    }

public:
    cudaStream_t ptr = nullptr;
    int priority_num = 0;
};
}

namespace fun::cuda {

template <typename T_ELEM>
struct shared_ptr {

public:
    shared_ptr(bool host_usage=false, bool buffer_reuse=true){
        container = std::make_shared<inner_implementation::CudaPtrWrapper<T_ELEM>>(host_usage, buffer_reuse);
    }

    shared_ptr(size_t n_elements, bool host_usage=false, bool buffer_reuse=true){
        container = std::make_shared<inner_implementation::CudaPtrWrapper<T_ELEM>>(host_usage, buffer_reuse);
        create(n_elements);
    }

    ~shared_ptr(){
        if(container)
            container.reset();
    }

    void destroy(){
        if(container)
            container->destroy();
    }

    void clear(){
        if(container)
            container->clear();
    }
    T_ELEM* data(){
        return container->dev_ptr();
    }

    const T_ELEM* data() const {
        return container->dev_ptr();
    }

    T_ELEM* data_host(){
        return container->host_ptr();
    }

    const T_ELEM* data_host() const {
        return container->host_ptr();
    }

    cudaError_t getStatus() const {
        return container->getStatus();
    }

    size_t size() const {
        return container->size();
    }

    bool create(size_t n_elements){
        return container->create(n_elements);
    }

    void upload_async(const T_ELEM* data_arr, size_t n_elements, cudaStream_t stream){
        container->upload(data_arr, n_elements, stream);
    }
    void upload_async(cudaStream_t stream){
        container->upload(stream);
    }
    void upload_async(const std::vector<T_ELEM> &data_vec, cudaStream_t stream){
        container->upload(data_vec, stream);
    }
    void upload(const T_ELEM* data_arr, size_t n_elements){
        upload_async(data_arr, n_elements, nullptr);
    }
    void upload(){
        upload_async(nullptr);
        cudaDeviceSynchronize();
    }
    void upload(const std::vector<T_ELEM> &data_vec){
        upload_async(data_vec, nullptr);
        cudaDeviceSynchronize();
    }

    void download_async(T_ELEM* data_vec, cudaStream_t stream) const {
        container->download(data_vec, stream);
    }
    void download_async(cudaStream_t stream) const {
        container->download(stream);
    }
    void download_async(std::vector<T_ELEM> &data_vec, cudaStream_t stream) const {
        container->download(data_vec, stream);
    }
    void download(T_ELEM* data_vec) const {
        download_async(data_vec, nullptr);
        cudaDeviceSynchronize();
    }
    void download() const {
        download_async(nullptr);
        cudaDeviceSynchronize();
    }
    void download(std::vector<T_ELEM> &data_vec) const {
        download_async(data_vec, nullptr);
        cudaDeviceSynchronize();
    }

protected:
    std::shared_ptr<inner_implementation::CudaPtrWrapper<T_ELEM>> container;
};


template <typename T_ELEM>
class unique_ptr : public shared_ptr<T_ELEM> {

public:
    ~unique_ptr() = default;
    unique_ptr(unique_ptr const&) = delete;
    unique_ptr& operator=(unique_ptr const&) = delete;


    unique_ptr(bool host_usage=false, bool buffer_reuse=true):
        shared_ptr<T_ELEM>(host_usage, buffer_reuse)
    {}

    unique_ptr(unique_ptr&& other) :
        shared_ptr<T_ELEM>::container(std::exchange(other.shared_ptr<T_ELEM>::container, nullptr))
    {}

    unique_ptr& operator=(unique_ptr&& other)
    {
        shared_ptr<T_ELEM>::container = std::exchange(other.shared_ptr<T_ELEM>::container, nullptr);
        return *this;
    }
};

struct Stream {

public:
    cudaError_t error_code = cudaSuccess;

private:
    StreamPriority priority;
    std::shared_ptr<inner_implementation::StreamPtrWrapper> c_stream;

public:
    Stream(StreamPriority priority) {
        this->priority = priority;
        create();
    }

    ~Stream(){
        c_stream.reset();
    }

    Stream(const Stream& other) = default;
    Stream& operator=(Stream const&other) = default;

    Stream(Stream&& other) noexcept:
        priority(other.priority),
        c_stream(std::exchange(other.c_stream, nullptr))
    {}

    Stream& operator=(Stream&& other)
    {
        priority = other.priority;
        c_stream = std::exchange(other.c_stream, nullptr);
        return *this;
    }

    cudaStream_t ptr() {
        return c_stream ? c_stream->ptr : nullptr;
    }

    StreamPriority get_priority() const {
        return priority;
    }

    int priority_num() const {
        return (c_stream.get() ? c_stream->priority_num : 0);
    }

    void waitForCompletion(){
        if(c_stream)
            error_code = cudaStreamSynchronize(c_stream.get()->ptr);
    }

private:
    void create(){
        c_stream = std::make_shared<inner_implementation::StreamPtrWrapper>(this->priority);
    }
};
}

#endif // CUDASHAREDPTR_H
