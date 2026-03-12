#pragma once

#include "config.hpp"
#include "errors.hpp"
#include <string>
#include <vector>
#include <cstdint>
#include <fstream>

class FileReader {
public:
    explicit FileReader(const std::string& filename) : filename_(filename), file_(filename, std::ios::binary) {
        if (!file_.is_open()) {
            throw FileOpenError(filename);
        }
    }
    
    ~FileReader(){
        if(file_.is_open()) file_.close();
    }

    bool is_open() const {
        return file_.is_open();
    }
    
    // 读取 size 大小的数据（先无视 MAX_FILE_SIZE 限制）
    std::vector<uint8_t> read(size_t size){
        if (!file_.is_open()) {
            throw FileReadError(filename_);
        }
        
        std::vector<uint8_t> buffer(size);
        file_.read(reinterpret_cast<char*>(buffer.data()), size);
        size_t read_size = file_.gcount();
        buffer.resize(read_size);
        
        if (file_.bad()) {
            throw FileReadError(filename_);
        }
        
        return buffer;
    }

    size_t file_size(){
        if (!file_.is_open()) {
            throw FileReadError(filename_);
        }
        
        file_.seekg(0, std::ios::end);
        size_t size = file_.tellg();
        file_.seekg(0, std::ios::beg);
        
        if (file_.fail()) {
            throw FileReadError(filename_);
        }
        
        return size;
    }

    // 全部读入
    std::vector<uint8_t> read_all(){
        size_t size = file_size();

        if(size > Config::MAX_FILE_SIZE){
            throw FileSizeError("File size " + std::to_string(size) + 
                               " exceeds limit " + std::to_string(Config::MAX_FILE_SIZE));
        }
        
        auto data = read(size);
        if (size != 0 && data.empty()) {
            throw FileReadError(filename_);
        }
        
        return data;
    }

    bool eof() const {
        return file_.eof();
    }

    void reset(){
        if (!file_.is_open()) {
            throw FileReadError(filename_);
        }
        file_.clear();
        file_.seekg(0, std::ios::beg);
        
        if (file_.fail()) {
            throw FileReadError(filename_);
        }
    }

private:
    std::string filename_;
    std::ifstream file_;
};

class FileWriter {
public:
    explicit FileWriter(const std::string& filename) : filename_(filename), file_(filename, std::ios::binary) {
        if (!file_.is_open()) {
            throw FileOpenError(filename);
        }
    }
    
    ~FileWriter(){
        if(file_.is_open()) file_.close();
    }

    bool is_open() const {
        return file_.is_open();
    }

    // 写入 size 大小的数据
    void write(const uint8_t* data, size_t size){
        if (!file_.is_open()) {
            throw FileWriteError(filename_);
        }
        
        file_.write(reinterpret_cast<const char*>(data), size);
        
        if (!file_.good()) {
            throw FileWriteError(filename_);
        }
    }

    // 全部写入
    void write(const std::vector<uint8_t>& data){
        write(data.data(), data.size());
    }

private:
    std::string filename_;
    std::ofstream file_;
};
