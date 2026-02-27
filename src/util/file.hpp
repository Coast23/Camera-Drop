// Author: Coast23
// Date: 2026-02-27
/*
提供文件读取与写入函数
*/
#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <fstream>

std::vector<uint8_t> read_file(const std::string& path){
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if(!file) return {};
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> buffer(size);
    if(file.read(reinterpret_cast<char*>(buffer.data()), size)) return buffer;
    return {};
}

void write_file(const std::string& path, const std::vector<uint8_t>& data){
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(data.data()), data.size());
}