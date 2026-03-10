#pragma once

#include "painter.hpp"
#include "constants.hpp"
#include "util/config.hpp"
#include "util/BitConverter.hpp"
#include "codec/interleaver.hpp"

#include <new>
#include <vector>
#include <cstdint>

inline bool is_calibration_cell(int r, int c){
    return r == Config::CALIB_ROW && c >= Config::CALIB_COL_BEGIN && c < Config::CALIB_COL_END;
}

void write_6bits_data(cv::Mat& img, uint8_t* data, size_t size){
    if(size != Config::UINTS_COUNT) return; // TODO: throw an exception

    Painter painter(img);

    // 写入前，先交织编码（实则是随机打散）
    Interleaver::get_instance().interleave(data, size);

    int id = 0;
    for(int r = 0; r < Config::GRID_R; ++r){
        for(int c = 0; c < Config::GRID_C; ++c){
            if(Painter::is_reserved(r, c)) continue;
            if(is_calibration_cell(r, c)){
                const uint8_t color_idx = static_cast<uint8_t>((c - Config::CALIB_COL_BEGIN) & 0x3);
                const uint8_t symbol = static_cast<uint8_t>((color_idx << P_BITS) | 0);
                painter.draw_tile(symbol, r, c);
                continue;
            }
            painter.draw_tile(data[id++], r, c);
        }
    }
}

void write_6bits_data(cv::Mat& img, std::vector<uint8_t>& data){
    write_6bits_data(img, data.data(), data.size());
}

void write_8bits_data(cv::Mat& img, uint8_t* data, size_t size){
    size_t data_6bits_size = Config::UINTS_COUNT;
    uint8_t* data_6bits = new uint8_t[data_6bits_size];

    size_t res = BitConverter::convert_826(data, size, data_6bits, data_6bits_size);
    if(res != Config::UINTS_COUNT) return; // TODO: throw an exception
    
    write_6bits_data(img, data_6bits, data_6bits_size);
    delete[] data_6bits;
}

void write_8bits_data(cv::Mat& img, std::vector<uint8_t>& data){
    write_8bits_data(img, data.data(), data.size());
}