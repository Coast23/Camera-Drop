#pragma once
#include <cmath>
#include <string>
#include <cstdint>
#include <stdexcept>

class Config {
public:
    // 图像基本配置
    static const int IMG_WIDTH  = 1024;
    static const int IMG_HEIGHT = 1024;
    static const int STRIDE     = 9;
    static const int MARGIN     = 8;

    static constexpr int GRID_R = (IMG_HEIGHT - MARGIN * 2) / STRIDE;
    static constexpr int GRID_C = (IMG_WIDTH - MARGIN * 2)  / STRIDE;

    static const uint32_t BITS_PER_UNIT = 6;                      // 每个图案单元能编码的位数 
    static constexpr uint32_t UINTS_COUNT = GRID_R * GRID_C - 4 * 6 * 6;  // 一帧的图案单元数
    static constexpr uint32_t UNITS_PER_BYTE =                    // 每个字节能编码多少图案单元
                                        8 / BITS_PER_UNIT;
    static constexpr uint32_t PACKET_CAPACITY =                   // 数据包容量（字节）
                                        UINTS_COUNT * BITS_PER_UNIT / 8;   

    // 动态参数，运行时计算
    inline static uint32_t RS_BLOCK_SIZE  = 0;  // RS 块大小
    inline static uint32_t RS_DATA_SIZE   = 0;  // RS 数据字节数
    inline static uint32_t RS_PARITY_SIZE = 0;  // RS 校验字节数
    
    inline static uint32_t RS_BLOCKS_PER_FOUNTAIN_CHUNK = 0;
                        // 这个值必须保证 Packet Count <= 64000！不然会运行错误

    inline static uint32_t FOUNTAIN_PAYLOAD_SIZE = 0;
    inline static uint32_t RS_BLOCKS_PER_FRAME = 0;
    inline static uint32_t FOUNTAIN_PACKETS_PER_FRAME = 0;
    inline static uint32_t FOUNTAIN_CHUNK_SIZE = 0;

    static const uint32_t FOUNTAIN_HEADER_SIZE = 10;  // 帧头大小 file_size(4) + original_size(4) + block_id(2)
    static const uint32_t FOUNTAIN_CRC_SIZE = 4;      // 帧尾 CRC 大小

    static constexpr uint32_t MAX_FILE_SIZE = 60 * 1024 * 1024; // 限制文件大小不超过 60 MB
    
    inline static float REDUNDANCY_FACTOR = 1.01f;  // 冗余系数，不要设太大，优先调 RS 的 ECC 比例！

    // 动态参数，可由命令行覆盖
    inline static int COMPRESSION_LEVEL = 9;        // Zstd 压缩等级
    inline static int OUTPUT_FPS = 30;              // 视频输出帧率
    inline static std::string INPUT_VIDEO_FILE = "";
    inline static std::string OUTPUT_VIDEO_FILE = "output.avi";
    inline static std::string VOUT_FILE = "vout.bin";

private:
    static double binomial_cdf(int n, int k, double p){
        if(p <= 0.0) return 1.0;
        if(p >= 1.0) return (k == n) ? 1.0 : 0.0;
        double cdf = 0.0;
        for(int i = 0; i <= k; ++i){
            double log_term = std::lgamma(n + 1) - std::lgamma(i + 1) - std::lgamma(n - i + 1)
                            + i * std::log(p) + (n - i) * std::log(1 - p);
            cdf += std::exp(log_term);
        }
        return cdf;
    }

public:
    static void auto_config(double acc = 0.95){
        if(acc >= 0.9999999) acc = 0.9990;
        if(acc <= 0.0000001) acc = 0.0001;
        double p = 1.0 - acc;
        // 选择能被整除的 RS_BLOCK_SIZE
        uint32_t best_N = 255, min_padding = PACKET_CAPACITY;
        for(uint32_t n = 255; n >= 120; --n){
            if(PACKET_CAPACITY % n < min_padding){
                min_padding = PACKET_CAPACITY % n;
                best_N = n;
                if(!min_padding) break;
            }
        }
        RS_BLOCK_SIZE = best_N;
        RS_BLOCKS_PER_FRAME = PACKET_CAPACITY / RS_BLOCK_SIZE;

        // 根据 RS 纠错能力计算 ECC。利用 3-sigma 原则计算
        double miu = best_N * p;
        double sigma = std::sqrt(best_N * p * (1 - p));
        int E = static_cast<int>(std::ceil( miu + 3.05 * sigma));
        
        if((E << 1) >= RS_BLOCK_SIZE){ // Oops
            throw std::runtime_error("The acc is too low to transmit data!");
        }

        RS_PARITY_SIZE = E << 1;
        RS_DATA_SIZE = RS_BLOCK_SIZE - RS_PARITY_SIZE;
    
        // 寻找最优 喷泉块:RS块 的比例 M
        uint32_t min_payload = (MAX_FILE_SIZE + 63999) / 64000 + FOUNTAIN_HEADER_SIZE + FOUNTAIN_CRC_SIZE;
        uint32_t M_min = (min_payload + RS_DATA_SIZE - 1) / RS_DATA_SIZE;
        if(M_min == 0) M_min = 1;

        // 找一个 >= M_min 且能整除 RS_BLOCKS_PER_FRAME 的 M 即可
        uint32_t best_M = M_min;
        for(uint32_t m = M_min; m <= RS_BLOCKS_PER_FRAME; ++m){
            if(RS_BLOCKS_PER_FRAME % m == 0){
                best_M = m; break;
            }
        }
        RS_BLOCKS_PER_FOUNTAIN_CHUNK = best_M;
        
        // 计算各个容量
        FOUNTAIN_PAYLOAD_SIZE = RS_BLOCKS_PER_FOUNTAIN_CHUNK * RS_DATA_SIZE;
        FOUNTAIN_PACKETS_PER_FRAME = RS_BLOCKS_PER_FRAME / RS_BLOCKS_PER_FOUNTAIN_CHUNK; // TODO: 能保证整除吗？
        FOUNTAIN_CHUNK_SIZE = FOUNTAIN_PAYLOAD_SIZE - FOUNTAIN_HEADER_SIZE - FOUNTAIN_CRC_SIZE;
        
        // 计算 Fountain 冗余系数
        double P_rs = binomial_cdf(RS_BLOCK_SIZE, E, p);
        double P_fm = std::pow(P_rs, RS_BLOCKS_PER_FOUNTAIN_CHUNK);
        REDUNDANCY_FACTOR = static_cast<float>(1.05 / P_fm);
        
        show_config(acc);
    }

    static void show_config(const double acc){

        uint32_t effective_bytes_per_frame = FOUNTAIN_CHUNK_SIZE * FOUNTAIN_PACKETS_PER_FRAME;
        double efficiency = (double)effective_bytes_per_frame / PACKET_CAPACITY * 100.0;
        double actual_max_mb = (64000.0 * FOUNTAIN_CHUNK_SIZE) / (1024.0 * 1024.0);

        printf("---------------------------------------\n");
        printf("Target Accuracy: %.2f%%\n", acc * 100.0);
        printf("RS Config: N = %u, K = %u, P = %u\n", RS_BLOCK_SIZE, RS_DATA_SIZE, RS_PARITY_SIZE);
        printf("Fountain:  M = %u blocks/chunk, %u chunks/frame\n", RS_BLOCKS_PER_FOUNTAIN_CHUNK, FOUNTAIN_PACKETS_PER_FRAME);
        printf("Redundancy factor: %.3f\n", REDUNDANCY_FACTOR);
        printf("Frame Capacity: %u bytes\n", PACKET_CAPACITY);
        printf("Effective Data: %u bytes/frame (%.2f%%)\n", effective_bytes_per_frame, efficiency);
        printf("Max File Size : %.2f MB\n", actual_max_mb);
        printf("---------------------------------------\n");
    }
};

