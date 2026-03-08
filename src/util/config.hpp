#pragma once
#include <string>
#include <cstdint>

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
    static constexpr uint32_t UINTS_COUNT = GRID_R * GRID_C - 4;  // 一帧的图案单元数
    static constexpr uint32_t UNITS_PER_BYTE =                    // 每个字节能编码多少图案单元
                                        8 / BITS_PER_UNIT;
    static constexpr uint32_t PACKET_CAPACITY =                   // 数据包容量（字节）
                                        UINTS_COUNT * BITS_PER_UNIT / 8;   

    /*
    固定 RS 块大小为 186
    有效 64，冗余 122，能抗 23% 左右的随机误码率
    有效 32，冗余 154，能抗 31% 左右的随机误码率
    有效 16，冗余 170，能抗 35% 左右的随机误码率
    */
    static const uint32_t RS_DATA_SIZE   = 130;     // RS 数据字节数
    static const uint32_t RS_PARITY_SIZE = 30;      // RS 校验字节数
    static constexpr uint32_t RS_BLOCK_SIZE =       // RS 块大小
                                        RS_DATA_SIZE + RS_PARITY_SIZE;

    static const uint32_t RS_BLOCKS_PER_FOUNTAIN_CHUNK = 8;

    static constexpr uint32_t FOUNTAIN_PAYLOAD_SIZE =
                            RS_BLOCKS_PER_FOUNTAIN_CHUNK * RS_DATA_SIZE;
    
    static constexpr uint32_t RS_BLOCKS_PER_FRAME = PACKET_CAPACITY / RS_BLOCK_SIZE;
    static constexpr uint32_t FOUNTAIN_PACKETS_PER_FRAME = RS_BLOCKS_PER_FRAME / RS_BLOCKS_PER_FOUNTAIN_CHUNK;

  //  static constexpr uint32_t FOUNTAIN_PAYLOAD_SIZE = // 有效载荷（不含 ECC）大小
  //                              PACKET_CAPACITY / RS_BLOCK_SIZE * RS_DATA_SIZE;
    static const uint32_t FOUNTAIN_HEADER_SIZE = 10;  // 帧头大小 file_size(4) + original_size(4) + block_id(2)
    static const uint32_t FOUNTAIN_CRC_SIZE = 4;      // 帧尾 CRC 大小
    static constexpr uint32_t FOUNTAIN_CHUNK_SIZE =   // 块大小
                                FOUNTAIN_PAYLOAD_SIZE - FOUNTAIN_HEADER_SIZE - FOUNTAIN_CRC_SIZE;

    static constexpr uint32_t MAX_FILE_SIZE = 200 * 1024 * 1024; // 限制文件大小不超过 200 MB
    
    // 动态参数，可由命令行覆盖
    inline static float REDUNDANCY_FACTOR = 1.5f;   // 冗余系数
    inline static int COMPRESSION_LEVEL = 9;        // Zstd 压缩等级
    inline static int OUTPUT_FPS = 15;              // 视频输出帧率
    inline static std::string INPUT_VIDEO_FILE = "";
    inline static std::string OUTPUT_VIDEO_FILE = "output.avi";
    inline static std::string VOUT_FILE = "vout.bin";
};

