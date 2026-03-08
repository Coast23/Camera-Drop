/*
参数说明：目前这套配置是按 acc = 95% 配的，能达到 6.53 KB / Frame 的密度
*/

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
    static constexpr uint32_t UINTS_COUNT = GRID_R * GRID_C - 4 * 6 * 6;  // 一帧的图案单元数
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
    static const uint32_t RS_DATA_SIZE   = 148;     // RS 数据字节数
    static const uint32_t RS_PARITY_SIZE = 38;      // RS 校验字节数
    static constexpr uint32_t RS_BLOCK_SIZE =       // RS 块大小
                                        RS_DATA_SIZE + RS_PARITY_SIZE;
    /*  调参说明：
        设误码率为 p。
        设 RS 块大小为 N（即上面的RS_BLOCK_SIZE），
        那么，应该取 RS_PARITY_SIZE = 2 * ceil((N * p + 3 * sqrt(N * p * (1 - p))))，
        自然地，RS_DATA_SIZE = N - RS_PARITY_SIZE。
    */


    static const uint32_t RS_BLOCKS_PER_FOUNTAIN_CHUNK = 48; // TODO: 推导这个的最优值
                                            // 这个值必须保证 Packet Count <= 64000！不然会运行错误

    static constexpr uint32_t FOUNTAIN_PAYLOAD_SIZE =
                            RS_BLOCKS_PER_FOUNTAIN_CHUNK * RS_DATA_SIZE;
    
    static constexpr uint32_t RS_BLOCKS_PER_FRAME = PACKET_CAPACITY / RS_BLOCK_SIZE;
    static constexpr uint32_t FOUNTAIN_PACKETS_PER_FRAME = RS_BLOCKS_PER_FRAME / RS_BLOCKS_PER_FOUNTAIN_CHUNK;

  //  static constexpr uint32_t FOUNTAIN_PAYLOAD_SIZE = // 有效载荷（不含 ECC）大小
  //                              PACKET_CAPACITY / RS_BLOCK_SIZE * RS_DATA_SIZE;
    static const uint32_t FOUNTAIN_HEADER_SIZE = 12;  // 帧头大小 file_size(4) + original_size(4) + block_id(2)
    static const uint32_t FOUNTAIN_CRC_SIZE = 4;      // 帧尾 CRC 大小
    static constexpr uint32_t FOUNTAIN_CHUNK_SIZE =   // 块大小
                                FOUNTAIN_PAYLOAD_SIZE - FOUNTAIN_HEADER_SIZE - FOUNTAIN_CRC_SIZE;

    static constexpr uint32_t MAX_FILE_SIZE = 200 * 1024 * 1024; // 限制文件大小不超过 200 MB
    
    // 动态参数，可由命令行覆盖
    inline static float REDUNDANCY_FACTOR = 1.06f;  // 冗余系数，不要设太大，优先调 RS 的 ECC 比例！
    /*  调参说明：
        这个东西要用二项分布去算，非常麻烦。只能给一些经验值。
        如果 acc 在 95%，取 REDUNDANCY_FACTOR 为略高于 1.06f 的值（如 1.08f），
        如果 acc 在 90%，取 REDUNDANCY_FACTOR 为略高于 1.08f 的值（如 1.11f）。
        总之，这个值应该保持在 1.05f ~ 1.25f。
    */
    inline static int COMPRESSION_LEVEL = 9;        // Zstd 压缩等级
    inline static int OUTPUT_FPS = 15;              // 视频输出帧率
    inline static std::string INPUT_VIDEO_FILE = "";
    inline static std::string OUTPUT_VIDEO_FILE = "output.avi";
    inline static std::string VOUT_FILE = "vout.bin";
};

