#pragma once

#include <string>
#include <exception>
#include <stdexcept>

// 基础异常类
class CameraDropError : public std::exception {
private:
    std::string _msg;
public:
    explicit CameraDropError(const std::string& msg) : _msg(msg) {}
    virtual const char* what() const noexcept override {
        return _msg.c_str();
    }
};

// 文件操作相关异常
class FileError : public CameraDropError {
public:
    explicit FileError(const std::string& msg) : CameraDropError("FileError: " + msg) {}
};

class FileOpenError : public FileError {
public:
    explicit FileOpenError(const std::string& filename) 
        : FileError("Failed to open file: " + filename) {}
};

class FileReadError : public FileError {
public:
    explicit FileReadError(const std::string& filename) 
        : FileError("Failed to read file: " + filename) {}
};

class FileWriteError : public FileError {
public:
    explicit FileWriteError(const std::string& filename) 
        : FileError("Failed to write file: " + filename) {}
};

class FileSizeError : public FileError {
public:
    explicit FileSizeError(const std::string& msg) 
        : FileError(msg) {}
};

class FileNotFoundError : public FileError {
public:
    explicit FileNotFoundError(const std::string& filename) 
        : FileError("File not found: " + filename) {}
};

// 编码器相关异常
class EncoderError : public CameraDropError {
public:
    explicit EncoderError(const std::string& msg) : CameraDropError("EncoderError: " + msg) {}
};

class EncoderInitError : public EncoderError {
public:
    explicit EncoderInitError(const std::string& msg) : EncoderError(msg) {}
};

class EncoderRuntimeError : public EncoderError {
public:
    explicit EncoderRuntimeError(const std::string& msg) : EncoderError(msg) {}
};

// 解码器相关异常
class DecoderError : public CameraDropError {
public:
    explicit DecoderError(const std::string& msg) : CameraDropError("DecoderError: " + msg) {}
};

class DecoderInitError : public DecoderError {
public:
    explicit DecoderInitError(const std::string& msg) : DecoderError(msg) {}
};

class DecoderRuntimeError : public DecoderError {
public:
    explicit DecoderRuntimeError(const std::string& msg) : DecoderError(msg) {}
};

class DecodeCompleteError : public DecoderError {
public:
    explicit DecodeCompleteError(const std::string& msg) : DecoderError(msg) {}
};

// 压缩相关异常
class CompressionError : public CameraDropError {
public:
    explicit CompressionError(const std::string& msg) : CameraDropError("CompressionError: " + msg) {}
};

class CompressError : public CompressionError {
public:
    explicit CompressError(const std::string& msg) : CompressionError("Failed to compress: " + msg) {}
};

class DecompressError : public CompressionError {
public:
    explicit DecompressError(const std::string& msg) : CompressionError("Failed to decompress: " + msg) {}
};

// 喷泉码相关异常
class FountainError : public CameraDropError {
public:
    explicit FountainError(const std::string& msg) : CameraDropError("FountainError: " + msg) {}
};

class FountainEncodeError : public FountainError {
public:
    explicit FountainEncodeError(const std::string& msg) : FountainError("Encode failed: " + msg) {}
};

class FountainDecodeError : public FountainError {
public:
    explicit FountainDecodeError(const std::string& msg) : FountainError("Decode failed: " + msg) {}
};

class FountainInitError : public FountainError {
public:
    explicit FountainInitError(const std::string& msg) : FountainError("Init failed: " + msg) {}
};

// Reed-Solomon 相关异常
class RSError : public CameraDropError {
public:
    explicit RSError(const std::string& msg) : CameraDropError("RSError: " + msg) {}
};

class RSEncodeError : public RSError {
public:
    explicit RSEncodeError(const std::string& msg) : RSError("Encode failed: " + msg) {}
};

class RSDecodeError : public RSError {
public:
    explicit RSDecodeError(const std::string& msg) : RSError("Decode failed: " + msg) {}
};

class RSInitError : public RSError {
public:
    explicit RSInitError(const std::string& msg) : RSError("Init failed: " + msg) {}
};

// 数据包相关异常
class DataPacketError : public CameraDropError {
public:
    explicit DataPacketError(const std::string& msg) : CameraDropError("DataPacketError: " + msg) {}
};

class DataPacketSerializeError : public DataPacketError {
public:
    explicit DataPacketSerializeError(const std::string& msg) : DataPacketError("Serialize failed: " + msg) {}
};

class DataPacketDeserializeError : public DataPacketError {
public:
    explicit DataPacketDeserializeError(const std::string& msg) : DataPacketError("Deserialize failed: " + msg) {}
};

class DataPacketCRCError : public DataPacketError {
public:
    explicit DataPacketCRCError(const std::string& msg) : DataPacketError("CRC check failed: " + msg) {}
};

// 图像处理相关异常
class ImageError : public CameraDropError {
public:
    explicit ImageError(const std::string& msg) : CameraDropError("ImageError: " + msg) {}
};

class ImageSizeError : public ImageError {
public:
    explicit ImageSizeError(const std::string& msg) : ImageError("Size mismatch: " + msg) {}
};

class ImageFormatError : public ImageError {
public:
    explicit ImageFormatError(const std::string& msg) : ImageError("Format error: " + msg) {}
};

class ImageWriteError : public ImageError {
public:
    explicit ImageWriteError(const std::string& msg) : ImageError("Write failed: " + msg) {}
};

// 图案字典相关异常
class PatternDictError : public CameraDropError {
public:
    explicit PatternDictError(const std::string& msg) : CameraDropError("PatternDictError: " + msg) {}
};

class PatternDictLoadError : public PatternDictError {
public:
    explicit PatternDictLoadError(const std::string& msg) : PatternDictError("Load failed: " + msg) {}
};

class PatternDictInvalidError : public PatternDictError {
public:
    explicit PatternDictInvalidError(const std::string& msg) : PatternDictError("Invalid dictionary: " + msg) {}
};

// 交织器相关异常
class InterleaverError : public CameraDropError {
public:
    explicit InterleaverError(const std::string& msg) : CameraDropError("InterleaverError: " + msg) {}
};

class InterleaverSizeError : public InterleaverError {
public:
    explicit InterleaverSizeError(const std::string& msg) : InterleaverError("Size mismatch: " + msg) {}
};

// 位转换相关异常
class BitConverterError : public CameraDropError {
public:
    explicit BitConverterError(const std::string& msg) : CameraDropError("BitConverterError: " + msg) {}
};

class BitConverterSizeError : public BitConverterError {
public:
    explicit BitConverterSizeError(const std::string& msg) : BitConverterError("Size mismatch: " + msg) {}
};

// 配置相关异常
class ConfigError : public CameraDropError {
public:
    explicit ConfigError(const std::string& msg) : CameraDropError("ConfigError: " + msg) {}
};

class ConfigInvalidError : public ConfigError {
public:
    explicit ConfigInvalidError(const std::string& msg) : ConfigError("Invalid config: " + msg) {}
};

class ConfigRangeError : public ConfigError {
public:
    explicit ConfigRangeError(const std::string& msg) : ConfigError("Value out of range: " + msg) {}
};

// 视觉处理相关异常（非ONNX相关）
class VisionError : public CameraDropError {
public:
    explicit VisionError(const std::string& msg) : CameraDropError("VisionError: " + msg) {}
};

class VisionInitError : public VisionError {
public:
    explicit VisionInitError(const std::string& msg) : VisionError("Init failed: " + msg) {}
};

class VisionProcessingError : public VisionError {
public:
    explicit VisionProcessingError(const std::string& msg) : VisionError("Processing failed: " + msg) {}
};

class VisionDeskewError : public VisionError {
public:
    explicit VisionDeskewError(const std::string& msg) : VisionError("Deskew failed: " + msg) {}
};

// 扫描相关异常（保留原有）
class ScannerError : public CameraDropError {
public:
    explicit ScannerError(const std::string& msg) : CameraDropError("ScannerError: " + msg) {}
};
