#include "codec/Encoder.hpp"
#include "codec/Decoder.hpp"
#include "util/config.hpp"
#include "util/parallel.hpp"

#include <atomic>
#include <cstdio>
#include <vector>
#include <random>
#include <chrono>
#include <string>
#include <cassert>
#include <algorithm>

namespace {

struct Options {
    int threads = 0;
    bool full = false;
};

void print_usage() {
    puts("Usage: debug [--threads <n>] [--full]");
}

Options parse_args(int argc, char** argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--threads" && i + 1 < argc) {
            opts.threads = std::stoi(argv[++i]);
        } else if (arg == "--full") {
            opts.full = true;
        } else {
            print_usage();
            std::exit(1);
        }
    }
    return opts;
}

}  // namespace

std::vector<uint8_t> generate_data(size_t size){
    std::vector<uint8_t> data(size);
    std::mt19937 rng;
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, 255);
    std::generate(data.begin(), data.end(), 
        [&](){
            return static_cast<uint8_t>(dist(rng));
        }
    );
    return data;
}

typedef std::vector<uint8_t> Packet;

class VideoChannel {
public:
    VideoChannel(const double lr = 0.0, const double er = 0.00) : loss_rate_(lr), error_rate_(er), total_frame_(0), lossed_frame_(0) {}
    void trans(const Packet& data){
        auto raw = data;
        ++total_frame_;
        
        static thread_local std::mt19937 rng(
            std::chrono::high_resolution_clock::now()
                .time_since_epoch().count()
        );

        std::uniform_real_distribution<double> prob(0.0, 1.0);

        if(prob(rng) < loss_rate_){      // 模拟丢包
            ++lossed_frame_;
            return;
        }

        for(size_t i = 0; i < raw.size(); ++i){
            if(prob(rng) < error_rate_){ // 简单模拟 flip
                raw[i] = ~raw[i];
            }
        }
        
        packets_.push_back(raw);
    }

    std::vector<Packet> recieved() const {
        return packets_;
    }

    int total_frame() const {return total_frame_;}
    int lossed_frame() const {return lossed_frame_;}

private:
    std::vector<Packet> packets_;
    int total_frame_;
    int lossed_frame_;
    double loss_rate_;
    double error_rate_;
};

int main(int argc, char** argv){
    const Options opts = parse_args(argc, argv);
  //  Config::auto_config(0.90);
    for(double i = 0.90; i <= 0.991; i += 0.01) Config::auto_config(i);
    if (!opts.full) {
        return 0;
    }
    puts("getting encoder...");

    auto data = generate_data(1024 * 1024 * 10); // 10 MB
  //  std::string str = "hello world.";
  //  std::vector<uint8_t> data(str.begin(), str.end());

    const char* inFile = "in.bin";
    const char* outFile = "out.bin";

    FILE* f = fopen(inFile, "wb");
    fwrite(data.data(), 1, data.size(), f);
    fclose(f);

    Encoder encoder(inFile);
    assert(encoder.is_valid());

    puts("Ready to send.");

    VideoChannel channel(0.02, 0.10);

  //  const uint32_t packet_count = encoder.packet_count_recommended();
    const uint32_t packet_count = std::max(encoder.packet_count_recommended(), 10u);

    uint32_t generate_frames = (
        packet_count + Config::FOUNTAIN_PACKETS_PER_FRAME - 1
        ) / Config::FOUNTAIN_PACKETS_PER_FRAME;

    printf("Packet Count: %u\n", packet_count);
    printf("Video Frames: %u\n", generate_frames);

    /*
    for(uint32_t i = 0; i < packet_count; ++i){
        auto packet = encoder.get_packet();
        printf("packet size: %u\n", packet.size());
        channel.trans(packet);
    }*/
    std::vector<Packet> frames;
    frames.reserve(generate_frames);
    for(uint32_t i = 0; i < generate_frames; ++i){
        auto frame_data = encoder.get_packet();
        if(!i){
            printf("Frame capacity: %zu bytes\n", frame_data.size());
        }
        frames.push_back(std::move(frame_data));
    }

    const size_t threads = camdrop::util::resolve_thread_count(opts.threads);
    std::vector<Packet> received(frames.size());
    std::vector<uint8_t> received_ok(frames.size(), 0);
    std::atomic<int> lossed{0};

    camdrop::util::parallel_for(frames.size(), threads, [&](size_t i) {
        static thread_local std::mt19937 rng(
            static_cast<unsigned int>(
                std::chrono::high_resolution_clock::now().time_since_epoch().count() ^
                (reinterpret_cast<std::uintptr_t>(&rng) >> 4)));
        std::uniform_real_distribution<double> prob(0.0, 1.0);
        Packet raw = frames[i];
        if (prob(rng) < 0.02) {
            lossed.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        for (size_t k = 0; k < raw.size(); ++k) {
            if (prob(rng) < 0.10) {
                raw[k] = ~raw[k];
            }
        }
        received[i] = std::move(raw);
        received_ok[i] = 1;
    });

    std::vector<Packet> recieved;
    recieved.reserve(frames.size());
    for (size_t i = 0; i < received_ok.size(); ++i) {
        if (received_ok[i]) {
            recieved.push_back(std::move(received[i]));
        }
    }

    Decoder decoder;
    int cnt = 0;

    printf("Start decoding...\n");

    /*
    for(auto& packet : recieved){
        bool res = decoder.process_packet(packet);
        if(res) ++cnt;
        if(decoder.is_complete()){
            puts("Decode complete!");
            decoder.save_to_file(outFile);
            break;
        }
    }*/
    for(auto& frame_data : recieved){
        decoder.process_packet(frame_data);
        ++cnt;
        if(decoder.is_complete()){
            puts("Decode complete!");
            decoder.save_to_file(outFile);
            puts("Save failed.");
            break;
        }
    }

    if(!decoder.is_complete()) puts("Decode failed.");
    else puts("Decode success!");
    printf("sent: %zu, loss: %d, processed: %d\n", frames.size(), lossed.load(), cnt);
}
