#include "stable-diffusion.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/resource.h>

using Clock = std::chrono::steady_clock;

static double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

static long peak_rss_bytes() {
    rusage usage{};
    return getrusage(RUSAGE_SELF, &usage) == 0 ? usage.ru_maxrss * 1024L : 0;
}

static void log_runtime(sd_log_level_t level, const char* message, void*) {
    if (level >= SD_LOG_INFO) {
        std::cerr << message;
    }
}

static void save_ppm(const sd_image_t& image, const std::string& path) {
    if (image.channel != 3 || image.data == nullptr) {
        throw std::runtime_error("Unexpected image format from diffusion runtime");
    }
    std::ofstream file(path, std::ios::binary);
    file << "P6\n" << image.width << ' ' << image.height << "\n255\n";
    file.write(reinterpret_cast<const char*>(image.data), size_t(image.width) * image.height * 3);
    if (!file) {
        throw std::runtime_error("Unable to save generated image");
    }
}

static bool run_image(sd_ctx_t* context, const std::string& id, int64_t seed, const std::string& prompt,
                      const std::string& output, int size, int steps) {
    sd_img_gen_params_t params;
    sd_img_gen_params_init(&params);
    params.prompt = prompt.c_str();
    params.width = size;
    params.height = size;
    params.seed = seed;
    params.batch_count = 1;
    params.sample_params.sample_method = DPMPP2M_SAMPLE_METHOD;
    params.sample_params.scheduler = DISCRETE_SCHEDULER;
    params.sample_params.sample_steps = steps;
    params.sample_params.guidance.txt_cfg = 7.5f;
    params.vae_tiling_params.enabled = true;

    const auto start = Clock::now();
    sd_image_generation_result_t* result_ptr = nullptr;
    const bool generated = generate_image(context, &params, &result_ptr);
    const auto latency = elapsed_ms(start);
    const std::unique_ptr<sd_image_generation_result_t, decltype(&free_sd_image_generation_result)>
        result(result_ptr, free_sd_image_generation_result);
    if (!generated || !result || result->count != 1) {
        std::cout << "P3_ERROR\t" << id << "\n" << std::flush;
        return false;
    }
    if (!output.empty()) {
        save_ppm(result->items[0].image, output + "/" + id + ".ppm");
    }
    std::cout << "P3_RUN\t" << id << '\t' << latency << '\t' << peak_rss_bytes() << "\n" << std::flush;
    return true;
}

int main(int argc, char** argv) {
    if (argc != 7) {
        std::cerr << "Usage: p3-native MODEL MANIFEST_TSV OUTPUT_DIR SIZE STEPS BACKEND\n";
        return 2;
    }
    try {
        const int size = std::stoi(argv[4]);
        const int steps = std::stoi(argv[5]);
        if (size < 256 || size > 512 || size % 64 != 0 || steps < 1 || steps > 50) {
            throw std::invalid_argument("Invalid native resolution or step count");
        }
        std::ifstream manifest(argv[2]);
        if (!manifest) {
            throw std::runtime_error("Unable to open the prompt manifest");
        }
        sd_set_log_callback(log_runtime, nullptr);
        sd_ctx_params_t options;
        sd_ctx_params_init(&options);
        options.model_path = argv[1];
        options.backend = argv[6];
        options.n_threads = 4;
        options.enable_mmap = true;
        options.diffusion_flash_attn = true;
        options.conditioning_cache_size = 0;
        const auto started = Clock::now();
        const std::unique_ptr<sd_ctx_t, decltype(&free_sd_ctx)> context(new_sd_ctx(&options), free_sd_ctx);
        if (!context) {
            throw std::runtime_error("The runtime could not allocate or load the checkpoint");
        }
        std::cout << "P3_LOAD\t" << elapsed_ms(started) << '\t' << peak_rss_bytes()
                  << '\t' << sd_get_model_version_name(context.get()) << "\n" << std::flush;
        if (!run_image(context.get(), "warmup", 0, "A green pear on a white kitchen counter.",
                       "", size, steps)) {
            return 3;
        }
        std::string line;
        bool complete = true;
        while (std::getline(manifest, line)) {
            const auto first = line.find('\t');
            const auto second = line.find('\t', first == std::string::npos ? 0 : first + 1);
            if (first == std::string::npos || second == std::string::npos) {
                throw std::runtime_error("Malformed prompt manifest");
            }
            const auto id = line.substr(0, first);
            if (id.empty() || id.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
                              != std::string::npos) {
                throw std::runtime_error("Invalid prompt ID");
            }
            const auto seed = std::stoll(line.substr(first + 1, second - first - 1));
            complete = run_image(context.get(), id, seed, line.substr(second + 1), argv[3], size, steps) && complete;
        }
        return complete ? 0 : 3;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
