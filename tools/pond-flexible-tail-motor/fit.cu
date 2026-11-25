#include "core/checkpoint.h"
#include "nn/adam.h"
#include "nn/loss.h"
#include "nn/mlp_gpu.h"
#include "rl/flexible_tail_motor.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace motor = cudarl::flexible_tail;
namespace nn = cudarl::nn;
using Clock = std::chrono::steady_clock;

constexpr std::uint32_t seed = 20261124;
constexpr int max_rows = 32768;
constexpr int batch_limit = 512;
constexpr int optimizer_steps = 384;
constexpr float learning_rate = 0.001f;

struct Examples {
    int rows;
    std::vector<float> inputs;
    std::vector<float> targets;
};

Examples read_examples(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    const auto length = input.tellg();
    std::uint32_t rows = 0, width = 0;
    input.seekg(0);
    input.read(reinterpret_cast<char*>(&rows), sizeof rows);
    input.read(reinterpret_cast<char*>(&width), sizeof width);
    const auto byte_count = 2 * sizeof(std::uint32_t) +
        std::uint64_t(rows) * (motor::motor_inputs + motor::motor_outputs) * sizeof(float);
    if (!input || rows == 0 || rows > max_rows || width != motor::motor_inputs ||
        length < 0 || static_cast<std::uint64_t>(length) != byte_count)
        throw std::runtime_error("Invalid motor examples file size or dimensions");

    const auto read_values = [&](int dimensions) {
        std::vector<float> values(std::size_t(rows) * dimensions);
        input.read(reinterpret_cast<char*>(values.data()),
                   static_cast<std::streamsize>(values.size() * sizeof(float)));
        if (!input || !std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); }))
            throw std::runtime_error("Invalid motor examples float values");
        return values;
    };
    const auto inputs = read_values(motor::motor_inputs);
    const auto targets = read_values(motor::motor_outputs);
    if (!std::all_of(targets.begin(), targets.end(), [](float value) { return std::abs(value) <= 1.f; }))
        throw std::runtime_error("Motor targets must be normalized to [-1, 1]");

    // One seeded permutation mixes the offline fixtures before contiguous
    // minibatches are uploaded. The split below keeps batch sizes balanced.
    std::vector<std::size_t> order(rows);
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::mt19937 generator(seed);
    std::shuffle(order.begin(), order.end(), generator);
    Examples result{static_cast<int>(rows), std::vector<float>(inputs.size()), std::vector<float>(targets.size())};
    for (std::size_t i = 0; i < rows; ++i) {
        std::copy_n(inputs.data() + order[i] * motor::motor_inputs, motor::motor_inputs,
                    result.inputs.data() + i * motor::motor_inputs);
        std::copy_n(targets.data() + order[i] * motor::motor_outputs, motor::motor_outputs,
                    result.targets.data() + i * motor::motor_outputs);
    }
    return result;
}

double prediction_mse(nn::GpuMlp& model, const float* inputs, const Examples& examples) {
    std::vector<float> prediction(std::size_t(model.max_batch()) * motor::motor_outputs);
    double squared_error = 0;
    for (int offset = 0; offset < examples.rows; offset += model.max_batch()) {
        const int rows = std::min(model.max_batch(), examples.rows - offset);
        const int values = rows * motor::motor_outputs;
        model.forward(inputs + std::size_t(offset) * motor::motor_inputs, rows);
        CUDA_CHECK_THROW(cudaMemcpy(prediction.data(), model.output(),
                                    std::size_t(values) * sizeof(float), cudaMemcpyDeviceToHost));
        const float* expected = examples.targets.data() + std::size_t(offset) * motor::motor_outputs;
        for (int i = 0; i < values; ++i) {
            const double difference = double(prediction[i]) - expected[i];
            squared_error += difference * difference;
        }
    }
    const auto mse = squared_error / examples.targets.size();
    if (!std::isfinite(mse)) throw std::runtime_error("Motor fitting produced nonfinite predictions");
    return mse;
}

struct Event {
    cudaEvent_t handle = nullptr;
    Event() { CUDA_CHECK_THROW(cudaEventCreate(&handle)); }
    ~Event() { cudaEventDestroy(handle); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};

void fit(const std::string& source, const std::string& destination) {
    const auto start = Clock::now();
    if (std::filesystem::exists(std::filesystem::u8path(destination)))
        throw std::runtime_error("Motor output must be a new path");
    const auto examples = read_examples(std::filesystem::u8path(source));
    int device = 0;
    cudaDeviceProp properties{};
    CUDA_CHECK_THROW(cudaGetDevice(&device));
    CUDA_CHECK_THROW(cudaGetDeviceProperties(&properties, device));
    const int batches = (examples.rows + batch_limit - 1) / batch_limit;
    const int capacity = (examples.rows + batches - 1) / batches;
    nn::GpuMlp model(motor::motor_spec(), capacity, seed);
    nn::AdamConfig config;
    config.lr = learning_rate;
    nn::GpuAdam optimizer(model.num_params(), config);
    device_ptr<float> inputs(examples.inputs.size()), targets(examples.targets.size());
    device_ptr<float> output_gradient(std::size_t(capacity) * motor::motor_outputs);
    inputs.copy_from_host(examples.inputs.data(), examples.inputs.size());
    targets.copy_from_host(examples.targets.data(), examples.targets.size());
    const double initial_mse = prediction_mse(model, inputs.get(), examples);
    std::cout << std::setprecision(9) << "device=" << properties.name
              << " rows=" << examples.rows << " initial_mse=" << initial_mse << std::endl;

    Event begin, end;
    CUDA_CHECK_THROW(cudaEventRecord(begin.handle));
    for (int step = 0; step < optimizer_steps; ++step) {
        const int batch = step % batches;
        const int offset = batch * examples.rows / batches;
        const int rows = (batch + 1) * examples.rows / batches - offset;
        model.zero_grad();
        model.forward(inputs.get() + std::size_t(offset) * motor::motor_inputs, rows);
        nn::mse_loss_grad(model.output(), targets.get() + std::size_t(offset) * motor::motor_outputs,
                          output_gradient.get(), rows * motor::motor_outputs);
        model.backward(output_gradient.get(), rows);
        optimizer.step(model.params(), model.grads());
    }
    CUDA_CHECK_THROW(cudaEventRecord(end.handle));
    CUDA_CHECK_THROW(cudaEventSynchronize(end.handle));
    float fit_ms = 0;
    CUDA_CHECK_THROW(cudaEventElapsedTime(&fit_ms, begin.handle, end.handle));

    const double final_mse = prediction_mse(model, inputs.get(), examples);
    motor::write_motor_weights(destination, model.get_params());
    const auto wall_s = std::chrono::duration<double>(Clock::now() - start).count();
    std::cout << "optimizer_steps=" << optimizer.steps() << " final_mse=" << final_mse
              << " optimizer_cuda_ms=" << fit_ms << " wall_s=" << wall_s << '\n';
}
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: pond_flexible_tail_motor_fit <examples.bin> <new actor.bin>\n";
        return 1;
    }
    try {
        fit(argv[1], argv[2]);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
