// End-of-run actor feature ablation on recorded v6 batch 8. No physical stepping.
void v6_ablation(const fs::path &root) {
    constexpr int width = pp::signed_inputs;
    constexpr uint64_t source = 370545787127571638ull;
    const auto actor = [&](const char *name, uint64_t checksum) {
        auto data = cp::read((root / name).string());
        check(cp::checksum(data) == checksum, "Unexpected v6 checkpoint");
        return pp::read_candidate(data, 4517507952728352494ull, 16817206019680038892ull, source, 1384, 368,
                                  width)
            .actor;
    };
    const auto init = actor("init/candidate-0.bin", 0x8238e861ca94e0b7ull);
    const auto weights = actor("updates-1-8/candidate-8.bin", 0xcf80b83efa6e6a91ull);
    nn::CpuMlp<float> model(
        {{width, 32, 32, 4}, {nn::Activation::Tanh, nn::Activation::Tanh, nn::Activation::Identity}});
    model.params() = weights;
    double l2 = 0, maximum_weight = 0;
    for (int h = 0; h < 32; ++h) {
        const int index = h * width + width - 1;
        check(init[index] == 0, "Initial feature weights not zero");
        l2 += double(weights[index]) * weights[index];
        maximum_weight = std::max(maximum_weight, std::abs(double(weights[index])));
    }
    const auto batch = cp::read((root / "updates-1-8/batch-8.bin").string());
    check(cp::checksum(batch) == 0x366965649ff2c4aeull, "Unexpected archive");
    cp::Reader in{batch};
    check(in.value<uint32_t>() == 0x31425450 && in.value<uint32_t>() == 3 && in.value<uint32_t>() == 8 &&
              in.value<uint32_t>() == steps,
          "Unexpected v6 batch header");
    in.value<uint64_t>();
    check(in.value<uint64_t>() == source, "Wrong v6 source");
    array<uint8_t>(in, 1384 * worlds);
    array<uint8_t>(in, 368 * worlds);
    array<float>(in, pp::actor_parameter_count(width));
    array<float>(in, pp::critic_weights);
    auto inputs = array<float>(in, rows * width);
    array<float>(in, rows * pp::actions);
    for (int j = 0; j < 6; ++j)
        array<float>(in, rows);
    array<float>(in, worlds);
    array<uint64_t>(in, rows);
    array<uint64_t>(in, rows);
    array<int>(in, rows);
    check(in.value<uint32_t>() == 192, "Unexpected trace ABI");
    const auto trace = array<uint8_t>(in, 192 * rows);
    in.finish();
    std::array<ProbeMoments, 2> yaw;
    std::array<double, 2> peaks{};
    double abs_feature = 0, max_feature = 0;
    int turns = 0;
    for (int i = 0; i < rows; ++i) {
        const auto kind = field<uint32_t>(trace.data() + 192 * i, 48);
        auto *row = inputs.data() + width * i;
        const double feature = row[width - 1];
        if (kind == 2)
            ++turns;
        if (feature == 0)
            continue;
        check(kind == 2 && std::abs(feature) <= 1, "Feature ineligible/outside contract");
        abs_feature += std::abs(feature);
        max_feature = std::max(max_feature, std::abs(feature));
        model.forward(row, 1);
        const float present[] = {std::tanh(model.output()[0]), std::tanh(model.output()[2])};
        row[width - 1] = 0;
        model.forward(row, 1);
        for (int j = 0; j < 2; ++j) {
            const auto delta = present[j] - std::tanh(model.output()[j * 2]);
            yaw[j].add(delta);
            peaks[j] = std::max(peaks[j], std::abs(double(delta)));
        }
    }
    check(yaw[0].n > 0, "No nonzero turn inputs");
    std::cout
        << std::setprecision(10)
        << "{\"study\":\"pond-motor-ppo-v6-signed-course-2026-11-24\",\"checkpoint\":8,\"source_sha256\":{"
           "\"candidate_0\":\"830b69a27b342c0d0ed1e3231708f09a5ae83e78aed1660631fff116ad877cab\",\"candidate_"
           "8\":\"9b68920b345b4be6699f8b979729e07d9f9d7a451664e4ca99f34487f2ca4ded\",\"batch_8\":"
           "\"2b5555a86e2a2fbcf56a0d476a5341f30c1de7eb36b8b9a07de3dde5c315c413\"},\n"
        << "\"new_physics_steps\":0,\"new_updates\":0,\"initial_input_column_l2\":0,\"final_input_column_"
           "l2\":"
        << std::sqrt(l2) << ",\"max_absolute_column_weight\":" << maximum_weight
        << ",\"batch_turn_rows\":" << turns << ",\"nonzero_feature_rows\":" << yaw[0].n
        << ",\"mean_abs_feature_deg\":" << abs_feature / yaw[0].n * 180
        << ",\"max_abs_feature_deg\":" << max_feature * 180
        << ",\n\"measurement\":\"Feed each recorded nonzero turn row of training batch eight through the "
           "frozen final actor twice, changing only the signed course-error feature to zero. Output "
           "difference is deterministic normalized joint target before physical filtering. Measured command "
           "sensitivity does not establish effective turning.\",";
    for (int j = 0; j < 2; ++j) {
        const auto m = yaw[j];
        std::cout << "\n\"" << (j ? "rear" : "front") << "_yaw\":{\"mean_delta\":" << m.sum / m.n
                  << ",\"rms_delta\":" << std::sqrt(m.sumsq / m.n) << ",\"max_abs_delta\":" << peaks[j] << "}"
                  << (j ? "" : ",");
    }
    std::cout << "\n}\n";
}
