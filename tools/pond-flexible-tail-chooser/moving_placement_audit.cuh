#pragma once

struct MovingPlacement {
    int row = -1, side = 0;
    float x = 0, z = 0, yaw = 0, speed = 0, slant = 0, runway = 0, gap = 0;
};

CUDARL_HD MovingPlacement measure_moving_placement(int lane, int episode,
        const tail::NavigationWorld* starts, int count, geometry::Parameters pond) {
    MovingPlacement result{};
    if (practice_for(lane) != Practice::Moving) return result;
    const auto row = depth_practice_row(lane, start_episode_seed(lane, episode), count);
    bank_training::ApproachEvidence proposed{};
    cudarl::physics::V position{};
    float yaw = 0;
    if (!bank_training::moving_start(lane, episode, starts[row], pond, position, yaw, proposed)) return result;
    tail::chooser::World w{};
    tail::chooser::initialize(w, starts[row], position, yaw);
    const auto v = tail::chooser::horizontal(w.pose.com_velocity_m_s);
    const auto speed = cudarl::physics::norm(v);
    if (w.fault || !(speed > .003f)) return result;
    const auto direction = v / speed;
    const auto distance = tail::bank::transport_range(w.navigation.motion.body, w.pose, pond, direction);
    const auto end = tail::chooser::horizontal(w.pose.com_m) + direction * proposed.straight_runway_m;
    const auto ax = float(pond.water_radius_x_m), az = float(pond.water_radius_z_m);
    auto outward = cudarl::physics::V{end.x/(ax*ax), 0, end.z/(az*az)};
    outward = outward / cudarl::physics::norm(outward);
    const auto tangent = cudarl::physics::cross({0,1,0}, outward);
    const float along = cudarl::physics::dot(direction, tangent);
    const float slant = atan2f(cudarl::physics::dot(direction, outward), fabsf(along)) * 180.f/tail::pi;
    const auto gap = tail::submerged_gap_m(pond, w.navigation.motion.body, w.pose);
    if (!distance.bounded || !(gap > 0.) || proposed.inward_side*along <= 0 ||
        !(fabsf(slant-proposed.slant_deg) < .002f) ||
        !(fabsf(distance.distance_m-proposed.straight_runway_m) < .00051f)) return result;
    return {int(row), proposed.inward_side, position.x, position.z, yaw, speed, slant,
            distance.distance_m, float(gap)};
}

__global__ void audit_moving_geometry(const tail::NavigationWorld* starts, int count,
        geometry::Parameters pond, int episode, MovingPlacement* result) {
    const int lane = blockIdx.x*blockDim.x+threadIdx.x;
    if (lane < worlds) result[lane] = measure_moving_placement(lane, episode, starts, count, pond);
}

void verify_moving_geometry(const std::vector<tail::NavigationWorld>& initial,
        const tail::NavigationWorld* device_starts, cudaStream_t stream, const fs::path& output) {
    device_ptr<MovingPlacement> measured(worlds);
    std::vector<MovingPlacement> gpu(worlds);
    std::ofstream rows_file(output/"moving-placements.csv");
    rows_file.exceptions(std::ios::badbit | std::ios::failbit);
    rows_file<<"episode,lane,source_pose,depth_mm,runway_band,side,x_m,z_m,actual_slant_deg,actual_COM_mm_s,measured_runway_mm,whole_body_gap_mm\n";
    int counts[3][3][2]{}, quadrants = 0, total = 0;
    float min_runway[3]{1,1,1}, max_runway[3]{}, max_position_error = 0, max_range_error = 0;
    for (int episode = 0; episode < 3; ++episode) {
        audit_moving_geometry<<<worlds/lane_threads,lane_threads,0,stream>>>(device_starts,
            int(initial.size()),larger_pond,episode,measured.get());
        check_launch(); CUDA_CHECK_THROW(cudaStreamSynchronize(stream)); measured.copy_to_host(gpu.data(),worlds);
        for (int lane=0;lane<worlds;++lane) {
            if (practice_for(lane)!=Practice::Moving) continue;
            const int depth=practice_depth_index(lane), band=bank_training::moving_runway_band(lane);
            const auto a=measure_moving_placement(lane,episode,initial.data(),int(initial.size()),larger_pond);
            const auto b=gpu[lane];
            if(a.row<0 || b.row!=a.row || b.side!=a.side || a.row%256<12 || a.row/256!=depth ||
               !(std::abs(a.yaw-b.yaw)<1.e-4f) || !(std::abs(a.speed-b.speed)<1.e-6f) ||
               !(std::abs(a.slant-b.slant)<.002f) || !(std::abs(a.gap-b.gap)<1.e-5f))
                throw std::runtime_error("Moving placement or CPU/CUDA comparison failed at lane "+std::to_string(lane));
            max_position_error=std::max(max_position_error,std::hypot(a.x-b.x,a.z-b.z));
            max_range_error=std::max(max_range_error,std::abs(a.runway-b.runway));
            min_runway[band]=std::min(min_runway[band],b.runway);
            max_runway[band]=std::max(max_runway[band],b.runway);
            ++counts[depth][band][b.side>0?1:0]; ++total;
            quadrants |= 1 << ((b.x>=0?1:0)+(b.z>=0?2:0));
            rows_file<<std::setprecision(10)<<episode<<','<<lane<<','<<b.row%256<<','
                <<std::lround(-1000*tail::distant_map::depth_at(depth))<<','<<band<<','<<b.side<<','
                <<b.x<<','<<b.z<<','<<b.slant<<','<<b.speed*1000<<','<<b.runway*1000<<','<<b.gap*1000<<'\n';
        }
    }
    for(int band=0;band<3;++band) {
        if(std::abs(min_runway[band]-bank_training::minimum_runway_m(band))>.002f ||
           std::abs(max_runway[band]-bank_training::maximum_runway_m(band))>.002f)
            throw std::runtime_error("Missing actual runway at a band endpoint");
        for(int depth=0;depth<3;++depth) for(int side=0;side<2;++side)
            if(!counts[depth][band][side]) throw std::runtime_error("Missing depth, runway or turn direction");
    }
    if(total!=1536 || quadrants!=15 || max_position_error>.00001f || max_range_error>.00050f)
        throw std::runtime_error("Moving placement coverage or CPU/CUDA geometry error");
    std::ofstream out(output/"moving-geometry.json");
    out<<"{\"pass\":true,\"paired_placements\":"<<total<<",\"positive_and_negative_sides_each_depth_band\":true,"
       <<"\"all_pond_quadrants\":true,\"excluded_pose_indices\":[0,11],\"max_cpu_cuda_position_difference_m\":"
       <<max_position_error<<",\"max_cpu_cuda_runway_difference_m\":"<<max_range_error
       <<",\"physics_steps\":0,\"optimizer_updates\":0}\n";
}
