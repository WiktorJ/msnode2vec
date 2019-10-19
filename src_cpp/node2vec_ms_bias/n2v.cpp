#include <chrono>
#include "stdafx.h"
#include "n2v.h"
#include "vector"
#include <unordered_map>
#include <map>

inline void
dowalk(const bool &Verbose, uint64 &current_walk_number, uint64 &AllWalks, TIntV &NIdsV, uint64 &bit_field_size,
       const int &NumWalks, PWNet &InNet, TVVec<TInt, uint64> &WalksVV, const int &WalkLen, TRnd &Rnd,
       std::vector<uint64> &previous_node, std::vector<uint64> &current_node, std::vector<uint64> &saved_step,
       const double &reuse_prob, std::map<int64, int64> &stats, const std::vector<uint64> &start_nodes) {

    auto walks_left = NumWalks;
    while (walks_left > 0) {
        auto walks_count = walks_left > bit_field_size ? bit_field_size : walks_left;
        SimulateWalk(InNet, WalksVV, start_nodes, WalkLen, walks_count, Rnd, current_walk_number, previous_node,
                     current_node, saved_step, stats, reuse_prob);
        current_walk_number += walks_count;
        walks_left -= bit_field_size;
    }
}

void node2vec(PWNet &InNet, const double &ParamP, const double &ParamQ,
              const int &Dimensions, const int &WalkLen, const int &NumWalks,
              const int &WinSize, const int &Iter, const bool &Verbose,
              const bool &OutputWalks, TVVec<TInt, uint64> &WalksVV,
              TIntFltVH &EmbeddingsHV, const double &reuse_prob) {
    auto start_time = std::chrono::high_resolution_clock::now();
    //Preprocess transition probabilities
    PreprocessTransitionProbs(InNet, ParamP, ParamQ, Verbose);
    auto walk_start_time = std::chrono::high_resolution_clock::now();
    TIntV NIdsV;
    for (TWNet::TNodeI NI = InNet->BegNI(); NI < InNet->EndNI(); NI++) {
        NIdsV.Add(NI.GetId());
    }
    //Generate random walks
    uint64 AllWalks = (uint64) NumWalks * NIdsV.Len();
//    uint64 AllWalks = (uint64) NIdsV.Len();
    WalksVV = TVVec<TInt, uint64>(AllWalks, WalkLen);
    TRnd Rnd(time(NULL));
    auto node_count = InNet->GetNodes();
    std::vector<uint64> previous_node(node_count, 0);
    std::vector<uint64> current_node(node_count, 0);
    std::vector<uint64> saved_step(node_count, -1);
    std::vector<bool> is_dist_1(node_count, false);
    std::map<int64, int64> stats;
//#pragma omp parallel for schedule(dynamic)
    uint64 bit_field_size = 64;
    uint64 current_walk_number = 0;

    for (int i = 0; i < NIdsV.Len() - (NIdsV.Len() % 2); i++) {
        if (Verbose) {
            printf("\rWalking Progress: %.2lf%%", (double) current_walk_number * 100 / (double) AllWalks);
            fflush(stdout);
        }
        std::vector<uint64> start_nodes(1);
        start_nodes[0] = NIdsV[i];
        dowalk(Verbose, current_walk_number, AllWalks, NIdsV, bit_field_size,
               NumWalks, InNet, WalksVV, WalkLen, Rnd,
               previous_node,
               current_node, saved_step, reuse_prob, stats, start_nodes);
    }
//    if (NIdsV.Len() % 2 == 1) {
//        std::vector<uint64> start_nodes(1);
//        start_nodes.push_back(NIdsV[NIdsV.Len() - 1]);
//        dowalk(Verbose, current_walk_number, AllWalks, NIdsV, bit_field_size,
//               NumWalks, InNet, WalksVV, WalkLen, Rnd,
//               previous_node,
//               current_node, saved_step, reuse_prob, stats, start_nodes);
//    }
    auto walk_end_time = std::chrono::high_resolution_clock::now();
    if (!OutputWalks) {
        LearnEmbeddings(WalksVV, Dimensions, WinSize, Iter, Verbose, EmbeddingsHV);
        auto learn_end_time = std::chrono::high_resolution_clock::now();
    }
    printf("\rWalk time: %ld ms, Total time: %ld ms",
           std::chrono::duration_cast<std::chrono::milliseconds>(walk_end_time - walk_start_time).count(),
           std::chrono::duration_cast<std::chrono::milliseconds>(walk_end_time - start_time).count());
    fflush(stdout);
//    for(auto const &[key, value] : stats) {
//        printf("\n%ld: %ld", key, value);
//        fflush(stdout);
//    }
}

