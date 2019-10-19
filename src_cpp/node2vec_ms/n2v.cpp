#include <chrono>
#include "stdafx.h"
#include "n2v.h"
#include "vector"
#include <unordered_map>
#include <map>

void node2vec(PWNet &InNet, const double &ParamP, const double &ParamQ,
              const int &Dimensions, const int &WalkLen, const int &NumWalks,
              const int &WinSize, const int &Iter, const bool &Verbose,
              const bool &OutputWalks, TVVec<TInt, uint64> &WalksVV,
              TIntFltVH &EmbeddingsHV) {
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
    std::map<int64, int64> stats;
//#pragma omp parallel for schedule(dynamic)
    uint64 bit_field_size = 64;
    uint64 current_walk_number = 0;
    for (int i = 0; i < NIdsV.Len(); ++i) {
        if (Verbose) {
            printf("\rWalking Progress: %.2lf%%", (double) current_walk_number * 100 / (double) AllWalks);
            fflush(stdout);
        }
        std::vector<uint64> start_nodes;
        start_nodes.push_back(NIdsV[i]);
        auto walks_left = NumWalks;
        while (walks_left > 0) {
            auto walks_count = walks_left > bit_field_size ? bit_field_size : walks_left;
            SimulateWalk(InNet, WalksVV, start_nodes, WalkLen, walks_count, Rnd, current_walk_number, previous_node,
                         current_node, stats);
            current_walk_number += walks_count;
            walks_left -= bit_field_size;
        }
    }


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

void node2vec(PWNet &InNet, const double &ParamP, const double &ParamQ,
              const int &Dimensions, const int &WalkLen, const int &NumWalks,
              const int &WinSize, const int &Iter, const bool &Verbose,
              TIntFltVH &EmbeddingsHV) {
    TVVec<TInt, uint64> WalksVV;
    bool OutputWalks = 0;
    node2vec(InNet, ParamP, ParamQ, Dimensions, WalkLen, NumWalks, WinSize,
             Iter, Verbose, OutputWalks, WalksVV, EmbeddingsHV);
}


void node2vec(const PNGraph &InNet, const double &ParamP, const double &ParamQ,
              const int &Dimensions, const int &WalkLen, const int &NumWalks,
              const int &WinSize, const int &Iter, const bool &Verbose,
              const bool &OutputWalks, TVVec<TInt, uint64> &WalksVV,
              TIntFltVH &EmbeddingsHV) {
    PWNet NewNet = PWNet::New();
    for (TNGraph::TEdgeI EI = InNet->BegEI(); EI < InNet->EndEI(); EI++) {
        if (!NewNet->IsNode(EI.GetSrcNId())) { NewNet->AddNode(EI.GetSrcNId()); }
        if (!NewNet->IsNode(EI.GetDstNId())) { NewNet->AddNode(EI.GetDstNId()); }
        NewNet->AddEdge(EI.GetSrcNId(), EI.GetDstNId(), 1.0);
    }
    node2vec(NewNet, ParamP, ParamQ, Dimensions, WalkLen, NumWalks, WinSize, Iter,
             Verbose, OutputWalks, WalksVV, EmbeddingsHV);
}

void node2vec(const PNGraph &InNet, const double &ParamP, const double &ParamQ,
              const int &Dimensions, const int &WalkLen, const int &NumWalks,
              const int &WinSize, const int &Iter, const bool &Verbose,
              TIntFltVH &EmbeddingsHV) {
    TVVec<TInt, uint64> WalksVV;
    bool OutputWalks = 0;
    node2vec(InNet, ParamP, ParamQ, Dimensions, WalkLen, NumWalks, WinSize,
             Iter, Verbose, OutputWalks, WalksVV, EmbeddingsHV);
}

void node2vec(const PNEANet &InNet, const double &ParamP, const double &ParamQ,
              const int &Dimensions, const int &WalkLen, const int &NumWalks,
              const int &WinSize, const int &Iter, const bool &Verbose,
              const bool &OutputWalks, TVVec<TInt, uint64> &WalksVV,
              TIntFltVH &EmbeddingsHV) {
    PWNet NewNet = PWNet::New();
    for (TNEANet::TEdgeI EI = InNet->BegEI(); EI < InNet->EndEI(); EI++) {
        if (!NewNet->IsNode(EI.GetSrcNId())) { NewNet->AddNode(EI.GetSrcNId()); }
        if (!NewNet->IsNode(EI.GetDstNId())) { NewNet->AddNode(EI.GetDstNId()); }
        NewNet->AddEdge(EI.GetSrcNId(), EI.GetDstNId(), InNet->GetFltAttrDatE(EI, "weight"));
    }
    node2vec(NewNet, ParamP, ParamQ, Dimensions, WalkLen, NumWalks, WinSize, Iter,
             Verbose, OutputWalks, WalksVV, EmbeddingsHV);
}

void node2vec(const PNEANet &InNet, const double &ParamP, const double &ParamQ,
              const int &Dimensions, const int &WalkLen, const int &NumWalks,
              const int &WinSize, const int &Iter, const bool &Verbose,
              TIntFltVH &EmbeddingsHV) {
    TVVec<TInt, uint64> WalksVV;
    bool OutputWalks = 0;
    node2vec(InNet, ParamP, ParamQ, Dimensions, WalkLen, NumWalks, WinSize,
             Iter, Verbose, OutputWalks, WalksVV, EmbeddingsHV);
}

