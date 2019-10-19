#include <chrono>
#include "stdafx.h"
#include "n2v.h"

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
    WalksVV = TVVec<TInt, uint64>(AllWalks, WalkLen);
    TRnd Rnd(time(NULL));
    uint64 WalksDone = 0;
    for (uint64 i = 0; i < NumWalks; i++) {
        NIdsV.Shuffle(Rnd);
//#pragma omp parallel for schedule(dynamic)
        for (uint64 j = 0; j < NIdsV.Len(); j++) {
            if (Verbose && WalksDone % 10000 == 0) {
                printf("\rWalking Progress: %.2lf%%", (double) WalksDone * 100 / (double) AllWalks);
                fflush(stdout);
            }
            TIntV WalkV;
            SimulateWalk(InNet, NIdsV[j], WalkLen, Rnd, WalkV);
            for (uint64 k = 0; k < WalkV.Len(); k++) {
                WalksVV.PutXY(i * NIdsV.Len() + j, k, WalkV[k]);
            }
            WalksDone++;
        }
    }
  if (Verbose) {
    printf("\n");
    fflush(stdout);
  }
    //Learning embeddings

    auto walk_end_time = std::chrono::high_resolution_clock::now();
    if (!OutputWalks) {
        LearnEmbeddings(WalksVV, Dimensions, WinSize, Iter, Verbose, EmbeddingsHV);
        auto learn_end_time = std::chrono::high_resolution_clock::now();
    }
    printf("\rWalk time: %ld ms, Total time: %ld ms",
           std::chrono::duration_cast<std::chrono::milliseconds>(walk_end_time - walk_start_time).count(),
           std::chrono::duration_cast<std::chrono::milliseconds>(walk_end_time - start_time).count());
    fflush(stdout);
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

