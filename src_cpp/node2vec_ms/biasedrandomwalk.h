#ifndef RAND_WALK_H
#define RAND_WALK_H
#include <base.h>
#include "network.h"
#include "vector"
#include "unordered_map"
#include "map"
typedef TNodeEDatNet<TIntIntVFltVPrH, TFlt> TWNet;
typedef TPt<TWNet> PWNet;

///Preprocesses transition probabilities for random walks. Has to be called once before SimulateWalk calls
void PreprocessTransitionProbs(PWNet& InNet, const double& ParamP, const double& ParamQ, const bool& verbose);

///Simulates one walk and writes it into Walk vector
void SimulateWalk(PWNet& InNet,
        TVVec<TInt, uint64> &WalksVV,
        const std::vector<uint64> &StartNId,
        const int& WalkLen,
        const int& NumWalk,
        TRnd& Rnd,
        uint64 current_walk_offset,
        std::vector<uint64> &previous_nodes,
        std::vector<uint64> &current_nodes,
        std::map<int64, int64> &stats);




//Predicts approximate memory required for preprocessing the graph
uint64 PredictMemoryRequirements(PWNet& InNet);

#endif //RAND_WALK_H
