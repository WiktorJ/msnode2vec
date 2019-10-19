#ifndef RAND_WALK_H
#define RAND_WALK_H
#include <base.h>
#include "network.h"

typedef TNodeEDatNet<TIntIntVFltVPrH, TFlt> TWNet;
typedef TPt<TWNet> PWNet;

///Preprocesses transition probabilities for random walks. Has to be called once before SimulateWalk calls
void PreprocessTransitionProbs(PWNet& InNet, const double& ParamP, const double& ParamQ, const bool& verbose);

///Simulates one walk and writes it into Walk vector
void SimulateWalk(PWNet& InNet, uint64_t StartNId, const int& WalkLen, TRnd& Rnd, TIntV& Walk);

//Predicts approximate memory required for preprocessing the graph
uint64 PredictMemoryRequirements(PWNet& InNet);

#endif //RAND_WALK_H
