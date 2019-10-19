#include "stdafx.h"
#include "Snap.h"
#include "biasedrandomwalk.h"
#include <unordered_map>
#include <tuple>
#include <numeric>
#include <deque>
#include <random>

#define BIT_SET(a, b) ((a) |= (1ULL<<(b)))
#define BIT_CLEAR(a, b) ((a) &= ~(1ULL<<(b)))

//Preprocess alias sampling method
void GetNodeAlias(TFltV &PTblV, TIntVFltVPr &NTTable) {
    uint64 N = PTblV.Len();
    TIntV &KTbl = NTTable.Val1;
    TFltV &UTbl = NTTable.Val2;
    for (uint64 i = 0; i < N; i++) {
        KTbl[i] = 0;
        UTbl[i] = 0;
    }
    TIntV UnderV;
    TIntV OverV;
    for (uint64 i = 0; i < N; i++) {
        UTbl[i] = PTblV[i] * N;
        if (UTbl[i] < 1) {
            UnderV.Add(i);
        } else {
            OverV.Add(i);
        }
    }
    while (UnderV.Len() > 0 && OverV.Len() > 0) {
        uint64 Small = UnderV.Last();
        uint64 Large = OverV.Last();
        UnderV.DelLast();
        OverV.DelLast();
        KTbl[Small] = Large;
        UTbl[Large] = UTbl[Large] + UTbl[Small] - 1;
        if (UTbl[Large] < 1) {
            UnderV.Add(Large);
        } else {
            OverV.Add(Large);
        }
    }
    while (UnderV.Len() > 0) {
        uint64 curr = UnderV.Last();
        UnderV.DelLast();
        UTbl[curr] = 1;
    }
    while (OverV.Len() > 0) {
        uint64 curr = OverV.Last();
        OverV.DelLast();
        UTbl[curr] = 1;
    }

}

//Get random element using alias sampling method
uint64 AliasDrawInt(TIntVFltVPr &NTTable, TRnd &Rnd) {
    uint64 N = NTTable.GetVal1().Len();
    TInt X = static_cast<uint64>(Rnd.GetUniDev() * N);
    double Y = Rnd.GetUniDev();
    return Y < NTTable.GetVal2()[X] ? X : NTTable.GetVal1()[X];
}

void PreprocessNode(PWNet &InNet, const double &ParamP, const double &ParamQ,
                    TWNet::TNodeI NI, uint64 &NCnt, const bool &Verbose) {
    if (Verbose && NCnt % 100 == 0) {
        printf("\rPreprocessing progress: %.2lf%% ", (double) NCnt * 100 / (double) (InNet->GetNodes()));
        fflush(stdout);
    }
    //for node t
    THash<TInt, TBool> NbrH;                                    //Neighbors of t
    for (uint64 i = 0; i < NI.GetOutDeg(); i++) {
        NbrH.AddKey(NI.GetNbrNId(i));
    }
    for (uint64 i = 0; i < NI.GetOutDeg(); i++) {
        TWNet::TNodeI CurrI = InNet->GetNI(NI.GetNbrNId(i));      //for each node v
        double Psum = 0;
        TFltV PTable;                              //Probability distribution table
        for (uint64 j = 0; j < CurrI.GetOutDeg(); j++) {           //for each node x
            uint64 FId = CurrI.GetNbrNId(j);
            TFlt Weight;
            if (!(InNet->GetEDat(CurrI.GetId(), FId, Weight))) { continue; }
            if (FId == NI.GetId()) {
                PTable.Add(Weight / ParamP);
                Psum += Weight / ParamP;
            } else if (NbrH.IsKey(FId)) {
                PTable.Add(Weight);
                Psum += Weight;
            } else {
                PTable.Add(Weight / ParamQ);
                Psum += Weight / ParamQ;
            }
        }
        //Normalizing table
        for (uint64 j = 0; j < CurrI.GetOutDeg(); j++) {
            PTable[j] /= Psum;
        }
        GetNodeAlias(PTable, CurrI.GetDat().GetDat(NI.GetId()));
    }
    NCnt++;
}

//Preprocess transition probabilities for each path t->v->x
void PreprocessTransitionProbs(PWNet &InNet, const double &ParamP, const double &ParamQ, const bool &Verbose) {
    for (TWNet::TNodeI NI = InNet->BegNI(); NI < InNet->EndNI(); NI++) {
        InNet->SetNDat(NI.GetId(), TIntIntVFltVPrH());
    }
    for (TWNet::TNodeI NI = InNet->BegNI(); NI < InNet->EndNI(); NI++) {
        for (uint64 i = 0; i <
                           NI.GetOutDeg(); i++) {                    //allocating space in advance to avoid issues with multithreading
            TWNet::TNodeI CurrI = InNet->GetNI(NI.GetNbrNId(i));
            CurrI.GetDat().AddDat(NI.GetId(), TPair<TIntV, TFltV>(TIntV(CurrI.GetOutDeg()), TFltV(CurrI.GetOutDeg())));
        }
    }
    uint64 NCnt = 0;
    TIntV NIds;
    for (TWNet::TNodeI NI = InNet->BegNI(); NI < InNet->EndNI(); NI++) {
        NIds.Add(NI.GetId());
    }
#pragma omp parallel for schedule(dynamic)
    for (uint64 i = 0; i < NIds.Len(); i++) {
        PreprocessNode(InNet, ParamP, ParamQ, InNet->GetNI(NIds[i]), NCnt, Verbose);
    }
    if (Verbose) { printf("\n"); }
}

//Simulates a random walk
void SimulateWalk(PWNet &InNet,
                  TVVec<TInt, uint64> &WalksVV,
                  const std::vector<uint64> &StartNIds,
                  const int &WalkLen,
                  const int &NumWalk,
                  TRnd &Rnd,
                  uint64 current_walk_offset,
                  std::vector<uint64> &previous_nodes,
                  std::vector<uint64> &current_nodes,
                  std::vector<uint64> &saved_step,
                  std::map<int64, int64> &stats,
                  const double &reuse_prob) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);
    std::deque<uint64> visit;
    std::deque<uint64> visit_next;
    uint64 current_walk_number = 0;
    for (auto &node: StartNIds) {
        auto node_iterator = InNet->GetNI(node);
        if (node_iterator.GetOutDeg() != 0) {
            for (auto i = 0; i < NumWalk; i++) {
                auto current_node = node_iterator.GetNbrNId(Rnd.GetUniDevInt(node_iterator.GetOutDeg()));
                visit.push_front(node);
                visit.push_front(current_node);
                BIT_SET(previous_nodes[node], current_walk_number);
                BIT_SET(current_nodes[current_node], current_walk_number);
                WalksVV.PutXY(current_walk_offset + current_walk_number, 0, node);
                WalksVV.PutXY(current_walk_offset + current_walk_number, 1, current_node);
                current_walk_number++;
            }
        } else {
            for (auto i = 0; i < NumWalk; i++) {
                WalksVV.PutXY(current_walk_number++, 0, node);
            }
        }
    }
    uint64 current_length = 2;
    std::vector<uint64> walk_lengths(current_walk_number, 2);
    while (!visit.empty()) {
        while (!visit.empty()) {
            auto previous_node = visit.back();
            visit.pop_back();
            auto current_node = visit.back();
            visit.pop_back();
            auto node_iterator = InNet->GetNI(current_node);
            if (node_iterator.GetOutDeg() != 0) {
                uint64 indexes = previous_nodes[previous_node] & current_nodes[current_node];
//                int c = 0;
                if (indexes > 0) {
                    previous_nodes[previous_node] &= ~indexes;
                    current_nodes[current_node] &= ~indexes;
                    TIntVFltVPr *cur_data = nullptr;
                    while (indexes > 0) {
//                        c++;
                        uint64 next_node;
                        if (saved_step[current_node] != -1 && dis(gen) < reuse_prob) {
                            next_node = saved_step[current_node];
                        } else {
                            if (cur_data == nullptr) {
                                cur_data = &InNet->GetNDat(current_node).GetDat(previous_node);
                            }
                            next_node = node_iterator.GetNbrNId(
                                    AliasDrawInt(*cur_data, Rnd));
                            saved_step[current_node] = next_node;
                        }
                        uint64 index = __builtin_ffsll(indexes) - 1;
                        if (walk_lengths[index] < WalkLen - 1) {
                            visit_next.push_front(current_node);
                            visit_next.push_front(next_node);
                            BIT_SET(previous_nodes[current_node], index);
                            BIT_SET(current_nodes[next_node], index);
                            walk_lengths[index]++;
                        }
                        WalksVV.PutXY(current_walk_offset + index, walk_lengths[index], next_node);
                        BIT_CLEAR(indexes, index);
                    }
                }
//                double p = (int64) (((double) c / node_iterator.GetOutDeg()) * 100);
//                stats[p]++;

            }
        }
        current_length++;
        std::swap(visit, visit_next);
    }

}


void SimulateWalkReducedBias(PWNet &InNet,
                             TVVec<TInt, uint64> &WalksVV,
                             const std::vector<uint64> &StartNIds,
                             const int &WalkLen,
                             const int &NumWalk,
                             TRnd &Rnd,
                             uint64 current_walk_offset,
                             std::vector<uint64> &previous_nodes,
                             std::vector<uint64> &current_nodes,
                             std::vector<uint64> &saved_step,
                             std::vector<bool> &is_dist_1,
                             std::map<int64, int64> &stats,
                             const double &reuse_prob) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);
    std::deque<uint64> visit;
    std::deque<uint64> visit_next;
    uint64 current_walk_number = 0;
    for (auto &node: StartNIds) {
        auto node_iterator = InNet->GetNI(node);
        if (node_iterator.GetOutDeg() != 0) {
            for (auto i = 0; i < NumWalk; i++) {
                auto current_node = node_iterator.GetNbrNId(Rnd.GetUniDevInt(node_iterator.GetOutDeg()));
                visit.push_front(node);
                visit.push_front(current_node);
                BIT_SET(previous_nodes[node], current_walk_number);
                BIT_SET(current_nodes[current_node], current_walk_number);
                WalksVV.PutXY(current_walk_offset + current_walk_number, 0, node);
                WalksVV.PutXY(current_walk_offset + current_walk_number, 1, current_node);
                current_walk_number++;
            }
        } else {
            for (auto i = 0; i < NumWalk; i++) {
                WalksVV.PutXY(current_walk_number++, 0, node);
            }
        }
    }
    uint64 current_length = 2;
    std::vector<uint64> walk_lengths(NumWalk, 2);
    while (!visit.empty()) {
        while (!visit.empty()) {
            auto previous_node = visit.back();
            visit.pop_back();
            auto current_node = visit.back();
            visit.pop_back();
            auto node_iterator = InNet->GetNI(current_node);
            auto previous_node_iterator = InNet->GetNI(previous_node);
//            IsOutNId
            if (node_iterator.GetOutDeg() != 0) {
                uint64 indexes = previous_nodes[previous_node] & current_nodes[current_node];
//                int c = 0;
                if (indexes > 0) {
                    previous_nodes[previous_node] &= ~indexes;
                    current_nodes[current_node] &= ~indexes;
                    TIntVFltVPr *cur_data = nullptr;
                    while (indexes > 0) {
//                        c++;
                        int64 next_node;
                        double reuse = dis(gen);
                        if (reuse < reuse_prob && saved_step[current_node] == -2) {
                            next_node = previous_node;
                        } else {
                            bool is_current_dist_1 = previous_node_iterator.IsOutNId(saved_step[current_node]);
                            if (reuse < reuse_prob &&
                                saved_step[current_node] != -1 &&
                                ((is_current_dist_1 && is_dist_1[current_node]) ||
                                 (~is_current_dist_1 && ~is_dist_1[current_node]))) {
                                next_node = saved_step[current_node];
                            } else {
                                if (cur_data == nullptr) {
                                    cur_data = &InNet->GetNDat(current_node).GetDat(previous_node);
                                }
                                next_node = node_iterator.GetNbrNId(
                                        AliasDrawInt(*cur_data, Rnd));
                                if (next_node == previous_node) {
                                    saved_step[current_node] = -2;
                                } else {
                                    saved_step[current_node] = next_node;
                                    is_dist_1[current_node] = previous_node_iterator.IsOutNId(next_node);
                                }
                            }
                        }
                        uint64 index = __builtin_ffsll(indexes) - 1;
                        if (walk_lengths[index] < WalkLen - 1) {
                            visit_next.push_front(current_node);
                            visit_next.push_front(next_node);
                            BIT_SET(previous_nodes[current_node], index);
                            BIT_SET(current_nodes[next_node], index);
                            walk_lengths[index]++;
                        }
                        WalksVV.PutXY(current_walk_offset + index, walk_lengths[index], next_node);
                        BIT_CLEAR(indexes, index);
                    }
                }
//                double p = (int64) (((double) c / node_iterator.GetOutDeg()) * 100);
//                stats[p]++;

            }
        }
        current_length++;
        std::swap(visit, visit_next);
    }

}


//uint64 PredictMemoryRequirements(PWNet &InNet) {
//    uint64 MemNeeded = 0;
//    for (TWNet::TNodeI NI = InNet->BegNI(); NI < InNet->EndNI(); NI++) {
//        for (uint64 i = 0; i < NI.GetOutDeg(); i++) {
//            TWNet::TNodeI CurrI = InNet->GetNI(NI.GetNbrNId(i));
//            MemNeeded += CurrI.GetOutDeg() * (sizeof(TInt) + sizeof(TFlt));
//        }
//    }
//    return MemNeeded;
//}
