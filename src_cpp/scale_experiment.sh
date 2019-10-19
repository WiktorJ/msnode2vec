#!/usr/bin/env bash

graph=lesmis-fix.edgelist
emb=ls.emb
flags="-dry -ow"
p=0.5
q=2
r=32
l=100
declare -a arr=("er_graph_1024" "er_graph_2048" "er_graph_4096" "er_graph_8192" "er_graph_16384" "er_graph_32768" "er_graph_65536" "er_graph_131072")
for paths in "${arr[@]}"
do
    echo ${paths}
    printf "\n\nbase\n"
    build/node2vec/node2vec -i:../graph/${paths} -o:../emb/ls.em -l:${l} -p:${p} -q:${q}  -r:${r} ${flags}
    printf "\n\nBiased 0.6\n"
    build/node2vec_ms_bias/node2vec_ms_bias  -i:../graph/${paths} -o:../emb/ls.em -l:${l} -p:${p} -q:${q}  -r:${r} -rp:0.6 ${flags}
    printf "\n------------------------------------------------"
    printf "\n------------------------------------------------"
    printf "\n------------------------------------------------\n\n\n"
done