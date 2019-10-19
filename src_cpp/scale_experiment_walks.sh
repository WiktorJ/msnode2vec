#!/usr/bin/env bash

graph=lesmis-fix.edgelist
emb=ls.emb
flags="-dry -ow"
p=0.5
q=2
r=32
l=100
declare -a arr=(16 32 64 128 256 512 1024)
for walks in "${arr[@]}"
do
    echo ${walks}
    printf "\n\nbase\n"
    build/node2vec/node2vec -i:../graph/er_graph_16384 -o:../emb/ls.em -l:${l} -p:${p} -q:${q}  -r:${walks} ${flags}
    printf "\n\nBiased 0.6\n"
    build/node2vec_ms_bias/node2vec_ms_bias  -i:../graph/er_graph_16384 -o:../emb/ls.em -l:${l} -p:${p} -q:${q}  -r:${walks} -rp:0.6 ${flags}
    printf "\n------------------------------------------------"
    printf "\n------------------------------------------------"
    printf "\n------------------------------------------------\n\n\n"
done