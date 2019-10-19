#!/usr/bin/env bash

graph=lesmis-fix.edgelist
emb=ls.emb
flags="-dry -ow"
p=0.5
q=2
r=32
l=100
declare -a arr=("-i:../graph/facebook_combined.edgelist -o:../emb/ls.em", "-i:../graph/roadNet-PA-fix.txt -o:../emb/r.em", "-i:../graph/twitter_combined-fix.txt -o:../emb/t.em",  "-i:../graph/com-youtube-fix.ungraph.txt -o:../emb/y.em",)
for paths in "${arr[@]}"
do
    echo ${paths}
    printf "\n\nbase\n"
    build/node2vec/node2vec ${paths} -l:${l} -p:${p} -q:${q}  -r:${r} ${flags}
    printf "\n\nMS\n"
    build/node2vec_ms/node2vec_ms ${paths} -l:${l} -p:${p} -q:${q}  -r:${r} ${flags}
    printf "\n\nBiased 0.2\n"
    build/node2vec_ms_bias/node2vec_ms_bias ${paths} -l:${l} -p:${p} -q:${q}  -r:${r} -rp:0.2 ${flags}
    printf "\n\nBiased 0.4\n"
    build/node2vec_ms_bias/node2vec_ms_bias ${paths} -l:${l} -p:${p} -q:${q}  -r:${r} -rp:0.4 ${flags}
    printf "\n\nBiased 0.6\n"
    build/node2vec_ms_bias/node2vec_ms_bias ${paths} -l:${l} -p:${p} -q:${q}  -r:${r} -rp:0.6 ${flags}
    printf "\n\nBiased 0.8\n"
    build/node2vec_ms_bias/node2vec_ms_bias ${paths} -l:${l} -p:${p} -q:${q}  -r:${r} -rp:0.8 ${flags}
    printf "\n------------------------------------------------"
    printf "\n------------------------------------------------"
    printf "\n------------------------------------------------\n\n\n"
done