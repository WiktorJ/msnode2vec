add_executable(
        node2vec_ms ${OBJS}
        biasedrandomwalk.cpp
        biasedrandomwalk.h
        n2v.cpp
        n2v.h
        node2vec.cpp
        stdafx.cpp
        stdafx.h
        targetver.h
        ${EXSNAPADV}/word2vec.cpp
)
#SET_TARGET_PROPERTIES(node2vec_ms PROPERTIES COMPILE_FLAGS "-fopenmp"
#        LINK_FLAGS "-fopenmp")


target_link_libraries(node2vec_ms -fopenmp)
