cmake_minimum_required(VERSION 3.12)
project(node2vec)


#find_package(OpenMP)
#set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
#set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

set(CMAKE_CXX_STANDARD 20)
set(SNAPROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../snap)
set(GLIB  glib-core)
set(SNAP  snap-core)
set(GLIBADV  glib-adv)
set(SNAPADV  snap-adv)
set(SOURCEROOT node2vec_ms)

set(EXGLIB ${SNAPROOT}/${GLIB})
set(EXSNAP ${SNAPROOT}/${SNAP})
set(EXGLIBADV ${SNAPROOT}/${GLIBADV})
set(EXSNAPADV ${SNAPROOT}/${SNAPADV})

#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
#SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
#SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")

set(OBJS ${EXSNAP}/Snap.o )

include_directories(${SOURCEROOT} ${EXGLIBADV} ${EXGLIB} ${EXSNAP} ${EXSNAPADV})

add_subdirectory(node2vec_ms_bias)
add_subdirectory(node2vec_ms)
add_subdirectory(node2vec)

SET_SOURCE_FILES_PROPERTIES(
        ${OBJS}
        PROPERTIES
        EXTERNAL_OBJECT true
        GENERATED true
)

