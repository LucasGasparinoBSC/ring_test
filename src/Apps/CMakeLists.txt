project(Apps)

file(GLOB_RECURSE SRC_FILES ${CMAKE_CURRENT_LIST_DIR}/*.cpp)
file(GLOB_RECURSE INC_FILES ${CMAKE_CURRENT_LIST_DIR}/*.hpp)
set(HEADER_DIR ${CMAKE_CURRENT_LIST_DIR}/includes)

include_directories(${HEADER_DIR})

add_executable(${PROJECT_NAME} ${SRC_FILES} ${INC_FILES})
target_link_libraries(${PROJECT_NAME} RingComms)