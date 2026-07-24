set(TARGET haven-server)

add_executable(${TARGET} main.cpp haven_sampler.cpp haven_sampler.h)
install(TARGETS ${TARGET} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../vendor/cpp-httplib
    ${CMAKE_CURRENT_SOURCE_DIR}/../../vendor/nlohmann
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common
)

target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
