project(GPU-path-tracing-tutorial-4)

cmake_minimum_required(VERSION 2.8)

find_package(CUDA REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)
find_package(OpenGL REQUIRED)


include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${GLEW_INCLUDE_DIR})
include_directories(${GLUT_INCLUDE_DIR})
include_directories(${OPENGL_INCLUDE_DIR})

set(HEADERS
    Array.h
    BVH.h
    BVHNode.h
    Camera.h
    CudaBVH.h
    CudaRenderKernel.h
    Geometry.h
    HDRloader.h
    MouseKeyboardInput.h
    Scene.h
    SceneLoader.h
    Sort.h
    SplitBVHBuilder.h
    Util.h
    cutil_math.h
    linear_math.h
)

set(SOURCES
    BVH.cpp
    BVHNode.cpp
    Camera.cpp
    CudaBVH.cpp
    HDRloader.cpp
    main.cpp
    renderkernel.cu
    SceneLoader.cpp
    Sort.cpp
    SplitBVHBuilder.cpp
    Util.cpp
)

cuda_add_executable(tutorial4 ${HEADERS} ${SOURCES})

target_link_libraries(tutorial4
    ${CUDA_LIBRARIES}
    ${GLEW_LIBRARIES}
    ${GLUT_LIBRARIES}
    ${OPENGL_LIBRARIES}
)
