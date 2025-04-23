cmake_minimum_required(VERSION 3.2)
project(rtune)

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_BUILD_TYPE Debug)

set(RTUNE_VERSION_MAJOR 0)
set(RTUNE_VERSION_MINOR 1)
set(RTUNE_VERSION ${RTUNE_VERSION_MAJOR}.${RTUNE_VERSION_MINOR})

#######################################################################################################################
################################ Feature config #######################################################################
#######################################################################################################################
# For tracing
#set(OMPT_TRACING_SUPPORT TRUE) # Tracing all the event and save them as record
#set(OMPT_ONLINE_TRACING_PRINT TRUE) # Print the traces online, not used anymore

set(RTUNE_MEASUREMENT_SUPPORT TRUE) # Enable measurement of timing. For enabling energy and PAPI counters, see below
set(RTUNE_AUTOTUNING TRUE)
#set(CPUFREQ_SUPPORT TRUE)
# For additional measurement, PE (power and energy) measurement needs root
#set(PAPI_MEASUREMENT_SUPPORT TRUE) # PAPI measurement
set(PE_MEASUREMENT_SUPPORT FALSE) # Power and energy measurement

# For optimization
#set(PE_OPTIMIZATION_SUPPORT TRUE)
#set(PE_OPTIMIZATION_DVFS TRUE)
#######################################################################################################################

set(SOURCE_FILES
    rtune_runtime.h
    rtune_runtime.c
)


add_library(rtune SHARED ${SOURCE_FILES})
target_link_libraries(rtune m)
#target_link_libraries(rtune unwind)

set(rtune_targets rtune)

install(FILES
        rtune_runtime.h
        DESTINATION include)

install(TARGETS ${rtune_targets}
        LIBRARY DESTINATION lib
        )
