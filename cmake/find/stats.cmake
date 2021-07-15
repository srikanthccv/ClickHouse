option(ENABLE_STATS "Enable StatsLib library" ${ENABLE_LIBRARIES})

if (ENABLE_STATS)
    if (NOT EXISTS "${ClickHouse_SOURCE_DIR}/contrib/stats")
        message (WARNING "submodule contrib/stats is missing. to fix try run: \n git submodule update --init --recursive")
        set (ENABLE_STATS 0)
        set (USE_STATS 0)
    elseif (NOT EXISTS "${ClickHouse_SOURCE_DIR}/contrib/gcem")
        message (WARNING "submodule contrib/gcem is missing. to fix try run: \n git submodule update --init --recursive")
        set (ENABLE_STATS 0)
        set (USE_STATS 0)
    else()
        set(STATS_INCLUDE_DIR ${ClickHouse_SOURCE_DIR}/contrib/stats/include)
        set(GCEM_INCLUDE_DIR ${ClickHouse_SOURCE_DIR}/contrib/gcem/include)
        set (USE_STATS 1)
    endif()

    if (NOT USE_STATS)
        message (${RECONFIGURE_MESSAGE_LEVEL} "Can't enable stats library")
    endif()
endif()

message (STATUS "Using stats=${USE_STATS} : ${STATS_INCLUDE_DIR}")
message (STATUS "Using gcem=${USE_STATS}: ${GCEM_INCLUDE_DIR}")
