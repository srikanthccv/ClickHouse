include (CheckCXXSourceCompiles)

# Try to add -Wflag if compiler supports it
macro (add_warning flag)
    string (REPLACE "-" "_" underscored_flag ${flag})
    string (REPLACE "+" "x" underscored_flag ${underscored_flag})
    check_cxx_compiler_flag("-W${flag}" SUPPORTS_FLAG_${underscored_flag})
    if (SUPPORTS_FLAG_${underscored_flag})
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W${flag}")
    else ()
        message (WARNING "Flag -W${flag} is unsupported")
    endif ()
endmacro (add_warning)

# Try to add -Wno flag if compiler supports it
macro (no_warning flag)
    add_warning(no-${flag})
endmacro (add_warning)
