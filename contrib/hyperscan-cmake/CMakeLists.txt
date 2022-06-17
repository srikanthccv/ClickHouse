# if (HAVE_SSSE3)
option (ENABLE_VECTORSCAN "Enable vectorscan library" ${ENABLE_LIBRARIES})
# elseif(ENABLE_VECTORSCAN)
#     message (${RECONFIGURE_MESSAGE_LEVEL} "Can't use vectorscan without SSSE3")
#     set (ENABLE_VECTORSCAN OFF)
# endif ()

if (NOT ENABLE_VECTORSCAN)
    message (STATUS "Not using vectorscan")
    return()
endif()

set (LIBRARY_DIR "${ClickHouse_SOURCE_DIR}/contrib/vectorscan")

set (SRCS
    "${LIBRARY_DIR}/src/alloc.c"
    "${LIBRARY_DIR}/src/crc32.c"
    "${LIBRARY_DIR}/src/database.c"
    "${LIBRARY_DIR}/src/dispatcher.c"
    "${LIBRARY_DIR}/src/grey.cpp"
    "${LIBRARY_DIR}/src/hs.cpp"
    "${LIBRARY_DIR}/src/hs_valid_platform.c"
    "${LIBRARY_DIR}/src/hs_version.c"
    "${LIBRARY_DIR}/src/runtime.c"
    "${LIBRARY_DIR}/src/scratch.c"
    "${LIBRARY_DIR}/src/scratch_dump.cpp"
    "${LIBRARY_DIR}/src/stream_compress.c"

    "${LIBRARY_DIR}/src/compiler/asserts.cpp"
    "${LIBRARY_DIR}/src/compiler/compiler.cpp"
    "${LIBRARY_DIR}/src/compiler/error.cpp"

    "${LIBRARY_DIR}/src/fdr/engine_description.cpp"
    "${LIBRARY_DIR}/src/fdr/fdr.c"
    "${LIBRARY_DIR}/src/fdr/fdr_compile.cpp"
    "${LIBRARY_DIR}/src/fdr/fdr_compile_util.cpp"
    "${LIBRARY_DIR}/src/fdr/fdr_confirm_compile.cpp"
    "${LIBRARY_DIR}/src/fdr/fdr_dump.cpp"
    "${LIBRARY_DIR}/src/fdr/fdr_engine_description.cpp"
    "${LIBRARY_DIR}/src/fdr/flood_compile.cpp"
    "${LIBRARY_DIR}/src/fdr/teddy.c"
    "${LIBRARY_DIR}/src/fdr/teddy_avx2.c"
    "${LIBRARY_DIR}/src/fdr/teddy_compile.cpp"
    "${LIBRARY_DIR}/src/fdr/teddy_engine_description.cpp"

    "${LIBRARY_DIR}/src/hwlm/hwlm.c"
    "${LIBRARY_DIR}/src/hwlm/hwlm_build.cpp"
    "${LIBRARY_DIR}/src/hwlm/hwlm_dump.cpp"
    "${LIBRARY_DIR}/src/hwlm/hwlm_literal.cpp"
    "${LIBRARY_DIR}/src/hwlm/noodle_build.cpp"
    "${LIBRARY_DIR}/src/hwlm/noodle_engine.cpp"

    "${LIBRARY_DIR}/src/nfa/accel.c"
    "${LIBRARY_DIR}/src/nfa/accel_dfa_build_strat.cpp"
    "${LIBRARY_DIR}/src/nfa/accel_dump.cpp"
    "${LIBRARY_DIR}/src/nfa/accelcompile.cpp"
    "${LIBRARY_DIR}/src/nfa/castle.c"
    "${LIBRARY_DIR}/src/nfa/castle_dump.cpp"
    "${LIBRARY_DIR}/src/nfa/castlecompile.cpp"
    "${LIBRARY_DIR}/src/nfa/dfa_build_strat.cpp"
    "${LIBRARY_DIR}/src/nfa/dfa_min.cpp"
    "${LIBRARY_DIR}/src/nfa/gough.c"
    "${LIBRARY_DIR}/src/nfa/goughcompile.cpp"
    "${LIBRARY_DIR}/src/nfa/goughcompile_accel.cpp"
    "${LIBRARY_DIR}/src/nfa/goughcompile_dump.cpp"
    "${LIBRARY_DIR}/src/nfa/goughcompile_reg.cpp"
    "${LIBRARY_DIR}/src/nfa/lbr.c"
    "${LIBRARY_DIR}/src/nfa/lbr_dump.cpp"
    "${LIBRARY_DIR}/src/nfa/limex_64.c"
    "${LIBRARY_DIR}/src/nfa/limex_accel.c"
    "${LIBRARY_DIR}/src/nfa/limex_compile.cpp"
    "${LIBRARY_DIR}/src/nfa/limex_dump.cpp"
    "${LIBRARY_DIR}/src/nfa/limex_native.c"
    "${LIBRARY_DIR}/src/nfa/limex_simd128.c"
    "${LIBRARY_DIR}/src/nfa/limex_simd256.c"
    "${LIBRARY_DIR}/src/nfa/limex_simd384.c"
    "${LIBRARY_DIR}/src/nfa/limex_simd512.c"
    "${LIBRARY_DIR}/src/nfa/mcclellan.c"
    "${LIBRARY_DIR}/src/nfa/mcclellancompile.cpp"
    "${LIBRARY_DIR}/src/nfa/mcclellancompile_util.cpp"
    "${LIBRARY_DIR}/src/nfa/mcclellandump.cpp"
    "${LIBRARY_DIR}/src/nfa/mcsheng.c"
    "${LIBRARY_DIR}/src/nfa/mcsheng_compile.cpp"
    "${LIBRARY_DIR}/src/nfa/mcsheng_data.c"
    "${LIBRARY_DIR}/src/nfa/mcsheng_dump.cpp"
    "${LIBRARY_DIR}/src/nfa/mpv.c"
    "${LIBRARY_DIR}/src/nfa/mpvcompile.cpp"
    "${LIBRARY_DIR}/src/nfa/nfa_api_dispatch.c"
    "${LIBRARY_DIR}/src/nfa/nfa_build_util.cpp"
    "${LIBRARY_DIR}/src/nfa/nfa_dump_dispatch.cpp"
    "${LIBRARY_DIR}/src/nfa/rdfa.cpp"
    "${LIBRARY_DIR}/src/nfa/rdfa_graph.cpp"
    "${LIBRARY_DIR}/src/nfa/rdfa_merge.cpp"
    "${LIBRARY_DIR}/src/nfa/repeat.c"
    "${LIBRARY_DIR}/src/nfa/repeatcompile.cpp"
    "${LIBRARY_DIR}/src/nfa/sheng.c"
    "${LIBRARY_DIR}/src/nfa/shengcompile.cpp"
    "${LIBRARY_DIR}/src/nfa/shengdump.cpp"
    "${LIBRARY_DIR}/src/nfa/shufti.cpp"
    "${LIBRARY_DIR}/src/nfa/shufticompile.cpp"
    "${LIBRARY_DIR}/src/nfa/tamarama.c"
    "${LIBRARY_DIR}/src/nfa/tamaramacompile.cpp"
    "${LIBRARY_DIR}/src/nfa/tamarama_dump.cpp"
    "${LIBRARY_DIR}/src/nfa/truffle.cpp"
    "${LIBRARY_DIR}/src/nfa/trufflecompile.cpp"
    "${LIBRARY_DIR}/src/nfa/vermicelli_simd.cpp"
    "${LIBRARY_DIR}/src/nfa/vermicellicompile.cpp"

    "${LIBRARY_DIR}/src/nfagraph/ng.cpp"
    "${LIBRARY_DIR}/src/nfagraph/ng_anchored_acyclic.cpp"
    "${LIBRARY_DIR}/src/nfagraph/ng_anchored_dots.cpp"
    "${LIBRARY_DIR}/src/nfagraph/ng_asserts.cpp"
    "${LIBRARY_DIR}/src/nfagraph/ng_holder.cpp"
    "${LIBRARY_DIR}/src/nfagraph/ng_is_equal.cpp"
    "${LIBRARY_DIR}/src/nfagraph/ng_lbr.cpp"
    "${LIBRARY_DIR}/src/nfagraph/ng_limex.cpp"
    "${LIBRARY_DIR}/src/nfagraph/ng_limex_accel.cpp"
    "${LIBRARY_DIR}/src/nfagraph/ng_literal_analysis.cpp"
    "${LIBRARY_DIR}/src/nfagraph/ng_literal_component.cpp"
    "${LIBRARY_DIR}/src/nfagraph/ng_literal_decorated.cpp"
    "${LIBRARY_DIR}/src/nfagraph/ng_prune.cpp"
    "${LIBRARY_DIR}/src/nfagraph/ng_puff.cpp"
    "${LIBRARY_DIR}/src/nfagraph/ng_redundancy.cpp"
    "${LIBRARY_DIR}/src/nfagraph/ng_region.cpp"
    "${LIBRARY_DIR}/src/nfagraph/ng_region_redundancy.cpp"
    "${LIBRARY_DIR}/src/nfagraph/ng_repeat.cpp"
    "${LIBRARY_DIR}/src/nfagraph/ng_reports.cpp"
    "${LIBRARY_DIR}/src/nfagraph/ng_restructuring.cpp"
    "${LIBRARY_DIR}/src/nfagraph/ng_revacc.cpp"
    "${LIBRARY_DIR}/src/nfagraph/ng_sep.cpp"
    "${LIBRARY_DIR}/src/nfagraph/ng_small_literal_set.cpp"
    "${LIBRARY_DIR}/src/nfagraph/ng_som.cpp"
    "${LIBRARY_DIR}/src/nfagraph/ng_som_add_redundancy.cpp"
    "${LIBRARY_DIR}/src/nfagraph/ng_som_util.cpp"
    "${LIBRARY_DIR}/src/nfagraph/ng_split.cpp"
    "${LIBRARY_DIR}/src/nfagraph/ng_squash.cpp"
    "${LIBRARY_DIR}/src/nfagraph/ng_stop.cpp"
    "${LIBRARY_DIR}/src/nfagraph/ng_vacuous.cpp"
    "${LIBRARY_DIR}/src/nfagraph/ng_violet.cpp"
    "${LIBRARY_DIR}/src/nfagraph/ng_width.cpp"

    "${LIBRARY_DIR}/src/parser/AsciiComponentClass.cpp"
    "${LIBRARY_DIR}/src/parser/Component.cpp"
    "${LIBRARY_DIR}/src/parser/ComponentAlternation.cpp"
    "${LIBRARY_DIR}/src/parser/ComponentAssertion.cpp"
    "${LIBRARY_DIR}/src/parser/ComponentByte.cpp"
    "${LIBRARY_DIR}/src/parser/ComponentClass.cpp"
    "${LIBRARY_DIR}/src/parser/ComponentCondReference.cpp"
    "${LIBRARY_DIR}/src/parser/ComponentEUS.cpp"
    "${LIBRARY_DIR}/src/parser/ComponentEmpty.cpp"
    "${LIBRARY_DIR}/src/parser/ComponentRepeat.cpp"
    "${LIBRARY_DIR}/src/parser/ComponentSequence.cpp"
    "${LIBRARY_DIR}/src/parser/ComponentVisitor.cpp"
    "${LIBRARY_DIR}/src/parser/ComponentWordBoundary.cpp"
    "${LIBRARY_DIR}/src/parser/ConstComponentVisitor.cpp"
    "${LIBRARY_DIR}/src/parser/Utf8ComponentClass.cpp"
    "${LIBRARY_DIR}/src/parser/buildstate.cpp"
    "${LIBRARY_DIR}/src/parser/check_refs.cpp"
    "${LIBRARY_DIR}/src/parser/logical_combination.cpp"
    "${LIBRARY_DIR}/src/parser/parse_error.cpp"
    "${LIBRARY_DIR}/src/parser/parser_util.cpp"
    "${LIBRARY_DIR}/src/parser/prefilter.cpp"
    "${LIBRARY_DIR}/src/parser/shortcut_literal.cpp"
    "${LIBRARY_DIR}/src/parser/ucp_table.cpp"
    "${LIBRARY_DIR}/src/parser/unsupported.cpp"
    "${LIBRARY_DIR}/src/parser/utf8_validate.cpp"

    "${LIBRARY_DIR}/src/rose/block.c"
    "${LIBRARY_DIR}/src/rose/catchup.c"
    "${LIBRARY_DIR}/src/rose/init.c"
    "${LIBRARY_DIR}/src/rose/match.c"
    "${LIBRARY_DIR}/src/rose/program_runtime.c"
    "${LIBRARY_DIR}/src/rose/rose_build_add.cpp"
    "${LIBRARY_DIR}/src/rose/rose_build_add_mask.cpp"
    "${LIBRARY_DIR}/src/rose/rose_build_anchored.cpp"
    "${LIBRARY_DIR}/src/rose/rose_build_bytecode.cpp"
    "${LIBRARY_DIR}/src/rose/rose_build_castle.cpp"
    "${LIBRARY_DIR}/src/rose/rose_build_width.cpp"
    "${LIBRARY_DIR}/src/rose/rose_in_util.cpp"
    "${LIBRARY_DIR}/src/rose/stream.c"

    "${LIBRARY_DIR}/src/smallwrite/smallwrite_build.cpp"
    "${LIBRARY_DIR}/src/smallwrite/smallwrite_dump.cpp"

    "${LIBRARY_DIR}/src/som/slot_manager.cpp"
    "${LIBRARY_DIR}/src/som/slot_manager_dump.cpp"
    "${LIBRARY_DIR}/src/som/som_runtime.c"
    "${LIBRARY_DIR}/src/som/som_stream.c"

    "${LIBRARY_DIR}/src/util/alloc.cpp"
    "${LIBRARY_DIR}/src/util/charreach.cpp"
    "${LIBRARY_DIR}/src/util/clique.cpp"
    "${LIBRARY_DIR}/src/util/compile_context.cpp"
    "${LIBRARY_DIR}/src/util/compile_error.cpp"
    "${LIBRARY_DIR}/src/util/depth.cpp"
    "${LIBRARY_DIR}/src/util/dump_charclass.cpp"
    "${LIBRARY_DIR}/src/util/dump_mask.cpp"
    "${LIBRARY_DIR}/src/util/fatbit_build.cpp"
    "${LIBRARY_DIR}/src/util/multibit.c"
    "${LIBRARY_DIR}/src/util/multibit_build.cpp"
    "${LIBRARY_DIR}/src/util/report_manager.cpp"
    "${LIBRARY_DIR}/src/util/state_compress.c"
    "${LIBRARY_DIR}/src/util/target_info.cpp"
    "${LIBRARY_DIR}/src/util/ue2string.cpp"

    "${LIBRARY_DIR}/src/util/arch/arm/cpuid_flags.c"
    "${LIBRARY_DIR}/src/util/arch/x86/cpuid_flags.c"
    "${LIBRARY_DIR}/src/util/arch/x86/masked_move.c"

    "${LIBRARY_DIR}/src/util/supervector/arch/arm/impl.cpp"
    "${LIBRARY_DIR}/src/util/supervector/arch/ppc64el/impl.cpp"
    "${LIBRARY_DIR}/src/util/supervector/arch/x86/impl.cpp"
)

add_library (_vectorscan ${SRCS})

# target_compile_options (_vectorscan
#     PRIVATE -g0 # Library has too much debug information
#     -mno-avx -mno-avx2 # The library is using dynamic dispatch and is confused if AVX is enabled globally
#     -march=corei7 -O2 -fno-strict-aliasing -fno-omit-frame-pointer -fvisibility=hidden # The options from original build system
#     -fno-sanitize=undefined # Assume the library takes care of itself
# )
target_include_directories (_vectorscan PRIVATE common "${LIBRARY_DIR}/include")
target_include_directories (_vectorscan SYSTEM PUBLIC "${LIBRARY_DIR}/src")

# if (ARCH_AMD64)
#     target_include_directories (_vectorscan PRIVATE x86_64)
# endif ()

target_link_libraries (_vectorscan PRIVATE boost::headers_only)

add_library (ch_contrib::vectorscan ALIAS _vectorscan)
