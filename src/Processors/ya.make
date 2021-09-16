# This file is generated automatically, do not edit. See 'ya.make.in' and use 'utils/generate-ya-make' to regenerate it.
OWNER(g:clickhouse)

LIBRARY()

PEERDIR(
    clickhouse/src/Common
    contrib/libs/msgpack
    contrib/libs/protobuf
    contrib/libs/apache/arrow
    contrib/libs/apache/orc
)

ADDINCL(
    contrib/libs/apache/arrow/src
    contrib/libs/apache/orc/c++/include
)

CFLAGS(-DUSE_ARROW=1 -DUSE_PARQUET=1 -DUSE_ORC=1)

SRCS(
    Chunk.cpp
    ConcatProcessor.cpp
    DelayedPortsProcessor.cpp
    Executors/ExecutingGraph.cpp
    Executors/PipelineExecutingBlockInputStream.cpp
    Executors/PipelineExecutor.cpp
    Executors/PollingQueue.cpp
    Executors/PullingAsyncPipelineExecutor.cpp
    Executors/PullingPipelineExecutor.cpp
    ForkProcessor.cpp
    Formats/IInputFormat.cpp
    Formats/IOutputFormat.cpp
    Formats/IRowInputFormat.cpp
    Formats/IRowOutputFormat.cpp
    Formats/Impl/ArrowBlockInputFormat.cpp
    Formats/Impl/ArrowBlockOutputFormat.cpp
    Formats/Impl/ArrowBufferedStreams.cpp
    Formats/Impl/ArrowColumnToCHColumn.cpp
    Formats/Impl/BinaryRowInputFormat.cpp
    Formats/Impl/BinaryRowOutputFormat.cpp
    Formats/Impl/CHColumnToArrowColumn.cpp
    Formats/Impl/CSVRowInputFormat.cpp
    Formats/Impl/CSVRowOutputFormat.cpp
    Formats/Impl/ConstantExpressionTemplate.cpp
    Formats/Impl/JSONAsStringRowInputFormat.cpp
    Formats/Impl/JSONCompactEachRowRowInputFormat.cpp
    Formats/Impl/JSONCompactEachRowRowOutputFormat.cpp
    Formats/Impl/JSONCompactRowOutputFormat.cpp
    Formats/Impl/JSONEachRowRowInputFormat.cpp
    Formats/Impl/JSONEachRowRowOutputFormat.cpp
    Formats/Impl/JSONEachRowWithProgressRowOutputFormat.cpp
    Formats/Impl/JSONRowOutputFormat.cpp
    Formats/Impl/LineAsStringRowInputFormat.cpp
    Formats/Impl/MarkdownRowOutputFormat.cpp
    Formats/Impl/MsgPackRowInputFormat.cpp
    Formats/Impl/MsgPackRowOutputFormat.cpp
    Formats/Impl/MySQLOutputFormat.cpp
    Formats/Impl/NullFormat.cpp
    Formats/Impl/ODBCDriver2BlockOutputFormat.cpp
    Formats/Impl/ORCBlockInputFormat.cpp
    Formats/Impl/ORCBlockOutputFormat.cpp
    Formats/Impl/ParallelFormattingOutputFormat.cpp
    Formats/Impl/ParallelParsingInputFormat.cpp
    Formats/Impl/ParquetBlockInputFormat.cpp
    Formats/Impl/ParquetBlockOutputFormat.cpp
    Formats/Impl/PostgreSQLOutputFormat.cpp
    Formats/Impl/PrettyBlockOutputFormat.cpp
    Formats/Impl/PrettyCompactBlockOutputFormat.cpp
    Formats/Impl/PrettySpaceBlockOutputFormat.cpp
    Formats/Impl/ProtobufRowInputFormat.cpp
    Formats/Impl/ProtobufRowOutputFormat.cpp
    Formats/Impl/RawBLOBRowInputFormat.cpp
    Formats/Impl/RawBLOBRowOutputFormat.cpp
    Formats/Impl/RegexpRowInputFormat.cpp
    Formats/Impl/TSKVRowInputFormat.cpp
    Formats/Impl/TSKVRowOutputFormat.cpp
    Formats/Impl/TabSeparatedRowInputFormat.cpp
    Formats/Impl/TabSeparatedRowOutputFormat.cpp
    Formats/Impl/TemplateBlockOutputFormat.cpp
    Formats/Impl/TemplateRowInputFormat.cpp
    Formats/Impl/ValuesBlockInputFormat.cpp
    Formats/Impl/ValuesRowOutputFormat.cpp
    Formats/Impl/VerticalRowOutputFormat.cpp
    Formats/Impl/XMLRowOutputFormat.cpp
    Formats/LazyOutputFormat.cpp
    Formats/OutputStreamToOutputFormat.cpp
    Formats/PullingOutputFormat.cpp
    Formats/RowInputFormatWithDiagnosticInfo.cpp
    IAccumulatingTransform.cpp
    IInflatingTransform.cpp
    IProcessor.cpp
    ISimpleTransform.cpp
    ISink.cpp
    ISource.cpp
    LimitTransform.cpp
    Merges/Algorithms/AggregatingSortedAlgorithm.cpp
    Merges/Algorithms/CollapsingSortedAlgorithm.cpp
    Merges/Algorithms/FinishAggregatingInOrderAlgorithm.cpp
    Merges/Algorithms/GraphiteRollupSortedAlgorithm.cpp
    Merges/Algorithms/IMergingAlgorithmWithDelayedChunk.cpp
    Merges/Algorithms/IMergingAlgorithmWithSharedChunks.cpp
    Merges/Algorithms/MergingSortedAlgorithm.cpp
    Merges/Algorithms/ReplacingSortedAlgorithm.cpp
    Merges/Algorithms/SummingSortedAlgorithm.cpp
    Merges/Algorithms/VersionedCollapsingAlgorithm.cpp
    Merges/IMergingTransform.cpp
    Merges/MergingSortedTransform.cpp
    OffsetTransform.cpp
    Pipe.cpp
    Port.cpp
    QueryPipeline.cpp
    QueryPlan/AggregatingStep.cpp
    QueryPlan/ArrayJoinStep.cpp
    QueryPlan/BuildQueryPipelineSettings.cpp
    QueryPlan/CreatingSetsStep.cpp
    QueryPlan/CubeStep.cpp
    QueryPlan/DistinctStep.cpp
    QueryPlan/ExpressionStep.cpp
    QueryPlan/ExtremesStep.cpp
    QueryPlan/FillingStep.cpp
    QueryPlan/FilterStep.cpp
    QueryPlan/FinishSortingStep.cpp
    QueryPlan/IQueryPlanStep.cpp
    QueryPlan/ISourceStep.cpp
    QueryPlan/ITransformingStep.cpp
    QueryPlan/IntersectOrExceptStep.cpp
    QueryPlan/JoinStep.cpp
    QueryPlan/LimitByStep.cpp
    QueryPlan/LimitStep.cpp
    QueryPlan/MergeSortingStep.cpp
    QueryPlan/MergingAggregatedStep.cpp
    QueryPlan/MergingSortedStep.cpp
    QueryPlan/OffsetStep.cpp
    QueryPlan/Optimizations/QueryPlanOptimizationSettings.cpp
    QueryPlan/Optimizations/filterPushDown.cpp
    QueryPlan/Optimizations/liftUpArrayJoin.cpp
    QueryPlan/Optimizations/limitPushDown.cpp
    QueryPlan/Optimizations/mergeExpressions.cpp
    QueryPlan/Optimizations/optimizeTree.cpp
    QueryPlan/Optimizations/splitFilter.cpp
    QueryPlan/PartialSortingStep.cpp
    QueryPlan/QueryIdHolder.cpp
    QueryPlan/QueryPlan.cpp
    QueryPlan/ReadFromMergeTree.cpp
    QueryPlan/ReadFromPreparedSource.cpp
    QueryPlan/ReadFromRemote.cpp
    QueryPlan/ReadNothingStep.cpp
    QueryPlan/RollupStep.cpp
    QueryPlan/SettingQuotaAndLimitsStep.cpp
    QueryPlan/TotalsHavingStep.cpp
    QueryPlan/UnionStep.cpp
    QueryPlan/WindowStep.cpp
    ResizeProcessor.cpp
    Sources/DelayedSource.cpp
    Sources/RemoteSource.cpp
    Sources/SinkToOutputStream.cpp
    Sources/SourceFromInputStream.cpp
    Sources/SourceWithProgress.cpp
    Transforms/AddingDefaultsTransform.cpp
    Transforms/AddingSelectorTransform.cpp
    Transforms/AggregatingInOrderTransform.cpp
    Transforms/AggregatingTransform.cpp
    Transforms/ArrayJoinTransform.cpp
    Transforms/CheckSortedTransform.cpp
    Transforms/CopyTransform.cpp
    Transforms/CreatingSetsTransform.cpp
    Transforms/CubeTransform.cpp
    Transforms/DistinctTransform.cpp
    Transforms/ExpressionTransform.cpp
    Transforms/ExtremesTransform.cpp
    Transforms/FillingTransform.cpp
    Transforms/FilterTransform.cpp
    Transforms/FinishSortingTransform.cpp
    Transforms/IntersectOrExceptTransform.cpp
    Transforms/JoiningTransform.cpp
    Transforms/LimitByTransform.cpp
    Transforms/LimitsCheckingTransform.cpp
    Transforms/MaterializingTransform.cpp
    Transforms/MergeSortingTransform.cpp
    Transforms/MergingAggregatedMemoryEfficientTransform.cpp
    Transforms/MergingAggregatedTransform.cpp
    Transforms/PartialSortingTransform.cpp
    Transforms/ReverseTransform.cpp
    Transforms/RollupTransform.cpp
    Transforms/SortingTransform.cpp
    Transforms/SquashingChunksTransform.cpp
    Transforms/TotalsHavingTransform.cpp
    Transforms/WindowTransform.cpp
    Transforms/getSourceFromASTInsertQuery.cpp
    printPipeline.cpp

)

END()
