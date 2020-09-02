# This file is generated automatically, do not edit. See 'ya.make.in' and use 'utils/generate-ya-make' to regenerate it.
LIBRARY()

CFLAGS(
    -DUSE_H3 -DUSE_SSL -DUSE_XXHASH
)

ADDINCL(
    library/cpp/consistent_hashing
    contrib/libs/farmhash
    contrib/libs/h3/h3lib/include
    contrib/libs/hyperscan/src
    contrib/libs/icu/common
    contrib/libs/libdivide
    contrib/libs/rapidjson/include
    contrib/libs/xxhash
)

PEERDIR(
    clickhouse/src/Common
    clickhouse/src/Parsers
    clickhouse/src/Dictionaries
    contrib/libs/farmhash
    contrib/libs/fastops/fastops
    contrib/libs/h3
    contrib/libs/hyperscan
    contrib/libs/icu
    contrib/libs/libdivide
    contrib/libs/metrohash
    contrib/libs/rapidjson
    contrib/libs/xxhash
    library/cpp/consistent_hashing
)

# "Arcadia" build is slightly deficient. It lacks many libraries that we need.
SRCS(
    abs.cpp
    acos.cpp
    addDays.cpp
    addHours.cpp
    addMinutes.cpp
    addMonths.cpp
    addQuarters.cpp
    addressToLine.cpp
    addressToSymbol.cpp
    addSeconds.cpp
    addWeeks.cpp
    addYears.cpp
    appendTrailingCharIfAbsent.cpp
    array/arrayAll.cpp
    array/arrayAUC.cpp
    array/arrayCompact.cpp
    array/arrayConcat.cpp
    array/arrayCount.cpp
    array/array.cpp
    array/arrayCumSum.cpp
    array/arrayCumSumNonNegative.cpp
    array/arrayDifference.cpp
    array/arrayDistinct.cpp
    array/arrayElement.cpp
    array/arrayEnumerate.cpp
    array/arrayEnumerateDense.cpp
    array/arrayEnumerateDenseRanked.cpp
    array/arrayEnumerateRanked.cpp
    array/arrayEnumerateUniq.cpp
    array/arrayEnumerateUniqRanked.cpp
    array/arrayExists.cpp
    array/arrayFill.cpp
    array/arrayFilter.cpp
    array/arrayFirst.cpp
    array/arrayFirstIndex.cpp
    array/arrayFlatten.cpp
    array/arrayIntersect.cpp
    array/arrayJoin.cpp
    array/arrayMap.cpp
    array/arrayPopBack.cpp
    array/arrayPopFront.cpp
    array/arrayPushBack.cpp
    array/arrayPushFront.cpp
    array/arrayReduce.cpp
    array/arrayReduceInRanges.cpp
    array/arrayResize.cpp
    array/arrayReverse.cpp
    array/arraySlice.cpp
    array/arraySort.cpp
    array/arraySplit.cpp
    array/arraySum.cpp
    array/arrayUniq.cpp
    array/arrayWithConstant.cpp
    array/arrayZip.cpp
    array/countEqual.cpp
    array/emptyArray.cpp
    array/emptyArrayToSingle.cpp
    array/hasAll.cpp
    array/hasAny.cpp
    array/has.cpp
    array/hasSubstr.cpp
    array/indexOf.cpp
    array/length.cpp
    array/mapOp.cpp
    array/range.cpp
    array/registerFunctionsArray.cpp
    asin.cpp
    assumeNotNull.cpp
    atan.cpp
    bar.cpp
    base64Decode.cpp
    base64Encode.cpp
    bitAnd.cpp
    bitBoolMaskAnd.cpp
    bitBoolMaskOr.cpp
    bitCount.cpp
    bitNot.cpp
    bitOr.cpp
    bitRotateLeft.cpp
    bitRotateRight.cpp
    bitShiftLeft.cpp
    bitShiftRight.cpp
    bitSwapLastTwo.cpp
    bitTestAll.cpp
    bitTestAny.cpp
    bitTest.cpp
    bitWrapperFunc.cpp
    bitXor.cpp
    blockNumber.cpp
    blockSerializedSize.cpp
    blockSize.cpp
    buildId.cpp
    caseWithExpression.cpp
    cbrt.cpp
    coalesce.cpp
    concat.cpp
    convertCharset.cpp
    cos.cpp
    countDigits.cpp
    CRC.cpp
    currentDatabase.cpp
    currentUser.cpp
    dateDiff.cpp
    date_trunc.cpp
    defaultValueOfArgumentType.cpp
    defaultValueOfTypeName.cpp
    demange.cpp
    divide.cpp
    dumpColumnStructure.cpp
    e.cpp
    empty.cpp
    endsWith.cpp
    equals.cpp
    erfc.cpp
    erf.cpp
    evalMLMethod.cpp
    exp10.cpp
    exp2.cpp
    exp.cpp
    extractAllGroupsHorizontal.cpp
    extractAllGroupsVertical.cpp
    extract.cpp
    extractGroups.cpp
    extractTimeZoneFromFunctionArguments.cpp
    filesystem.cpp
    finalizeAggregation.cpp
    formatDateTime.cpp
    formatRow.cpp
    formatString.cpp
    fromUnixTimestamp64Micro.cpp
    fromUnixTimestamp64Milli.cpp
    fromUnixTimestamp64Nano.cpp
    FunctionFactory.cpp
    FunctionFQDN.cpp
    FunctionHelpers.cpp
    FunctionJoinGet.cpp
    FunctionsCoding.cpp
    FunctionsConversion.cpp
    FunctionsEmbeddedDictionaries.cpp
    FunctionsExternalDictionaries.cpp
    FunctionsExternalModels.cpp
    FunctionsFormatting.cpp
    FunctionsHashing.cpp
    FunctionsJSON.cpp
    FunctionsLogical.cpp
    FunctionsRandom.cpp
    FunctionsRound.cpp
    FunctionsStringArray.cpp
    FunctionsStringSimilarity.cpp
    fuzzBits.cpp
    GatherUtils/concat.cpp
    GatherUtils/createArraySink.cpp
    GatherUtils/createArraySource.cpp
    GatherUtils/createValueSource.cpp
    GatherUtils/has.cpp
    GatherUtils/push.cpp
    GatherUtils/resizeConstantSize.cpp
    GatherUtils/resizeDynamicSize.cpp
    GatherUtils/sliceDynamicOffsetBounded.cpp
    GatherUtils/sliceDynamicOffsetUnbounded.cpp
    GatherUtils/sliceFromLeftConstantOffsetBounded.cpp
    GatherUtils/sliceFromLeftConstantOffsetUnbounded.cpp
    GatherUtils/sliceFromRightConstantOffsetBounded.cpp
    GatherUtils/sliceFromRightConstantOffsetUnbounded.cpp
    gcd.cpp
    generateUUIDv4.cpp
    GeoHash.cpp
    geohashDecode.cpp
    geohashEncode.cpp
    geohashesInBox.cpp
    geoToH3.cpp
    getMacro.cpp
    getScalar.cpp
    getSetting.cpp
    getSizeOfEnumType.cpp
    globalVariable.cpp
    greatCircleDistance.cpp
    greater.cpp
    greaterOrEquals.cpp
    greatest.cpp
    h3EdgeAngle.cpp
    h3EdgeLengthM.cpp
    h3GetBaseCell.cpp
    h3GetResolution.cpp
    h3HexAreaM2.cpp
    h3IndexesAreNeighbors.cpp
    h3IsValid.cpp
    h3kRing.cpp
    h3ToChildren.cpp
    h3ToParent.cpp
    h3ToString.cpp
    hasColumnInTable.cpp
    hasThreadFuzzer.cpp
    hasTokenCaseInsensitive.cpp
    hasToken.cpp
    hostName.cpp
    identity.cpp
    if.cpp
    ifNotFinite.cpp
    ifNull.cpp
    IFunction.cpp
    ignore.cpp
    ilike.cpp
    in.cpp
    initializeAggregation.cpp
    intDiv.cpp
    intDivOrZero.cpp
    intExp10.cpp
    intExp2.cpp
    isConstant.cpp
    isDecimalOverflow.cpp
    isFinite.cpp
    isInfinite.cpp
    isNaN.cpp
    isNotNull.cpp
    isNull.cpp
    isValidUTF8.cpp
    isZeroOrNull.cpp
    jumpConsistentHash.cpp
    lcm.cpp
    least.cpp
    lengthUTF8.cpp
    less.cpp
    lessOrEquals.cpp
    lgamma.cpp
    like.cpp
    log10.cpp
    log2.cpp
    log.cpp
    lowCardinalityIndices.cpp
    lowCardinalityKeys.cpp
    lower.cpp
    lowerUTF8.cpp
    match.cpp
    materialize.cpp
    minus.cpp
    modulo.cpp
    moduloOrZero.cpp
    multiFuzzyMatchAllIndices.cpp
    multiFuzzyMatchAny.cpp
    multiFuzzyMatchAnyIndex.cpp
    multiIf.cpp
    multiMatchAllIndices.cpp
    multiMatchAny.cpp
    multiMatchAnyIndex.cpp
    multiply.cpp
    multiSearchAllPositionsCaseInsensitive.cpp
    multiSearchAllPositionsCaseInsensitiveUTF8.cpp
    multiSearchAllPositions.cpp
    multiSearchAllPositionsUTF8.cpp
    multiSearchAnyCaseInsensitive.cpp
    multiSearchAnyCaseInsensitiveUTF8.cpp
    multiSearchAny.cpp
    multiSearchAnyUTF8.cpp
    multiSearchFirstIndexCaseInsensitive.cpp
    multiSearchFirstIndexCaseInsensitiveUTF8.cpp
    multiSearchFirstIndex.cpp
    multiSearchFirstIndexUTF8.cpp
    multiSearchFirstPositionCaseInsensitive.cpp
    multiSearchFirstPositionCaseInsensitiveUTF8.cpp
    multiSearchFirstPosition.cpp
    multiSearchFirstPositionUTF8.cpp
    negate.cpp
    neighbor.cpp
    normalizedQueryHash.cpp
    normalizeQuery.cpp
    notEmpty.cpp
    notEquals.cpp
    notILike.cpp
    notLike.cpp
    now64.cpp
    now.cpp
    nullIf.cpp
    pi.cpp
    plus.cpp
    pointInEllipses.cpp
    pointInPolygon.cpp
    positionCaseInsensitive.cpp
    positionCaseInsensitiveUTF8.cpp
    position.cpp
    positionUTF8.cpp
    pow.cpp
    rand64.cpp
    randConstant.cpp
    rand.cpp
    randomFixedString.cpp
    randomPrintableASCII.cpp
    randomString.cpp
    randomStringUTF8.cpp
    regexpQuoteMeta.cpp
    registerFunctionsArithmetic.cpp
    registerFunctionsComparison.cpp
    registerFunctionsConditional.cpp
    registerFunctionsConsistentHashing.cpp
    registerFunctions.cpp
    registerFunctionsDateTime.cpp
    registerFunctionsGeo.cpp
    registerFunctionsHigherOrder.cpp
    registerFunctionsIntrospection.cpp
    registerFunctionsMath.cpp
    registerFunctionsMiscellaneous.cpp
    registerFunctionsNull.cpp
    registerFunctionsRandom.cpp
    registerFunctionsReinterpret.cpp
    registerFunctionsString.cpp
    registerFunctionsStringRegexp.cpp
    registerFunctionsStringSearch.cpp
    registerFunctionsTuple.cpp
    registerFunctionsUnixTimestamp64.cpp
    registerFunctionsVisitParam.cpp
    reinterpretAsFixedString.cpp
    reinterpretAsString.cpp
    reinterpretStringAs.cpp
    repeat.cpp
    replaceAll.cpp
    replaceOne.cpp
    replaceRegexpAll.cpp
    replaceRegexpOne.cpp
    replicate.cpp
    reverse.cpp
    reverseUTF8.cpp
    roundAge.cpp
    roundDuration.cpp
    roundToExp2.cpp
    rowNumberInAllBlocks.cpp
    rowNumberInBlock.cpp
    runningAccumulate.cpp
    runningDifference.cpp
    runningDifferenceStartingWithFirstValue.cpp
    sigmoid.cpp
    sin.cpp
    sleep.cpp
    sleepEachRow.cpp
    sqrt.cpp
    startsWith.cpp
    stringToH3.cpp
    substring.cpp
    subtractDays.cpp
    subtractHours.cpp
    subtractMinutes.cpp
    subtractMonths.cpp
    subtractQuarters.cpp
    subtractSeconds.cpp
    subtractWeeks.cpp
    subtractYears.cpp
    tan.cpp
    tanh.cpp
    TargetSpecific.cpp
    tgamma.cpp
    throwIf.cpp
    timeSlot.cpp
    timeSlots.cpp
    timezone.cpp
    toColumnTypeName.cpp
    toCustomWeek.cpp
    today.cpp
    toDayOfMonth.cpp
    toDayOfWeek.cpp
    toDayOfYear.cpp
    toFixedString.cpp
    toHour.cpp
    toISOWeek.cpp
    toISOYear.cpp
    toLowCardinality.cpp
    toMinute.cpp
    toMonday.cpp
    toMonth.cpp
    toNullable.cpp
    toQuarter.cpp
    toRelativeDayNum.cpp
    toRelativeHourNum.cpp
    toRelativeMinuteNum.cpp
    toRelativeMonthNum.cpp
    toRelativeQuarterNum.cpp
    toRelativeSecondNum.cpp
    toRelativeWeekNum.cpp
    toRelativeYearNum.cpp
    toSecond.cpp
    toStartOfDay.cpp
    toStartOfFifteenMinutes.cpp
    toStartOfFiveMinute.cpp
    toStartOfHour.cpp
    toStartOfInterval.cpp
    toStartOfISOYear.cpp
    toStartOfMinute.cpp
    toStartOfMonth.cpp
    toStartOfQuarter.cpp
    toStartOfSecond.cpp
    toStartOfTenMinutes.cpp
    toStartOfYear.cpp
    toTime.cpp
    toTimeZone.cpp
    toTypeName.cpp
    toUnixTimestamp64Micro.cpp
    toUnixTimestamp64Milli.cpp
    toUnixTimestamp64Nano.cpp
    toValidUTF8.cpp
    toYear.cpp
    toYYYYMM.cpp
    toYYYYMMDD.cpp
    toYYYYMMDDhhmmss.cpp
    transform.cpp
    trap.cpp
    trim.cpp
    tryBase64Decode.cpp
    tuple.cpp
    tupleElement.cpp
    upper.cpp
    upperUTF8.cpp
    uptime.cpp
    URL/basename.cpp
    URL/cutFragment.cpp
    URL/cutQueryStringAndFragment.cpp
    URL/cutQueryString.cpp
    URL/cutToFirstSignificantSubdomain.cpp
    URL/cutURLParameter.cpp
    URL/cutWWW.cpp
    URL/decodeURLComponent.cpp
    URL/domain.cpp
    URL/domainWithoutWWW.cpp
    URL/extractURLParameter.cpp
    URL/extractURLParameterNames.cpp
    URL/extractURLParameters.cpp
    URL/firstSignificantSubdomain.cpp
    URL/fragment.cpp
    URL/netloc.cpp
    URL/path.cpp
    URL/pathFull.cpp
    URL/port.cpp
    URL/protocol.cpp
    URL/queryStringAndFragment.cpp
    URL/queryString.cpp
    URL/registerFunctionsURL.cpp
    URL/tldLookup.generated.cpp
    URL/topLevelDomain.cpp
    URL/URLHierarchy.cpp
    URL/URLPathHierarchy.cpp
    version.cpp
    visibleWidth.cpp
    visitParamExtractBool.cpp
    visitParamExtractFloat.cpp
    visitParamExtractInt.cpp
    visitParamExtractRaw.cpp
    visitParamExtractString.cpp
    visitParamExtractUInt.cpp
    visitParamHas.cpp
    yandexConsistentHash.cpp
    yesterday.cpp

)

END()
