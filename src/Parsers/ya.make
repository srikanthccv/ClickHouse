# This file is generated automatically, do not edit. See 'ya.make.in' and use 'utils/generate-ya-make' to regenerate it.
LIBRARY()

PEERDIR(
    clickhouse/src/Common
)

SRCS(
    ASTAlterQuery.cpp
    ASTAsterisk.cpp
    ASTColumnDeclaration.cpp
    ASTColumnsMatcher.cpp
    ASTConstraintDeclaration.cpp
    ASTCreateQuery.cpp
    ASTCreateQuotaQuery.cpp
    ASTCreateRoleQuery.cpp
    ASTCreateRowPolicyQuery.cpp
    ASTCreateSettingsProfileQuery.cpp
    ASTCreateUserQuery.cpp
    ASTDictionaryAttributeDeclaration.cpp
    ASTDictionary.cpp
    ASTDropAccessEntityQuery.cpp
    ASTDropQuery.cpp
    ASTExpressionList.cpp
    ASTFunction.cpp
    ASTFunctionWithKeyValueArguments.cpp
    ASTGrantQuery.cpp
    ASTIdentifier.cpp
    ASTIndexDeclaration.cpp
    ASTInsertQuery.cpp
    ASTKillQueryQuery.cpp
    ASTLiteral.cpp
    ASTNameTypePair.cpp
    ASTOptimizeQuery.cpp
    ASTOrderByElement.cpp
    ASTPartition.cpp
    ASTQualifiedAsterisk.cpp
    ASTQueryParameter.cpp
    ASTQueryWithOnCluster.cpp
    ASTQueryWithOutput.cpp
    ASTQueryWithTableAndOutput.cpp
    ASTRolesOrUsersSet.cpp
    ASTRowPolicyName.cpp
    ASTSampleRatio.cpp
    ASTSelectQuery.cpp
    ASTSelectWithUnionQuery.cpp
    ASTSetRoleQuery.cpp
    ASTSettingsProfileElement.cpp
    ASTShowAccessEntitiesQuery.cpp
    ASTShowCreateAccessEntityQuery.cpp
    ASTShowGrantsQuery.cpp
    ASTShowPrivilegesQuery.cpp
    ASTShowTablesQuery.cpp
    ASTSubquery.cpp
    ASTSystemQuery.cpp
    ASTTablesInSelectQuery.cpp
    ASTTTLElement.cpp
    ASTUserNameWithHost.cpp
    ASTWithAlias.cpp
    CommonParsers.cpp
    ExpressionElementParsers.cpp
    ExpressionListParsers.cpp
    formatAST.cpp
    IAST.cpp
    iostream_debug_helpers.cpp
    IParserBase.cpp
    Lexer.cpp
    makeASTForLogicalFunction.cpp
    parseDatabaseAndTableName.cpp
    parseIdentifierOrStringLiteral.cpp
    parseIntervalKind.cpp
    parseQuery.cpp
    ParserAlterQuery.cpp
    ParserCase.cpp
    ParserCheckQuery.cpp
    ParserCreateQuery.cpp
    ParserCreateQuotaQuery.cpp
    ParserCreateRoleQuery.cpp
    ParserCreateRowPolicyQuery.cpp
    ParserCreateSettingsProfileQuery.cpp
    ParserCreateUserQuery.cpp
    ParserDataType.cpp
    ParserDescribeTableQuery.cpp
    ParserDictionaryAttributeDeclaration.cpp
    ParserDictionary.cpp
    ParserDropAccessEntityQuery.cpp
    ParserDropQuery.cpp
    ParserExplainQuery.cpp
    ParserGrantQuery.cpp
    ParserInsertQuery.cpp
    ParserKillQueryQuery.cpp
    ParserOptimizeQuery.cpp
    ParserPartition.cpp
    ParserQuery.cpp
    ParserQueryWithOutput.cpp
    ParserRenameQuery.cpp
    ParserRolesOrUsersSet.cpp
    ParserRowPolicyName.cpp
    ParserSampleRatio.cpp
    ParserSelectQuery.cpp
    ParserSelectWithUnionQuery.cpp
    ParserSetQuery.cpp
    ParserSetRoleQuery.cpp
    ParserSettingsProfileElement.cpp
    ParserShowAccessEntitiesQuery.cpp
    ParserShowCreateAccessEntityQuery.cpp
    ParserShowGrantsQuery.cpp
    ParserShowPrivilegesQuery.cpp
    ParserShowTablesQuery.cpp
    ParserSystemQuery.cpp
    ParserTablePropertiesQuery.cpp
    ParserTablesInSelectQuery.cpp
    ParserUnionQueryElement.cpp
    ParserUseQuery.cpp
    ParserUserNameWithHost.cpp
    ParserWatchQuery.cpp
    parseUserName.cpp
    queryToString.cpp
    QueryWithOutputSettingsPushDownVisitor.cpp
    TokenIterator.cpp

)

END()
