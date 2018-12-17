#include <Compression/CompressionFactory.h>
#include <Parsers/parseQuery.h>
#include <Parsers/ParserCreateQuery.h>
#include <Parsers/ASTFunction.h>
#include <Parsers/ASTIdentifier.h>
#include <Parsers/ASTLiteral.h>
#include <Common/typeid_cast.h>
#include <Poco/String.h>
#include <IO/ReadBuffer.h>
#include <Parsers/queryToString.h>
#include <Compression/CompressionCodecMultiple.h>
#include <Compression/CompressionCodecLZ4.h>
#include <Compression/CompressionCodecNone.h>
#include <IO/WriteHelpers.h>


namespace DB
{
namespace ErrorCodes
{
    extern const int UNKNOWN_CODEC;
    extern const int UNEXPECTED_AST_STRUCTURE;
    extern const int ILLEGAL_SYNTAX_FOR_CODEC_TYPE;
    extern const int DATA_TYPE_CANNOT_HAVE_ARGUMENTS;
}

CompressionCodecPtr CompressionCodecFactory::getDefaultCodec() const
{
    return default_codec;
}

CompressionCodecPtr CompressionCodecFactory::get(const ASTPtr & ast) const
{
    if (const auto * func = typeid_cast<const ASTFunction *>(ast.get()))
    {
        Codecs codecs;
        codecs.reserve(func->arguments->children.size());
        for (const auto & inner_codec_ast : func->arguments->children)
        {
            if (const auto * family_name = typeid_cast<const ASTIdentifier *>(inner_codec_ast.get()))
                codecs.emplace_back(getImpl(family_name->name, {}));
            else if (const auto * ast_func = typeid_cast<const ASTFunction *>(inner_codec_ast.get()))
                codecs.emplace_back(getImpl(ast_func->name, ast_func->arguments));
            else
                throw Exception("Unexpected AST element for compression codec", ErrorCodes::UNEXPECTED_AST_STRUCTURE);
        }

        if (codecs.size() == 1)
            return codecs.back();
        else if (codecs.size() > 1)
            return std::make_shared<CompressionCodecMultiple>(codecs);
    }

    throw Exception("Unknown codec family: " + queryToString(ast), ErrorCodes::UNKNOWN_CODEC);
}

CompressionCodecPtr CompressionCodecFactory::get(const UInt8 byte_code) const
{
    const auto family_code_and_creator = family_code_with_codec.find(byte_code);

    if (family_code_and_creator == family_code_with_codec.end())
        throw Exception("Unknown codec family code : " + toString(byte_code), ErrorCodes::UNKNOWN_CODEC);

    return family_code_and_creator->second({});
}

CompressionCodecPtr CompressionCodecFactory::getImpl(const String & family_name, const ASTPtr & arguments) const
{
    if (family_name == "MULTIPLE")
        throw Exception("Codec MULTIPLE cannot be specified directly", ErrorCodes::UNKNOWN_CODEC);

    const auto family_and_creator = family_name_with_codec.find(family_name);

    if (family_and_creator == family_name_with_codec.end())
        throw Exception("Unknown codec family: " + family_name, ErrorCodes::UNKNOWN_CODEC);

    return family_and_creator->second(arguments);
}

void CompressionCodecFactory::registerCompressionCodec(const String & family_name, UInt8 byte_code, Creator creator)
{
    if (creator == nullptr)
        throw Exception("CompressionCodecFactory: the codec family " + family_name + " has been provided a null constructor",
                        ErrorCodes::LOGICAL_ERROR);

    if (!family_name_with_codec.emplace(family_name, creator).second)
        throw Exception("CompressionCodecFactory: the codec family name '" + family_name + "' is not unique", ErrorCodes::LOGICAL_ERROR);

    if (!family_code_with_codec.emplace(byte_code, creator).second)
        throw Exception("CompressionCodecFactory: the codec family name '" + family_name + "' is not unique", ErrorCodes::LOGICAL_ERROR);
}

void CompressionCodecFactory::registerSimpleCompressionCodec(const String & family_name, UInt8 byte_code,
                                                                 std::function<CompressionCodecPtr()> creator)
{
    registerCompressionCodec(family_name, byte_code, [family_name, creator](const ASTPtr & ast)
    {
        if (ast)
            throw Exception("Compression codec " + family_name + " cannot have arguments", ErrorCodes::DATA_TYPE_CANNOT_HAVE_ARGUMENTS);
        return creator();
    });
}

void registerCodecLZ4(CompressionCodecFactory & factory);
void registerCodecNone(CompressionCodecFactory & factory);
void registerCodecZSTD(CompressionCodecFactory & factory);
void registerCodecMultiple(CompressionCodecFactory & factory);
//void registerCodecDelta(CompressionCodecFactory & factory);

CompressionCodecFactory::CompressionCodecFactory()
{
    default_codec = std::make_shared<CompressionCodecLZ4>();
    registerCodecLZ4(*this);
    registerCodecNone(*this);
    registerCodecZSTD(*this);
    registerCodecMultiple(*this);
//    registerCodecDelta(*this);
}

}
