#include <Functions/geometryFromColumn.h>
#include <DataTypes/DataTypeCustomGeo.h>


namespace DB {

namespace {

size_t getArrayDepth(DataTypePtr data_type, size_t max_depth)
{
    LOG_FATAL(&Poco::Logger::get("geometryFromColumn"), "start get depth");
    size_t depth = 0;
    while (data_type && isArray(data_type) && depth != max_depth + 1)
    {
        LOG_FATAL(&Poco::Logger::get("geometryFromColumn"), data_type->getName());
        depth++;
        data_type = static_cast<const DataTypeArray &>(*data_type).getNestedType();
    }
    LOG_FATAL(&Poco::Logger::get("geometryFromColumn"), "End get depth");
    return depth;
}

class ContainerCreator : public boost::static_visitor<Float64Geometry>
{
public:
    template <class T>
    Float64Geometry operator()(const T & parser) const
    {
        return parser.createContainer();
    }
};

class Getter : public boost::static_visitor<void>
{
public:
    Getter(Float64Geometry & container_, size_t i_)
        : container(container_)
        , i(i_)
    {}

    template <class T>
    void operator()(const T & parser) const
    {
        parser.get(container, i);
    }

private:
    Float64Geometry & container;
    size_t i;
};

template <class DataType, class Parser>
Parser makeParser(const ColumnWithTypeAndName & col)
{
    auto wanted_data_type = DataType::nestedDataType();
    ColumnPtr casted = castColumn(col,
        std::make_shared<DataTypeTuple>(DataTypes{
            std::make_shared<DataTypeFloat64>(),
            std::make_shared<DataTypeFloat64>()}));
    LOG_FATAL(&Poco::Logger::get("geometryFromColumn"), col.type->getName() + " to " + wanted_data_type->getName());
    if (!casted)
    {
        throw Exception("Failed to cast " + col.type->getName() + " to " + wanted_data_type->getName(), ErrorCodes::ILLEGAL_COLUMN);
    }

    return Parser(*casted);
}

}

GeometryFromColumnParser makeGeometryFromColumnParser(const ColumnWithTypeAndName & col)
{
    switch (getArrayDepth(col.type, 3)) {
        case 0: return makeParser<DataTypeCustomPointSerialization, Float64PointFromColumnParser>(col);
        case 1: return makeParser<DataTypeCustomRingSerialization, Float64RingFromColumnParser>(col);
        case 2: return makeParser<DataTypeCustomPolygonSerialization, Float64PolygonFromColumnParser>(col);
        case 3: return makeParser<DataTypeCustomMultiPolygonSerialization, Float64MultiPolygonFromColumnParser>(col);
        default: throw Exception("Cannot parse geometry from column with type " + col.type->getName()
                + ", array depth is too big", ErrorCodes::ILLEGAL_COLUMN);
    }
}

Float64Geometry createContainer(const GeometryFromColumnParser & parser)
{
    static ContainerCreator creator;
    return boost::apply_visitor(creator, parser);
}

void get(const GeometryFromColumnParser & parser, Float64Geometry & container, size_t i)
{
    boost::apply_visitor(Getter(container, i), parser);
}

}
