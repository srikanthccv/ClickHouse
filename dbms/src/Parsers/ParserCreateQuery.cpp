#include <DB/Parsers/ASTFunction.h>
#include <DB/Parsers/ASTIdentifier.h>
#include <DB/Parsers/ASTExpressionList.h>
#include <DB/Parsers/ASTCreateQuery.h>
#include <DB/Parsers/ExpressionListParsers.h>
#include <DB/Parsers/ParserCreateQuery.h>
#include <DB/Parsers/ParserSelectQuery.h>


namespace DB
{

bool ParserNestedTable::parseImpl(Pos & pos, Pos end, ASTPtr & node, Pos & max_parsed_pos, Expected & expected)
{
	ParserWhiteSpaceOrComments ws;
	ParserString open("(");
	ParserString close(")");
	ParserIdentifier name_p;
	ParserNameTypePairList columns_p;

	ASTPtr name;
	ASTPtr columns;

	Pos begin = pos;

	/// Пока name == 'Nested', возможно потом появятся альтернативные вложенные структуры данных
	if (!name_p.parse(pos, end, name, max_parsed_pos, expected))
		return false;

	ws.ignore(pos, end);

	if (!open.ignore(pos, end))
		return false;

	ws.ignore(pos, end);

	if (!columns_p.parse(pos, end, columns, max_parsed_pos, expected))
		return false;

	ws.ignore(pos, end);

	if (!close.ignore(pos, end))
		return false;

	ASTFunction * func = new ASTFunction(StringRange(begin, pos));
	node = func;
	func->name = typeid_cast<ASTIdentifier &>(*name).name;
	func->arguments = columns;
	func->children.push_back(columns);

	return true;
}


bool ParserIdentifierWithParameters::parseImpl(Pos & pos, Pos end, ASTPtr & node, Pos & max_parsed_pos, Expected & expected)
{
	ParserFunction function_or_array;
	if (function_or_array.parse(pos, end, node, max_parsed_pos, expected))
		return true;

	ParserNestedTable nested;
	if (nested.parse(pos, end, node, max_parsed_pos, expected))
		return true;

	return false;
}


bool ParserIdentifierWithOptionalParameters::parseImpl(Pos & pos, Pos end, ASTPtr & node, Pos & max_parsed_pos, Expected & expected)
{
	ParserIdentifier non_parametric;
	ParserIdentifierWithParameters parametric;

	Pos begin = pos;

	if (parametric.parse(pos, end, node, max_parsed_pos, expected))
		return true;

	ASTPtr ident;
	if (non_parametric.parse(pos, end, ident, max_parsed_pos, expected))
	{
		ASTFunction * func = new ASTFunction(StringRange(begin, pos));
		node = func;
		func->name = typeid_cast<ASTIdentifier &>(*ident).name;
		return true;
	}

	return false;
}

bool ParserNameTypePairList::parseImpl(Pos & pos, Pos end, ASTPtr & node, Pos & max_parsed_pos, Expected & expected)
{
	return ParserList(ParserPtr(new ParserNameTypePair), ParserPtr(new ParserString(",")), false).parse(pos, end, node, max_parsed_pos, expected);
}

bool ParserColumnDeclarationList::parseImpl(Pos & pos, Pos end, ASTPtr & node, Pos & max_parsed_pos, Expected & expected)
{
	return ParserList{ParserPtr{new ParserColumnDeclaration}, ParserPtr{new ParserString{","}}, false}.parse(pos, end, node, max_parsed_pos, expected);
}


bool ParserEngine::parseImpl(Pos & pos, Pos end, ASTPtr & node, Pos & max_parsed_pos, Expected & expected)
{
	ParserWhiteSpaceOrComments ws;
	ParserString s_engine("ENGINE", true, true);
	ParserString s_eq("=");
	ParserIdentifierWithOptionalParameters storage_p;

	ws.ignore(pos, end);

	if (s_engine.ignore(pos, end, max_parsed_pos, expected))
	{
		ws.ignore(pos, end);

		if (!s_eq.ignore(pos, end, max_parsed_pos, expected))
			return false;

		ws.ignore(pos, end);

		if (!storage_p.parse(pos, end, node, max_parsed_pos, expected))
			return false;

		ws.ignore(pos, end);
	}

	return true;
}


bool ParserCreateQuery::parseImpl(Pos & pos, Pos end, ASTPtr & node, Pos & max_parsed_pos, Expected & expected)
{
	Pos begin = pos;

	ParserWhiteSpaceOrComments ws;
	ParserString s_create("CREATE", true, true);
	ParserString s_temporary("TEMPORARY", true, true);
	ParserString s_attach("ATTACH", true, true);
	ParserString s_table("TABLE", true, true);
	ParserString s_database("DATABASE", true, true);
	ParserString s_dot(".");
	ParserString s_lparen("(");
	ParserString s_rparen(")");
	ParserString s_if("IF", true, true);
	ParserString s_not("NOT", true, true);
	ParserString s_exists("EXISTS", true, true);
	ParserString s_as("AS", true, true);
	ParserString s_select("SELECT", true, true);
	ParserString s_view("VIEW", true, true);
	ParserString s_materialized("MATERIALIZED", true, true);
	ParserString s_populate("POPULATE", true, true);
	ParserEngine engine_p;
	ParserIdentifier name_p;
	ParserColumnDeclarationList columns_p;

	ASTPtr database;
	ASTPtr table;
	ASTPtr columns;
	ASTPtr storage;
	ASTPtr inner_storage;
	ASTPtr as_database;
	ASTPtr as_table;
	ASTPtr select;
	bool attach = false;
	bool if_not_exists = false;
	bool is_view = false;
	bool is_materialized_view = false;
	bool is_populate = false;
	bool is_temporary = false;

	ws.ignore(pos, end);

	if (!s_create.ignore(pos, end, max_parsed_pos, expected))
	{
		if (s_attach.ignore(pos, end, max_parsed_pos, expected))
			attach = true;
		else
			return false;
	}

	ws.ignore(pos, end);

	if (s_temporary.ignore(pos, end, max_parsed_pos, expected))
	{
		is_temporary = true;
		ws.ignore(pos, end);
	}

	if (s_database.ignore(pos, end, max_parsed_pos, expected))
	{
		ws.ignore(pos, end);

		if (s_if.ignore(pos, end, max_parsed_pos, expected)
			&& ws.ignore(pos, end)
			&& s_not.ignore(pos, end, max_parsed_pos, expected)
			&& ws.ignore(pos, end)
			&& s_exists.ignore(pos, end, max_parsed_pos, expected)
			&& ws.ignore(pos, end))
			if_not_exists = true;

		if (!name_p.parse(pos, end, database, max_parsed_pos, expected))
			return false;
	}
	else if (s_table.ignore(pos, end, max_parsed_pos, expected))
	{
		ws.ignore(pos, end);

		if (s_if.ignore(pos, end, max_parsed_pos, expected)
			&& ws.ignore(pos, end)
			&& s_not.ignore(pos, end, max_parsed_pos, expected)
			&& ws.ignore(pos, end)
			&& s_exists.ignore(pos, end, max_parsed_pos, expected)
			&& ws.ignore(pos, end))
			if_not_exists = true;

		if (!name_p.parse(pos, end, table, max_parsed_pos, expected))
			return false;

		ws.ignore(pos, end);

		if (s_dot.ignore(pos, end, max_parsed_pos, expected))
		{
			database = table;
			if (!name_p.parse(pos, end, table, max_parsed_pos, expected))
				return false;

			ws.ignore(pos, end);
		}

		/// Список столбцов
		if (s_lparen.ignore(pos, end, max_parsed_pos, expected))
		{
			ws.ignore(pos, end);

			if (!columns_p.parse(pos, end, columns, max_parsed_pos, expected))
				return false;

			ws.ignore(pos, end);

			if (!s_rparen.ignore(pos, end, max_parsed_pos, expected))
				return false;


			ws.ignore(pos, end);

			if (!engine_p.parse(pos, end, storage, max_parsed_pos, expected))
				return false;

			/// Для engine VIEW необходимо так же считать запрос AS SELECT
			if (storage && (typeid_cast<ASTFunction &>(*storage).name == "View"
						|| typeid_cast<ASTFunction &>(*storage).name == "MaterializedView"))
			{
				if (!s_as.ignore(pos, end, max_parsed_pos, expected))
					return false;
				ws.ignore(pos, end);
				Pos before_select = pos;
				if (!s_select.ignore(pos, end, max_parsed_pos, expected))
					return false;
				pos = before_select;
				ParserSelectQuery select_p;
				select_p.parse(pos, end, select, max_parsed_pos, expected);
			}
		}
		else
		{
			engine_p.parse(pos, end, storage, max_parsed_pos, expected);

			if (!s_as.ignore(pos, end, max_parsed_pos, expected))
				return false;

			ws.ignore(pos, end);

			/// AS SELECT ...
			Pos before_select = pos;
			if (s_select.ignore(pos, end, max_parsed_pos, expected))
			{
				pos = before_select;
				ParserSelectQuery select_p;
				select_p.parse(pos, end, select, max_parsed_pos, expected);
			}
			else
			{
				/// AS [db.]table
				if (!name_p.parse(pos, end, as_table, max_parsed_pos, expected))
					return false;

				ws.ignore(pos, end);

				if (s_dot.ignore(pos, end, max_parsed_pos, expected))
				{
					as_database = as_table;
					if (!name_p.parse(pos, end, as_table, max_parsed_pos, expected))
						return false;

					ws.ignore(pos, end);
				}

				ws.ignore(pos, end);

				/// Опционально - может быть указана ENGINE.
				engine_p.parse(pos, end, storage, max_parsed_pos, expected);
			}
		}
	}
	else
	{
		/// VIEW or MATERIALIZED VIEW
		if (s_materialized.ignore(pos, end, max_parsed_pos, expected) && ws.ignore(pos, end, max_parsed_pos, expected))
			is_materialized_view = true;
		else
			is_view = true;

		if (!s_view.ignore(pos, end, max_parsed_pos, expected) || !ws.ignore(pos, end, max_parsed_pos, expected))
			return false;

		if (s_if.ignore(pos, end, max_parsed_pos, expected)
			&& ws.ignore(pos, end)
			&& s_not.ignore(pos, end, max_parsed_pos, expected)
			&& ws.ignore(pos, end)
			&& s_exists.ignore(pos, end, max_parsed_pos, expected)
			&& ws.ignore(pos, end))
			if_not_exists = true;

		if (!name_p.parse(pos, end, table, max_parsed_pos, expected))
			return false;
		ws.ignore(pos, end);

		if (s_dot.ignore(pos, end, max_parsed_pos, expected))
		{
			database = table;
			if (!name_p.parse(pos, end, table, max_parsed_pos, expected))
				return false;

			ws.ignore(pos, end);
		}

		/// Опционально - может быть указан список столбцов. Он должен полностью соответствовать SELECT-у.
		if (s_lparen.ignore(pos, end, max_parsed_pos, expected))
		{
			ws.ignore(pos, end);

			if (!columns_p.parse(pos, end, columns, max_parsed_pos, expected))
				return false;

			ws.ignore(pos, end);

			if (!s_rparen.ignore(pos, end, max_parsed_pos, expected))
				return false;
		}

		/// Опционально - может быть указана внутренняя ENGINE для MATERIALIZED VIEW
		engine_p.parse(pos, end, inner_storage, max_parsed_pos, expected);

		ws.ignore(pos, end);

		if (s_populate.ignore(pos, end, max_parsed_pos, expected))
			is_populate = true;

		ws.ignore(pos, end);

		/// AS SELECT ...
		if (!s_as.ignore(pos, end, max_parsed_pos, expected))
			return false;
		ws.ignore(pos, end);
		Pos before_select = pos;
		if (!s_select.ignore(pos, end, max_parsed_pos, expected))
			return false;
		pos = before_select;
		ParserSelectQuery select_p;
		select_p.parse(pos, end, select, max_parsed_pos, expected);
	}

	ws.ignore(pos, end);

	ASTCreateQuery * query = new ASTCreateQuery(StringRange(begin, pos));
	node = query;

	query->attach = attach;
	query->if_not_exists = if_not_exists;
	query->is_view = is_view;
	query->is_materialized_view = is_materialized_view;
	query->is_populate = is_populate;
	query->is_temporary = is_temporary;

	if (database)
		query->database = typeid_cast<ASTIdentifier &>(*database).name;
	if (table)
		query->table = typeid_cast<ASTIdentifier &>(*table).name;
	if (inner_storage)
		query->inner_storage = inner_storage;

	query->columns = columns;
	query->storage = storage;
	if (as_database)
		query->as_database = typeid_cast<ASTIdentifier &>(*as_database).name;
	if (as_table)
		query->as_table = typeid_cast<ASTIdentifier &>(*as_table).name;
	query->select = select;

	if (columns)
		query->children.push_back(columns);
	if (storage)
		query->children.push_back(storage);
	if (select)
		query->children.push_back(select);
	if (inner_storage)
		query->children.push_back(inner_storage);

	return true;
}


}
