#include <DB/Parsers/ASTSelectQuery.h>
#include <DB/Parsers/ParserSelectQuery.h>
#include <DB/Parsers/parseQuery.h>
#include <DB/Parsers/queryToString.h>
#include <DB/Interpreters/LogicalExpressionsOptimizer.h>
#include <DB/Interpreters/Settings.h>

#include <iostream>
#include <vector>
#include <utility>
#include <string>

namespace
{

struct TestEntry
{
	std::string input;
	std::string expected_output;
	UInt64 limit;
};

using TestEntries = std::vector<TestEntry>;
using TestResult = std::pair<bool, std::string>;

void run();
void performTests(const TestEntries & entries);
TestResult check(const TestEntry & entry);
bool parse(DB::ASTPtr  & ast, const std::string & query);
bool equals(const DB::ASTPtr & lhs, const DB::ASTPtr & rhs);
void reorder(DB::IAST * ast);


void run()
{
	/// NOTE: Запросы не всегда реалистичные, однако лишь синтаксис нас интересует.
	TestEntries entries =
	{
		{
			"SELECT 1",
			"SELECT 1",
			3
		},

		// WHERE

		{
			"SELECT name, value FROM report WHERE (name = 'Alice') OR (name = 'Bob') OR (name = 'Carol')",
			"SELECT name, value FROM report WHERE (name = 'Alice') OR (name = 'Bob') OR (name = 'Carol')",
			4
		},

		{
			"SELECT name, value FROM report WHERE (name = 'Alice') OR (name = 'Bob') OR (name = 'Carol')",
			"SELECT name, value FROM report WHERE name IN ('Alice', 'Bob', 'Carol')",
			3
		},

		{
			"SELECT name, value FROM report WHERE (name = 'Alice') OR (name = 'Bob') OR (name = 'Carol')",
			"SELECT name, value FROM report WHERE name IN ('Alice', 'Bob', 'Carol')",
			2
		},

		{
			"SELECT name, value FROM report WHERE (name = 'Alice') OR (value = 1000) OR (name = 'Bob') OR (name = 'Carol')",
			"SELECT name, value FROM report WHERE (value = 1000) OR name IN ('Alice', 'Bob', 'Carol')",
			2
		},

		{
			"SELECT name, value FROM report WHERE (name = 'Alice') OR (value = 1000) OR (name = 'Bob') OR (name = 'Carol') OR (value = 2000)",
			"SELECT name, value FROM report WHERE name IN ('Alice', 'Bob', 'Carol') OR value IN (1000, 2000)",
			2
		},

		{
			"SELECT value FROM report WHERE ((value + 1) = 1000) OR ((2 * value) = 2000) OR ((2 * value) = 4000) OR ((value + 1) = 3000)",
			"SELECT value FROM report WHERE (value + 1) IN (1000, 3000) OR (2 * value) IN (2000, 4000)",
			2
		},

		{
			"SELECT name, value FROM report WHERE ((name = 'Alice') OR (name = 'Bob') OR (name = 'Carol')) AND ((value = 1000) OR (value = 2000))",
			"SELECT name, value FROM report WHERE name IN ('Alice', 'Bob', 'Carol') AND ((value = 1000) OR (value = 2000))",
			3
		},

		// PREWHERE

		{
			"SELECT name, value FROM report PREWHERE (name = 'Alice') OR (name = 'Bob') OR (name = 'Carol')",
			"SELECT name, value FROM report PREWHERE (name = 'Alice') OR (name = 'Bob') OR (name = 'Carol')",
			4
		},

		{
			"SELECT name, value FROM report PREWHERE (name = 'Alice') OR (name = 'Bob') OR (name = 'Carol')",
			"SELECT name, value FROM report PREWHERE name IN ('Alice', 'Bob', 'Carol')",
			3
		},

		{
			"SELECT name, value FROM report PREWHERE (name = 'Alice') OR (name = 'Bob') OR (name = 'Carol')",
			"SELECT name, value FROM report PREWHERE name IN ('Alice', 'Bob', 'Carol')",
			2
		},

		{
			"SELECT name, value FROM report PREWHERE (name = 'Alice') OR (value = 1000) OR (name = 'Bob') OR (name = 'Carol')",
			"SELECT name, value FROM report PREWHERE (value = 1000) OR name IN ('Alice', 'Bob', 'Carol')",
			2
		},

		{
			"SELECT name, value FROM report PREWHERE (name = 'Alice') OR (value = 1000) OR (name = 'Bob') OR (name = 'Carol') OR (value = 2000)",
			"SELECT name, value FROM report PREWHERE name IN ('Alice', 'Bob', 'Carol') OR value IN (1000, 2000)",
			2
		},

		{
			"SELECT value FROM report PREWHERE ((value + 1) = 1000) OR ((2 * value) = 2000) OR ((2 * value) = 4000) OR ((value + 1) = 3000)",
			"SELECT value FROM report PREWHERE (value + 1) IN (1000, 3000) OR (2 * value) IN (2000, 4000)",
			2
		},

		// HAVING

		{
			"SELECT number, count() FROM (SELECT * FROM system.numbers LIMIT 10) GROUP BY number HAVING number = 1",
			"SELECT number, count() FROM (SELECT * FROM system.numbers LIMIT 10) GROUP BY number HAVING number = 1",
			2
		},

		{
			"SELECT number, count() FROM (SELECT * FROM system.numbers LIMIT 10) GROUP BY number HAVING (number = 1) OR (number = 2)",
			"SELECT number, count() FROM (SELECT * FROM system.numbers LIMIT 10) GROUP BY number HAVING number IN (1, 2)",
			2
		},

		{
			"SELECT number, count() FROM (SELECT * FROM system.numbers LIMIT 10) GROUP BY number HAVING (number = 1) OR (number = 2)",
			"SELECT number, count() FROM (SELECT * FROM system.numbers LIMIT 10) GROUP BY number HAVING (number = 1) OR (number = 2)",
			3
		},

		{
			"SELECT number, count() FROM (SELECT * FROM system.numbers LIMIT 10) GROUP BY number HAVING ((number + 1) = 1) OR ((number + 1) = 2) OR ((number + 3) = 7)",
			"SELECT number, count() FROM (SELECT * FROM system.numbers LIMIT 10) GROUP BY number HAVING ((number + 3) = 7) OR (number + 1) IN (1, 2)",
			2
		},

		// PREWHERE + WHERE + HAVING

		{
			"SELECT number, count(), 1 AS T, 2 AS U FROM (SELECT * FROM system.numbers LIMIT 10) PREWHERE (U = 1) OR (U = 2) "
			"WHERE (T = 1) OR (T = 2) GROUP BY number HAVING (number = 1) OR (number = 2)",
			"SELECT number, count(), 1 AS T, 2 AS U FROM (SELECT * FROM system.numbers LIMIT 10) PREWHERE U IN (1, 2) "
			"WHERE T IN (1, 2) GROUP BY number HAVING number IN (1, 2)",
			2
		},

		{
			"SELECT number, count(), 1 AS T, 2 AS U FROM (SELECT * FROM system.numbers LIMIT 10) PREWHERE (U = 1) OR (U = 2) OR (U = 3) "
			"WHERE (T = 1) OR (T = 2) GROUP BY number HAVING (number = 1) OR (number = 2)",
			"SELECT number, count(), 1 AS T, 2 AS U FROM (SELECT * FROM system.numbers LIMIT 10) PREWHERE U IN (1, 2, 3) "
			"WHERE (T = 1) OR (T = 2) GROUP BY number HAVING (number = 1) OR (number = 2)",
			3
		}
	};

	performTests(entries);
}

void performTests(const TestEntries & entries)
{
	unsigned int count = 0;
	unsigned int i = 1;

	for (const auto & entry : entries)
	{
		auto res = check(entry);
		if (res.first)
		{
			++count;
			std::cout << "Test " << i << " passed.\n";
		}
		else
			std::cout << "Test " << i << " failed. Expected: " << entry.expected_output << ". Received: " << res.second << "\n";

		++i;
	}
	std::cout << count << " out of " << entries.size() << " test(s) passed.\n";
}

TestResult check(const TestEntry & entry)
{
	try
	{
		/// Парсить и оптимизировать входящий запрос.
		DB::ASTPtr ast_input;
		if (!parse(ast_input, entry.input))
			return TestResult(false, "parse error");

		auto select_query = typeid_cast<DB::ASTSelectQuery *>(&*ast_input);

		DB::Settings settings;
		settings.optimize_min_equality_disjunction_chain_length = entry.limit;

		DB::LogicalExpressionsOptimizer optimizer(select_query, settings);
		optimizer.optimizeDisjunctiveEqualityChains();

		/// Парсить ожидаемый результат.
		DB::ASTPtr ast_expected;
		if (!parse(ast_expected, entry.expected_output))
			return TestResult(false, "parse error");

		/// Сравнить оптимизированный запрос и ожидаемый результат.
		bool res = equals(ast_input, ast_expected);
		std::string output = DB::queryToString(ast_input);

		return TestResult(res, output);
	}
	catch (DB::Exception & e)
	{
		return TestResult(false, e.displayText());
	}
}

bool parse(DB::ASTPtr & ast, const std::string & query)
{
	DB::ParserSelectQuery parser;
	std::string message;
	ast = DB::tryParseQuery(parser, query.data(), query.data() + query.size(), message, false, "");
	return !ast.isNull();
}

bool equals(const DB::ASTPtr & lhs, const DB::ASTPtr & rhs)
{
	DB::ASTPtr lhs_reordered = lhs->clone();
	reorder(&*lhs_reordered);

	DB::ASTPtr rhs_reordered = rhs->clone();
	reorder(&*rhs_reordered);

	return lhs_reordered->getTreeID() == rhs_reordered->getTreeID();
}

void reorder(DB::IAST * ast)
{
	if (ast == nullptr)
		return;

	auto & children = ast->children;
	if (children.empty())
		return;

	for (auto & child : children)
		reorder(&*child);

	std::sort(children.begin(), children.end(), [](const DB::ASTPtr & lhs, const DB::ASTPtr & rhs)
	{
		return lhs->getTreeID() < rhs->getTreeID();
	});
}

}

int main()
{
	run();
	return 0;
}
