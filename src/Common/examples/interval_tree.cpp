#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <set>
#include <vector>

#include <Common/randomSeed.h>
#include <Common/Stopwatch.h>
#include <Common/IntervalTree.h>

using namespace DB;
using Int64Interval = Interval<Int64>;

struct CollectIntervalsCallback
{
    explicit CollectIntervalsCallback(std::set<Int64Interval> & result_intervals_)
        : result_intervals(result_intervals_)
    {
    }

    bool operator()(Int64Interval interval)
    {
        result_intervals.insert(interval);
        return true;
    }

    std::set<Int64Interval> & result_intervals;
};

void findNecessaryIntervals(const std::vector<Int64Interval> & intervals, Int64 point, std::set<Int64Interval> & result_intervals)
{
    for (const auto & interval : intervals)
    {
        if (interval.contains(point))
            result_intervals.insert(interval);
    }
}

int main(int, char **)
{
    {
        IntervalSet<Int64Interval> tree;

        tree.emplace(Int64Interval(0, 5));
        tree.emplace(Int64Interval(10, 15));

        tree.construct();

        for (const auto & interval : tree)
        {
            std::cout << "Interval left " << interval.left << " right " << interval.right << std::endl;
        }
    }
    {
        IntervalMap<Int64Interval, std::string> tree;

        tree.emplace(Int64Interval(0, 5), "value1");
        tree.emplace(Int64Interval(10, 15), "value2");

        tree.construct();

        for (const auto & [interval, value] : tree)
        {
            std::cout << "Interval left " << interval.left << " right " << interval.right;
            std::cout << " value " << value << std::endl;
        }
    }
    {
        IntervalSet<Int64Interval> tree;
        for (size_t i = 0; i < 5; ++i)
        {
            tree.emplace(Int64Interval(0, i));
        }

        tree.construct();

        for (const auto & interval : tree)
        {
            std::cout << "Interval left " << interval.left << " right " << interval.right << std::endl;
        }

        for (Int64 i = 0; i < 5; ++i)
        {
            tree.find(i, [](auto & interval)
            {
                std::cout << "Interval left " << interval.left << " right " << interval.right << std::endl;
                return true;
            });
        }
    }
    {
        IntervalMap<Int64Interval, std::string> tree;
        for (size_t i = 0; i < 5; ++i)
        {
            tree.emplace(Int64Interval(0, i), "Value " + std::to_string(i));
        }

        tree.construct();

        for (const auto & [interval, value] : tree)
        {
            std::cout << "Interval left " << interval.left << " right " << interval.right;
            std::cout << " value " << value << std::endl;
        }

        for (Int64 i = 0; i < 5; ++i)
        {
            tree.find(i, [](auto & interval, auto & value)
            {
                std::cout << "Interval left " << interval.left << " right " << interval.right;
                std::cout << " value " << value << std::endl;

                return true;
            });
        }
    }

    return 0;
}
