#pragma once

#if USE_REPLXX
#   include <common/ReplxxLineReader.h>
#elif defined(USE_READLINE) && USE_READLINE
#   include <common/ReadlineLineReader.h>
#else
#   include <common/LineReader.h>
#endif

#include <boost/program_options.hpp>
#include <Poco/Util/Application.h>
#include <Core/Names.h>
#include <Core/Types.h>
#include <Interpreters/Context.h>
#include <Common/ProgressIndication.h>
#include <Client/Suggest.h>


namespace DB
{

class IClient : public Poco::Util::Application
{
public:
    using Arguments = std::vector<std::string>;

    int main(const std::vector<std::string> & /*args*/) override;

    int mainImpl();

    void init(int argc, char ** argv);

protected:
    NameSet exit_strings{"exit", "quit", "logout", "учше", "йгше", "дщпщге", "exit;", "quit;", "logout;", "учшеж",
                         "йгшеж", "дщпщгеж", "q", "й", "\\q", "\\Q", "\\й", "\\Й", ":q", "Жй"};

    bool is_interactive = false; /// Use either interactive line editing interface or batch mode.

    bool need_render_progress = true;
    bool written_first_block = false;
    ProgressIndication progress_indication;

    bool echo_queries = false; /// Print queries before execution in batch mode.
    bool ignore_error = false; /// In case of errors, don't print error message, continue to next query. Only applicable for non-interactive mode.
    bool print_time_to_stderr = false; /// Output execution time to stderr in batch mode.

    bool stdin_is_a_tty = false; /// stdin is a terminal.
    bool stdout_is_a_tty = false; /// stdout is a terminal.

    uint64_t terminal_width = 0;

    /// Settings specified via command line args
    Settings cmd_settings;

    /// If not empty, queries will be read from these files
    std::vector<String> queries_files;

    std::optional<Suggest> suggest;

    /// Path to a file containing command history.
    String history_file;

    String home_path;

    bool has_vertical_output_suffix = false; /// Is \G present at the end of the query string?

    String prompt_by_server_display_name;
    String server_display_name;

    static bool isNewYearMode();

    static bool isChineseNewYearMode(const String & local_tz);

#if USE_REPLXX
    static void highlight(const String & query, std::vector<replxx::Replxx::Color> & colors);
#endif

    static void clearTerminal();

    void runInteractive();

    virtual int childMainImpl() = 0;

    virtual void processMainImplException(const Exception & e) = 0;

    virtual bool isInteractive() = 0;

    virtual void loadSuggestionDataIfPossible() {}

    virtual bool processQueryFromInteractive(const String & input) = 0;

    virtual void readArguments(int argc, char ** argv,
                               Arguments & common_arguments, std::vector<Arguments> &) = 0;

    using ProgramOptionsDescription = boost::program_options::options_description;
    using CommandLineOptions = boost::program_options::variables_map;

    struct OptionsDescription
    {
        std::optional<ProgramOptionsDescription> main_description;
        std::optional<ProgramOptionsDescription> external_description;
    };

    virtual void printHelpMessage(const OptionsDescription & options_description) = 0;

    virtual void addOptions(OptionsDescription & options_description) = 0;

    virtual void processOptions(const OptionsDescription & options_description,
                                const CommandLineOptions & options,
                                const std::vector<Arguments> & external_tables_arguments) = 0;

    virtual bool supportPasswordOption() = 0;

private:
    inline String prompt() const
    {
        return boost::replace_all_copy(prompt_by_server_display_name, "{database}", config().getString("database", "default"));
    }

};

}
