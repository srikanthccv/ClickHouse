#if defined(OS_LINUX)

#include <unistd.h>
#include <cassert>
#include <string>
#include <ctime>

#include "ProcessorStatisticsOS.h"


#include <Core/Types.h>

#include <common/logger_useful.h>

#include <Common/Exception.h>

#include <IO/ReadBufferFromFile.h>
#include <IO/ReadHelpers.h>

namespace DB
{

namespace ErrorCodes
{
    extern const int FILE_DOESNT_EXIST;
    extern const int CANNOT_OPEN_FILE;
    extern const int CANNOT_READ_FROM_FILE_DESCRIPTOR;
    extern const int CANNOT_CLOSE_FILE;
}

static constexpr auto loadavg_filename = "/proc/loadavg";
static constexpr auto procst_filename  = "/proc/stat";
static constexpr auto cpuinfo_filename = "/proc/cpuinfo";

static const long USER_HZ = sysconf(_SC_CLK_TCK);

ProcessorStatisticsOS::ProcessorStatisticsOS()
    : loadavg_in(loadavg_filename, DBMS_DEFAULT_BUFFER_SIZE, O_RDONLY | O_CLOEXEC)
    , procst_in(procst_filename,   DBMS_DEFAULT_BUFFER_SIZE, O_RDONLY | O_CLOEXEC)
    , cpuinfo_in(cpuinfo_filename, DBMS_DEFAULT_BUFFER_SIZE, O_RDONLY | O_CLOEXEC)
{
    ProcStLoad unused;
    calcStLoad(unused);
}

ProcessorStatisticsOS::~ProcessorStatisticsOS() {}

ProcessorStatisticsOS::Data ProcessorStatisticsOS::ProcessorStatisticsOS::get()
{
    Data data;
    readLoadavg(data.loadavg);
    calcStLoad(data.stload);
    readFreq(data.freq); 
    return data;
}

void ProcessorStatisticsOS::readLoadavg(ProcLoadavg& loadavg)
{
    loadavg_in.seek(0, SEEK_SET);
    
    readFloatAndSkipWhitespaceIfAny(loadavg.avg1,  loadavg_in);
    readFloatAndSkipWhitespaceIfAny(loadavg.avg5,  loadavg_in);
    readFloatAndSkipWhitespaceIfAny(loadavg.avg15, loadavg_in);
}

void ProcessorStatisticsOS::calcStLoad(ProcStLoad & stload) 
{
    ProcTime cur_proc_time;
    readProcTimeAndProcesses(cur_proc_time, stload);

    std::time_t cur_time = std::time(nullptr);
    float time_dif = static_cast<float>(cur_time - last_stload_call_time);

    stload.user_time = 
        (cur_proc_time.user - last_proc_time.user) / time_dif;
    stload.nice_time = 
        (cur_proc_time.nice - last_proc_time.nice) / time_dif;
    stload.system_time = 
        (cur_proc_time.system - last_proc_time.system) / time_dif;
    stload.idle_time = 
        (cur_proc_time.idle - last_proc_time.idle) / time_dif;
    stload.iowait_time = 
        (cur_proc_time.iowait - last_proc_time.iowait) / time_dif;
    stload.steal_time = 
       (cur_proc_time.steal - last_proc_time.steal) / time_dif;
    stload.guest_time = 
       (cur_proc_time.guest - last_proc_time.guest) / time_dif;
    stload.guest_nice_time = 
        (cur_proc_time.guest_nice - last_proc_time.guest_nice) / time_dif;
    
    last_stload_call_time = cur_time;
    last_proc_time = cur_proc_time;
}

void ProcessorStatisticsOS::readProcTimeAndProcesses(ProcTime & proc_time, ProcStLoad& stload)
{
    procst_in.seek(0, SEEK_SET);

    String field_name, field_val;
    uint64_t unused; 
   
    readStringUntilWhitespaceAndSkipWhitespaceIfAny(field_name, procst_in);

    readIntTextAndSkipWhitespaceIfAny(proc_time.user,   procst_in);
    readIntTextAndSkipWhitespaceIfAny(proc_time.nice,   procst_in);
    readIntTextAndSkipWhitespaceIfAny(proc_time.system, procst_in);
    readIntTextAndSkipWhitespaceIfAny(proc_time.idle,   procst_in);
    readIntTextAndSkipWhitespaceIfAny(proc_time.iowait, procst_in);
    proc_time.user   /= USER_HZ;
    proc_time.nice   /= USER_HZ;
    proc_time.system /= USER_HZ;
    proc_time.idle   /= USER_HZ;
    proc_time.iowait /= USER_HZ;
    
    readIntTextAndSkipWhitespaceIfAny(unused, procst_in);
    readIntTextAndSkipWhitespaceIfAny(unused, procst_in);
    
    readIntTextAndSkipWhitespaceIfAny(proc_time.steal,      procst_in);
    readIntTextAndSkipWhitespaceIfAny(proc_time.guest,      procst_in);
    readIntTextAndSkipWhitespaceIfAny(proc_time.guest_nice, procst_in);
    proc_time.steal      /= USER_HZ;
    proc_time.guest      /= USER_HZ;
    proc_time.guest_nice /= USER_HZ;

    do 
    {
        readStringUntilWhitespaceAndSkipWhitespaceIfAny(field_name, procst_in);
        readStringUntilWhitespaceAndSkipWhitespaceIfAny(field_val,  procst_in);
    } while (field_name != String("processes"));
    
    stload.processes = static_cast<uint32_t>(std::stoul(field_val));
    
    readStringUntilWhitespaceAndSkipWhitespaceIfAny(field_name, procst_in);
    readIntTextAndSkipWhitespaceIfAny(stload.procs_running, procst_in);
    
    readStringUntilWhitespaceAndSkipWhitespaceIfAny(field_name, procst_in);
    readIntTextAndSkipWhitespaceIfAny(stload.procs_blocked, procst_in);
}

void ProcessorStatisticsOS::readFreq(ProcFreq & freq)
{   
    cpuinfo_in.seek(0, SEEK_SET);
    
    String field_name, field_val;
    char unused;
    int cpu_count = 0;

    do
    {
        do 
        {
            readStringUntilWhitespaceAndSkipWhitespaceIfAny(field_name, cpuinfo_in);
        } while (!cpuinfo_in.eof() && field_name != String("cpu MHz"));
        
        if (cpuinfo_in.eof()) 
            break;

        readCharAndSkipWhitespaceIfAny(unused, cpuinfo_in);
        readStringUntilWhitespaceAndSkipWhitespaceIfAny(field_val,  cpuinfo_in);

        cpu_count++;
        
        float cur_cpu_freq = stof(field_val);

        freq.avg += cur_cpu_freq;
        freq.max = (cpu_count == 1 ? cur_cpu_freq : 
                                          std::max(freq.max, cur_cpu_freq));
        freq.min = (cpu_count == 1 ? cur_cpu_freq : 
                                          std::min(freq.min, cur_cpu_freq));
    } while (true);

    freq.avg /= static_cast<float>(cpu_count);
}

template<typename T>
void ProcessorStatisticsOS::readIntTextAndSkipWhitespaceIfAny(T& x, ReadBuffer& buf)
{
    readIntText(x, buf);
    skipWhitespaceIfAny(buf);
}

void ProcessorStatisticsOS::readStringUntilWhitespaceAndSkipWhitespaceIfAny(String & s, ReadBuffer & buf)
{
    readStringUntilWhitespace(s, buf);
    skipWhitespaceIfAny(buf);
}

void ProcessorStatisticsOS::readCharAndSkipWhitespaceIfAny(char & c, ReadBuffer & buf)
{
    readChar(c, buf);
    skipWhitespaceIfAny(buf);
}

void ProcessorStatisticsOS::readFloatAndSkipWhitespaceIfAny(float & f, ReadBuffer & buf)
{
    readFloatText(f, buf);
    skipWhitespaceIfAny(buf);
}

}

#endif
