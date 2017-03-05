#ifndef IO_UTILITY_H_
#define IO_UTILITY_H_



#include <fstream>
#include <string>
#include <ctime>


std::string  get_time_and_data() {
    time_t rawtime;
    struct tm * timeinfo;
    char buffer[80];

    time (&rawtime);
    timeinfo = localtime(&rawtime);

    strftime(buffer,sizeof(buffer),"%d-%m-%Y %I:%M:%S",timeinfo);
    return std::string(buffer);
}











#endif // IO_UTILITY_H_
