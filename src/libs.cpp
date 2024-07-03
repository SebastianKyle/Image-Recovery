#include "libs.h"

bool str_compare(const char* a, std::string b) {
    if (strlen(a) != b.length()) {
        return 0;
    }

    for (int i = 0; i < strlen(a); i++) {
        if (a[i] != b[i]) return 0;
    }

    return 1;
}

double char_2_double(char* argv[], int n) {
    double temp = std::stod(argv[n], NULL);
    return temp;
}

int char_2_int(char* argv[], int n) {
    int temp = std::stoi(argv[n], NULL);
    return temp;
}