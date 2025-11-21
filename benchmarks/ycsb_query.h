#pragma once

#include "global.h"
#include "helper.h"
#include "query.h"
#include <queue>

class workload;

struct RequestYCSB {
public:
    access_t rtype;
    uint64_t key;
    uint32_t value;
};

class QueryYCSB : public QueryBase {
public:
    static void calculateDenom();

    QueryYCSB();
    QueryYCSB(char * raw_data);
    QueryYCSB(RequestYCSB * requests, uint32_t num_requests);
    ~QueryYCSB();

    uint32_t serialize(char * &raw_data);

    uint64_t get_request_count()     { return _request_cnt; }
    RequestYCSB * get_requests()    { return _requests; }
    float * get_query_vector() { return query_vector; }
    int get_query_id() { return query_id; }
    std::vector<int32_t> get_knn_indices(); 
    void add_to_list(float * vector, int index); 
    void gen_requests();
    bool is_all_remote_readonly() { return _is_all_remote_readonly; }

private:
    uint32_t _request_cnt;
    int query_id; 
    RequestYCSB * _requests;
    float* query_vector;
    std::priority_queue<std::pair<float,int>> heap;  // max-heap

    // for Zipfian distribution
    uint64_t zipf(uint64_t n, double theta);
    static double zeta(uint64_t n, double theta);
    static uint64_t the_n;
    static double denom;
    static double zeta_2_theta;
    bool _is_all_remote_readonly;
};
